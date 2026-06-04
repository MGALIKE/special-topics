// cudaBackend.js — CUDA backend for the FDTD time loop.
//
// Implements the Backend contract (src/backends/CONTRACT.md): runs updateH /
// updateE / CPML / source injection / sampling on an NVIDIA GPU through the
// native addon in ./native (built via `npm run build:cuda`).
//
// SAFETY: isAvailable() must NEVER throw. If the addon did not compile or no
// CUDA device is present, it returns false and the registry falls back to
// wasm-cpu.
//
// PRECISION: the reference engine is f64. The GPU is f32. We cast on upload and
// validate sampled traces within f32 tolerance (CONTRACT §1, §4).
//
// FAR-FIELD: the DFT accumulation (src/farfield.js) is intentionally kept on the
// CPU here. It needs E and H over the Huygens surface every step; rather than
// porting it to CUDA we read the 6 fields back per step into f32 mirrors and
// call accumulateFarfieldDFT. This is a CORRECTNESS-FIRST fallback and is the
// main performance cost of this backend (clearly marked TODO: port to GPU).
// If ff.nFreq === 0 the readback is skipped entirely.

import { createRequire } from 'module';
import path from 'path';
import { fileURLToPath } from 'url';
import { accumulateFarfieldDFT } from '../../farfield.js';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Try a few conventional node-gyp output locations.
const ADDON_CANDIDATES = [
  path.join(__dirname, 'build', 'Release', 'fdtd_cuda.node'),
  path.join(__dirname, 'build', 'Debug', 'fdtd_cuda.node'),
  path.join(__dirname, 'native', 'build', 'Release', 'fdtd_cuda.node'),
];

let _addon = null;
let _addonTried = false;

function loadAddon() {
  if (_addonTried) return _addon;
  _addonTried = true;
  for (const p of ADDON_CANDIDATES) {
    try {
      _addon = require(p);
      return _addon;
    } catch { /* keep trying */ }
  }
  _addon = null;
  return null;
}

// ── coeff id mapping (must match fdtd_cuda.cu) ──────────────────────────────
const COEFF_ORDER = [
  'Cexe','Cexhz','Cexhy',
  'Ceye','Ceyhx','Ceyhz',
  'Ceze','Cezhy','Cezhx',
  'Chxh','Chxey','Chxez',
  'Chyh','Chyez','Chyex',
  'Chzh','Chzex','Chzey',
];

const DIR_ID = { x: 0, y: 1, z: 2 };

// f64 (or any) typed array -> fresh Float32Array copy.
function toF32(arr) {
  if (arr instanceof Float32Array) return arr;
  const out = new Float32Array(arr.length);
  out.set(arr);
  return out;
}

// ── CPML face packing ───────────────────────────────────────────────────────
// The native side stores 4 (Psi,CPsi) slots per face. The cpml.js face objects
// name their arrays per axis; we map them onto the generic slots in the SAME
// order the .cu kernels consume them.
function packCpmlFace(face, faceId) {
  // Returns the object binding.cpp/uploadCpml expects, or null if face absent.
  if (!face) return null;
  const common = {
    nc: face.nc,
    b_e: toF32(face.b_e), a_e: toF32(face.a_e),
    b_m: toF32(face.b_m), a_m: toF32(face.a_m),
    ascending: face.ascending ? 1 : 0,
  };
  // X faces (0=xn,1=xp): m_start=i_m_start, e_start=i_e_start.
  // mag: CPsi0=CPsi_hyx, CPsi1=CPsi_hzx; ele: CPsi2=CPsi_eyx, CPsi3=CPsi_ezx.
  if (faceId === 0 || faceId === 1) {
    return {
      ...common,
      m_start: face.i_m_start,
      e_start: face.i_e_start,
      CPsi0: toF32(face.CPsi_hyx), CPsi1: toF32(face.CPsi_hzx),
      CPsi2: toF32(face.CPsi_eyx), CPsi3: toF32(face.CPsi_ezx),
    };
  }
  // Y faces (2=yn,3=yp): mag CPsi0=CPsi_hxy, CPsi1=CPsi_hzy;
  //                      ele CPsi2=CPsi_ezy, CPsi3=CPsi_exy.
  if (faceId === 2 || faceId === 3) {
    return {
      ...common,
      m_start: face.j_m_start,
      e_start: face.j_e_start,
      CPsi0: toF32(face.CPsi_hxy), CPsi1: toF32(face.CPsi_hzy),
      CPsi2: toF32(face.CPsi_ezy), CPsi3: toF32(face.CPsi_exy),
    };
  }
  // Z faces (4=zn,5=zp): mag CPsi0=CPsi_hxz, CPsi1=CPsi_hyz;
  //                      ele CPsi2=CPsi_exz, CPsi3=CPsi_eyz.
  return {
    ...common,
    m_start: face.k_m_start,
    e_start: face.k_e_start,
    CPsi0: toF32(face.CPsi_hxz), CPsi1: toF32(face.CPsi_hyz),
    CPsi2: toF32(face.CPsi_exz), CPsi3: toF32(face.CPsi_eyz),
  };
}

// ── sampled-current weighted gather lists ───────────────────────────────────
// Reproduces sampling.js::captureSampledCurrents as 3 (idx, weight) lists, one
// per H component. weight already folds dx/dy/dz, the ± of each path segment,
// and the overall 'n' direction sign. sum over all 3 lists == sampled current.
function buildCurrentLists(obs, grid) {
  const { dx, dy, dz, ny, nz, nyp1, nzp1 } = grid;
  const idxHx = (i,j,k) => i*ny*nz + j*nz + k;
  const idxHy = (i,j,k) => i*nyp1*nz + j*nz + k;
  const idxHz = (i,j,k) => i*ny*nzp1 + j*nzp1 + k;

  const hx = { idx: [], w: [] };
  const hy = { idx: [], w: [] };
  const hz = { idx: [], w: [] };
  const sign = obs.direction[1] === 'n' ? -1 : 1;

  const I = obs.is-1, J = obs.js-1, K = obs.ks-1;
  const IE = obs.ie-1, JE = obs.je-1, KE = obs.ke-1;

  switch (obs.direction[0]) {
    case 'x': {
      const ix = IE - 1;
      for (let j = J; j <= JE-1; j++) { hy.idx.push(idxHy(ix, j, K-1)); hy.w.push(sign*dy); }
      for (let k = K; k <= KE-1; k++) { hz.idx.push(idxHz(ix, JE-1, k)); hz.w.push(sign*dz); }
      for (let j = J; j <= JE-1; j++) { hy.idx.push(idxHy(ix, j, KE-1)); hy.w.push(-sign*dy); }
      for (let k = K; k <= KE-1; k++) { hz.idx.push(idxHz(ix, J-1, k)); hz.w.push(-sign*dz); }
      break;
    }
    case 'y': {
      const jx = JE - 1;
      for (let k = K; k <= KE-1; k++) { hz.idx.push(idxHz(I-1, jx, k)); hz.w.push(sign*dz); }
      for (let i = I; i <= IE-1; i++) { hx.idx.push(idxHx(i, jx, KE-1)); hx.w.push(sign*dx); }
      for (let k = K; k <= KE-1; k++) { hz.idx.push(idxHz(IE, jx, k)); hz.w.push(-sign*dz); }
      for (let i = I; i <= IE-1; i++) { hx.idx.push(idxHx(i, jx, K-1)); hx.w.push(-sign*dx); }
      break;
    }
    case 'z': {
      const kx = KE - 1;
      for (let i = I; i <= IE-1; i++) { hx.idx.push(idxHx(i, J-1, kx)); hx.w.push(sign*dx); }
      for (let j = J; j <= JE-1; j++) { hy.idx.push(idxHy(IE, j, kx)); hy.w.push(sign*dy); }
      for (let i = I; i <= IE-1; i++) { hx.idx.push(idxHx(i, JE, kx)); hx.w.push(-sign*dx); }
      for (let j = J; j <= JE-1; j++) { hy.idx.push(idxHy(I-1, j, kx)); hy.w.push(-sign*dy); }
      break;
    }
  }
  const pack = (l) => ({ idx: Int32Array.from(l.idx), w: Float32Array.from(l.w) });
  return { hx: pack(hx), hy: pack(hy), hz: pack(hz) };
}

const backend = {
  name: 'cuda',

  async isAvailable() {
    try {
      const addon = loadAddon();
      if (!addon || typeof addon.deviceCount !== 'function') return false;
      return addon.deviceCount() > 0;
    } catch {
      return false; // never throw — fall back to wasm-cpu
    }
  },

  meta() {
    return { name: 'cuda', precision: 'f32', farfield: 'cpu-fallback' };
  },

  async *run(problem) {
    const addon = loadAddon();
    if (!addon) throw new Error('cuda addon not built — run `npm run build:cuda`');

    const { grid, coeffs, cpml, samplers, sources, ff, options = {} } = problem;
    const batchSize = options.batchSize ?? 50;
    const { nx, ny, nz, nxp1, nyp1, nzp1, dx, dy, dz, dt, numberOfTimeSteps } = grid;

    const { sampledVoltages = [], sampledCurrents = [] } = samplers;
    const { voltageSources = [] } = sources;

    // ── 1. init device + upload everything once ──────────────────────────────
    addon.init({ nx, ny, nz, dx, dy, dz, dt, numberOfTimeSteps });
    try {
      // coefficients
      for (let i = 0; i < COEFF_ORDER.length; i++) {
        addon.uploadCoeff(i, toF32(coeffs[COEFF_ORDER[i]]));
      }
      // CPML faces (skip absent)
      const faceNames = ['xn','xp','yn','yp','zn','zp'];
      for (let fid = 0; fid < 6; fid++) {
        const packed = packCpmlFace(cpml[faceNames[fid]], fid);
        if (packed) addon.uploadCpml(fid, packed);
      }
      // voltage source (default problem: exactly one)
      let vsActive = false;
      if (voltageSources.length > 0) {
        const vs = voltageSources[0];
        const dir = vs.direction[0];
        const coefArr = dir === 'x' ? vs.Cexs : dir === 'y' ? vs.Ceys : vs.Cezs;
        addon.uploadVSource(
          DIR_ID[dir],
          Int32Array.from(vs.field_indices),
          toF32(coefArr),
          toF32(vs.voltage_per_e_field),
        );
        vsActive = true;
        if (voltageSources.length > 1) {
          console.warn('[cuda] only the first voltage source is uploaded (default problem has one).');
        }
      }
      // sampled voltage (first observer drives ProgressSnapshot.voltage)
      if (sampledVoltages.length > 0) {
        const sv = sampledVoltages[0];
        addon.uploadSVoltage(DIR_ID[sv.direction[0]], Int32Array.from(sv.field_indices), sv.Csvf);
      }
      // sampled current
      let siActive = false;
      if (sampledCurrents.length > 0) {
        const lists = buildCurrentLists(sampledCurrents[0], grid);
        addon.uploadSCurrent(lists.hx, lists.hy, lists.hz);
        siActive = true;
      }

      // Far-field CPU fallback: allocate f32 field mirrors only if needed.
      const wantFF = ff && ff.nFreq > 0;
      let mirror = null;
      if (wantFF) {
        mirror = {
          Hx: new Float32Array(nxp1*ny*nz),
          Hy: new Float32Array(nx*nyp1*nz),
          Hz: new Float32Array(nx*ny*nzp1),
          Ex: new Float32Array(nx*nyp1*nzp1),
          Ey: new Float32Array(nxp1*ny*nzp1),
          Ez: new Float32Array(nxp1*nyp1*nz),
        };
      }

      // ── 2. batch loop ──────────────────────────────────────────────────────
      const startTime = Date.now();

      if (wantFF) {
        // Far-field needs per-step field snapshots → run one step at a time and
        // read back the 6 fields. Correctness over speed (documented fallback).
        for (let ts = 0; ts < numberOfTimeSteps; ts++) {
          const { voltage, current } = addon.runBatch(ts, 1);
          if (sampledVoltages.length > 0) sampledVoltages[0].sampled_value[ts] = voltage[0];
          if (siActive) sampledCurrents[0].sampled_value[ts] = current[0];

          addon.readField(0, mirror.Hx); addon.readField(1, mirror.Hy); addon.readField(2, mirror.Hz);
          addon.readField(3, mirror.Ex); addon.readField(4, mirror.Ey); addon.readField(5, mirror.Ez);
          accumulateFarfieldDFT(ff, mirror, grid, ts);

          if ((ts + 1) % batchSize === 0 || ts === numberOfTimeSteps - 1) {
            yield snapshot(ts, numberOfTimeSteps, startTime, sampledVoltages, dt);
            await new Promise(r => setTimeout(r, 0));
          }
        }
      } else {
        // Fast path: no far-field. Run in batches entirely on the GPU.
        for (let start = 0; start < numberOfTimeSteps; start += batchSize) {
          const count = Math.min(batchSize, numberOfTimeSteps - start);
          const { voltage, current } = addon.runBatch(start, count);
          for (let s = 0; s < count; s++) {
            const ts = start + s;
            if (sampledVoltages.length > 0) sampledVoltages[0].sampled_value[ts] = voltage[s];
            if (siActive) sampledCurrents[0].sampled_value[ts] = current[s];
          }
          const lastTs = start + count - 1;
          yield snapshot(lastTs, numberOfTimeSteps, startTime, sampledVoltages, dt);
          await new Promise(r => setTimeout(r, 0));
        }
      }
    } finally {
      addon.destroy();
    }
  },
};

function snapshot(ts, total, startTime, sampledVoltages, dt) {
  const voltage = sampledVoltages.length > 0 ? sampledVoltages[0].sampled_value[ts] : null;
  const elapsed = (Date.now() - startTime) / 1000;
  return {
    step: ts + 1,
    total,
    elapsed,
    percent: (((ts + 1) / total) * 100).toFixed(1),
    voltage,
    time_ns: (ts + 0.5) * dt * 1e9,
    backend: 'cuda',
  };
}

export default backend;
