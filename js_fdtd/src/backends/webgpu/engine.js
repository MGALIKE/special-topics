// engine.js — environment-agnostic WebGPU FDTD driver.
// ---------------------------------------------------------------------------
// Takes an already-acquired GPUDevice (from Node `webgpu` pkg or browser
// navigator.gpu) and runs the entire FDTD time loop on the GPU, reproducing the
// golden CPU/WASM reference per CONTRACT.md §2:
//
//   H bulk → H CPML → H captures → E bulk → E CPML → source inject → E captures
//   → far-field DFT.
//
// All field + coefficient arrays are uploaded once (f64 → f32). Per-step we only
// read back the small sampled scalars. Far-field DFT for v1 is computed on the
// CPU by mapping back the field buffers at the cadence far-field needs — see the
// clearly-marked FARFIELD section. (TODO: move DFT on-GPU.)
//
// This module is pure ESM with no Node-only imports so it is importable by both
// nodeBackend.js and browserBackend.js (and the Next.js app in visual/).
// ---------------------------------------------------------------------------

import { SHADERS } from './kernels.wgsl.js';
import { accumulateFarfieldDFT } from '../../farfield.js';

const U32 = Uint32Array.BYTES_PER_ELEMENT;
const F32 = Float32Array.BYTES_PER_ELEMENT;

// The updateH / updateE shaders bind 15 storage buffers in a single compute
// stage (bindings 2..16). The WebGPU *default* device limit is only 8, so we
// must opt into a higher `maxStorageBuffersPerShaderStage` when creating the
// device or pipeline creation fails with "exceeds the maximum per-stage limit".
// We request exactly what the shaders need, clamped to what the adapter offers
// (requiredLimits must be <= adapter.limits). Both nodeBackend and browserBackend
// pass the result to adapter.requestDevice().
const REQUIRED_STORAGE_BUFFERS_PER_STAGE = 15;

/** @param {GPUAdapter} adapter @returns {GPUDeviceDescriptor|undefined} */
export function deviceDescriptor(adapter) {
  const supported = adapter?.limits?.maxStorageBuffersPerShaderStage;
  if (typeof supported !== 'number') return undefined;
  if (supported < REQUIRED_STORAGE_BUFFERS_PER_STAGE) return undefined; // can't run here
  return {
    requiredLimits: {
      maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    },
  };
}

// Round a byte length up to 4 (storage buffers must be 4-byte aligned; we use
// f32/u32 so this is automatic, but keep a helper for clarity).
const align4 = (n) => (n + 3) & ~3;

function f32copy(f64arr) {
  const out = new Float32Array(f64arr.length);
  out.set(f64arr); // implicit f64 -> f32 narrowing
  return out;
}

export class WebGPUEngine {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    this.buffers = {};       // name -> GPUBuffer (fields + coeffs + cpml + src)
    this.pipelines = {};
    this._initialized = false;
  }

  // ─── Buffer helpers ────────────────────────────────────────────────────────
  _storageF32(name, data, extraUsage = 0) {
    const arr = data instanceof Float32Array ? data : f32copy(data);
    const buf = this.device.createBuffer({
      label: name,
      size: align4(Math.max(1, arr.length) * F32),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(arr);
    buf.unmap();
    this.buffers[name] = buf;
    return buf;
  }

  _storageU32(name, data) {
    const arr = data instanceof Uint32Array ? data : Uint32Array.from(data);
    const buf = this.device.createBuffer({
      label: name,
      size: align4(Math.max(1, arr.length) * U32),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Uint32Array(buf.getMappedRange()).set(arr);
    buf.unmap();
    this.buffers[name] = buf;
    return buf;
  }

  _uniform(label, byteLength) {
    return this.device.createBuffer({
      label,
      size: align4(byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // ─── init: upload everything, build pipelines ───────────────────────────────
  async init(problem) {
    const { grid, coeffs, cpml, samplers, sources, ff } = problem;
    const dev = this.device;
    const { nx, ny, nz, nxp1, nyp1, nzp1 } = grid;

    const nhx = nxp1 * ny * nz;
    const nhy = nx * nyp1 * nz;
    const nhz = nx * ny * nzp1;
    const nex = nx * nyp1 * nzp1;
    const ney = nxp1 * ny * nzp1;
    const nez = nxp1 * nyp1 * nz;
    this.sizes = { nhx, nhy, nhz, nex, ney, nez };
    this.grid = grid;
    this.ff = ff;
    this.samplers = samplers;
    this.sources = sources;

    // Fields (read_write, will be read back for far-field).
    this._storageF32('Hx', grid.fields.Hx);
    this._storageF32('Hy', grid.fields.Hy);
    this._storageF32('Hz', grid.fields.Hz);
    this._storageF32('Ex', grid.fields.Ex);
    this._storageF32('Ey', grid.fields.Ey);
    this._storageF32('Ez', grid.fields.Ez);

    // 18 coefficients.
    for (const name of [
      'Cexe','Cexhz','Cexhy','Ceye','Ceyhx','Ceyhz','Ceze','Cezhy','Cezhx',
      'Chxh','Chxey','Chxez','Chyh','Chyez','Chyex','Chzh','Chzex','Chzey',
    ]) {
      this._storageF32(name, coeffs[name]);
    }

    // ── Dims uniform (shared by bulk + cpml shaders) ──────────────────────────
    this.dimsBuf = this._uniform('Dims', 12 * U32);
    dev.queue.writeBuffer(this.dimsBuf, 0, new Uint32Array([
      nx, ny, nz, nxp1, nyp1, nzp1, nhx, nhy, nhz, nex, ney, nez,
    ]));

    // ── Build pipelines ───────────────────────────────────────────────────────
    const mk = (code) => dev.createComputePipeline({
      layout: 'auto',
      compute: { module: dev.createShaderModule({ code }), entryPoint: 'main' },
    });
    this.pipelines.updateH = mk(SHADERS.updateH);
    this.pipelines.updateE = mk(SHADERS.updateE);
    // CPML is two passes (see kernels.wgsl.js): update Psi, then correct Field.
    this.pipelines.cpmlPsi = mk(SHADERS.cpmlUpdatePsi);
    this.pipelines.cpmlField = mk(SHADERS.cpmlCorrectField);
    this.pipelines.injectVoltage = mk(SHADERS.injectVoltage);
    this.pipelines.sampleVoltage = mk(SHADERS.sampleVoltage);
    this.pipelines.sampleCurrent = mk(SHADERS.sampleCurrent);

    // ── Per-pass uniforms for bulk H/E (pass = 0/1/2) ─────────────────────────
    this.hPassBuf = [0, 1, 2].map((pass) => {
      const b = this._uniform(`HParams${pass}`, 4 * U32);
      dev.queue.writeBuffer(b, 0, new Uint32Array([pass, 0, 0, 0]));
      return b;
    });
    this.ePassBuf = [0, 1, 2].map((pass) => {
      const b = this._uniform(`EParams${pass}`, 4 * U32);
      dev.queue.writeBuffer(b, 0, new Uint32Array([pass, 0, 0, 0]));
      return b;
    });

    // Bind groups for H/E bulk (static — buffers never move).
    this._buildBulkBindGroups();

    // ── CPML setup ────────────────────────────────────────────────────────────
    this._setupCPML(cpml);

    // ── Source (voltage) setup ────────────────────────────────────────────────
    this._setupVoltageSource(sources);

    // ── Voltage samplers setup ────────────────────────────────────────────────
    this._setupVoltageSamplers(samplers, grid);

    // ── Readback staging buffers for far-field (CPU fallback) ─────────────────
    this._setupFarfieldReadback();

    this._initialized = true;
  }

  _buildBulkBindGroups() {
    const dev = this.device;
    const B = this.buffers;
    const ent = (i, buf) => ({ binding: i, resource: { buffer: buf } });

    this.hBind = this.hPassBuf.map((pb) => dev.createBindGroup({
      layout: this.pipelines.updateH.getBindGroupLayout(0),
      entries: [
        ent(0, this.dimsBuf), ent(1, pb),
        ent(2, B.Hx), ent(3, B.Hy), ent(4, B.Hz),
        ent(5, B.Ex), ent(6, B.Ey), ent(7, B.Ez),
        ent(8, B.Chxh), ent(9, B.Chxey), ent(10, B.Chxez),
        ent(11, B.Chyh), ent(12, B.Chyez), ent(13, B.Chyex),
        ent(14, B.Chzh), ent(15, B.Chzex), ent(16, B.Chzey),
      ],
    }));

    this.eBind = this.ePassBuf.map((pb) => dev.createBindGroup({
      layout: this.pipelines.updateE.getBindGroupLayout(0),
      entries: [
        ent(0, this.dimsBuf), ent(1, pb),
        ent(2, B.Ex), ent(3, B.Ey), ent(4, B.Ez),
        ent(5, B.Hx), ent(6, B.Hy), ent(7, B.Hz),
        ent(8, B.Cexe), ent(9, B.Cexhz), ent(10, B.Cexhy),
        ent(11, B.Ceye), ent(12, B.Ceyhx), ent(13, B.Ceyhz),
        ent(14, B.Ceze), ent(15, B.Cezhy), ent(16, B.Cezhx),
      ],
    }));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CPML — build a flat list of "ops". Each op = one (face,component) dispatch:
  // it owns its own b/a/Psi/CPsi buffers (uploaded from the cpml face object),
  // a CPmlParams uniform (the strides/bases below), a bind group, and a workgroup
  // count. The strides reproduce src/cpml.js exactly.
  //
  // Generic kernel index math (see kernels.wgsl.js cpml_wgsl):
  //   pi = ci*sPi_ci + u*sPi_u + v*sPi_v
  //   s0 = base0 + ci*src_ci + u*src_u + v*src_v ;  s1 = s0 + srcDelta
  //   f  = fbase + ci*fld_ci + u*fld_u + v*fld_v
  // We map (u,v) to the two inner reference loops (in their loop order).
  // ─────────────────────────────────────────────────────────────────────────
  _setupCPML(cpml) {
    this.cpmlMag = []; // ops applied after H bulk
    this.cpmlEle = []; // ops applied after E bulk
    if (!cpml) return;
    const { nx, ny, nz, nxp1, nyp1, nzp1 } = this.grid;

    // helper to register one op
    const addOp = (list, face, comp, cfg) => {
      const dev = this.device;
      // upload b/a (magnetic vs electric chosen by caller), Psi, CPsi
      const tag = `${face}_${comp}`;
      const bBuf = this._storageF32(`cpml_b_${tag}`, cfg.b);
      const aBuf = this._storageF32(`cpml_a_${tag}`, cfg.a);
      const psiBuf = this._storageF32(`cpml_Psi_${tag}`, cfg.Psi);
      const cpsiBuf = this._storageF32(`cpml_CPsi_${tag}`, cfg.CPsi);
      const params = this._uniform(`cpml_P_${tag}`, 16 * U32);
      dev.queue.writeBuffer(params, 0, new Uint32Array([
        cfg.nc, cfg.nU, cfg.nV,
        cfg.sPi_ci, cfg.sPi_u, cfg.sPi_v,
        cfg.base0, cfg.src_ci, cfg.src_u, cfg.src_v, cfg.srcDelta,
        cfg.fbase, cfg.fld_ci, cfg.fld_u, cfg.fld_v, 0,
      ]));
      const FieldBuf = this.buffers[cfg.fieldName];
      const SrcBuf = this.buffers[cfg.srcName];
      const ent = (i, buf) => ({ binding: i, resource: { buffer: buf } });
      // Two bind groups, one per CPML pass. Each has a single writable buffer, so
      // Src and Field aliasing the same component is legal (see kernels.wgsl.js).
      const psiBind = dev.createBindGroup({
        layout: this.pipelines.cpmlPsi.getBindGroupLayout(0),
        entries: [ent(1, params), ent(2, bBuf), ent(3, aBuf), ent(4, psiBuf), ent(7, SrcBuf)],
      });
      const fieldBind = dev.createBindGroup({
        layout: this.pipelines.cpmlField.getBindGroupLayout(0),
        entries: [ent(1, params), ent(4, psiBuf), ent(5, cpsiBuf), ent(6, FieldBuf)],
      });
      const total = cfg.nc * cfg.nU * cfg.nV;
      list.push({ psiBind, fieldBind, total });
    };

    // ── Magnetic CPML ─────────────────────────────────────────────────────────
    // XN/XP (modify Hy from Ez, Hz from Ey)
    for (const key of ['xn', 'xp']) {
      const f = cpml[key]; if (!f) continue;
      const nc = f.nc;
      const i_h = f.i_m_start; // 0-based Hy/Hz start; ei = i_h + ci
      // comp Hy: Psi_hyx (nc,nyp1,nz); src Ez (nxp1,nyp1,nz); field Hy (nx,nyp1,nz)
      addOp(this.cpmlMag, key, 'hy', {
        nc, nU: nyp1, nV: nz,
        b: f.b_m, a: f.a_m, Psi: f.Psi_hyx, CPsi: f.CPsi_hyx,
        sPi_ci: nyp1 * nz, sPi_u: nz, sPi_v: 1,
        // Src Ez index: ((ei)*nyp1*nz + j*nz + k), with ei = i_h+ci ; +1 neighbour i+1
        base0: i_h * (nyp1 * nz), src_ci: nyp1 * nz, src_u: nz, src_v: 1, srcDelta: nyp1 * nz,
        // Field Hy index: ((i_h+ci)*nyp1*nz + j*nz + k)
        fbase: i_h * (nyp1 * nz), fld_ci: nyp1 * nz, fld_u: nz, fld_v: 1,
        fieldName: 'Hy', srcName: 'Ez',
      });
      // comp Hz: Psi_hzx (nc,ny,nzp1); src Ey (nx,nyp1->no: Ey is (nxp1,ny,nzp1)); field Hz (nx,ny,nzp1)
      addOp(this.cpmlMag, key, 'hz', {
        nc, nU: ny, nV: nzp1,
        b: f.b_m, a: f.a_m, Psi: f.Psi_hzx, CPsi: f.CPsi_hzx,
        sPi_ci: ny * nzp1, sPi_u: nzp1, sPi_v: 1,
        base0: i_h * (ny * nzp1), src_ci: ny * nzp1, src_u: nzp1, src_v: 1, srcDelta: ny * nzp1,
        fbase: i_h * (ny * nzp1), fld_ci: ny * nzp1, fld_u: nzp1, fld_v: 1,
        fieldName: 'Hz', srcName: 'Ey',
      });
    }
    // YN/YP (modify Hx from ∂Ez/∂y; Hz from ∂Ex/∂y) — Psi layout (ni,nc,nk).
    // The magnetic CPML is driven by the E-field curl, so the differenced source
    // is Ez (for Hx) / Ex (for Hz) — NOT Hx/Hz. (Differencing the H-field here was
    // a bug that made these faces amplify instead of absorb; see cpml.js.)
    for (const key of ['yn', 'yp']) {
      const f = cpml[key]; if (!f) continue;
      const nc = f.nc;
      const j_h = f.j_m_start; // src j = j_h+ci, neighbour j+1
      // comp Hx: loops i∈[0,nxp1) k∈[0,nz); Psi_hxy (nxp1,nc,nz); src Ez (nxp1,nyp1,nz).
      // Psi layout (i,ci,k): pi = i*nc*nz + ci*nz + k. We map u=i, v=k.
      addOp(this.cpmlMag, key, 'hx', {
        nc, nU: nxp1, nV: nz,
        b: f.b_m, a: f.a_m, Psi: f.Psi_hxy, CPsi: f.CPsi_hxy,
        sPi_ci: nz, sPi_u: nc * nz, sPi_v: 1,
        // Src Ez index: i*nyp1*nz + (j_h+ci)*nz + k ; neighbour (j_h+ci+1)
        base0: j_h * nz, src_ci: nz, src_u: nyp1 * nz, src_v: 1, srcDelta: nz,
        // Field Hx: i*ny*nz + (j_h+ci)*nz + k
        fbase: j_h * nz, fld_ci: nz, fld_u: ny * nz, fld_v: 1,
        fieldName: 'Hx', srcName: 'Ez',
      });
      // comp Hz: loops i∈[0,nx) k∈[0,nzp1); Psi_hzy (nx,nc,nzp1); src Ex (nx,nyp1,nzp1); field Hz
      addOp(this.cpmlMag, key, 'hz', {
        nc, nU: nx, nV: nzp1,
        b: f.b_m, a: f.a_m, Psi: f.Psi_hzy, CPsi: f.CPsi_hzy,
        sPi_ci: nzp1, sPi_u: nc * nzp1, sPi_v: 1,
        base0: j_h * nzp1, src_ci: nzp1, src_u: nyp1 * nzp1, src_v: 1, srcDelta: nzp1,
        fbase: j_h * nzp1, fld_ci: nzp1, fld_u: ny * nzp1, fld_v: 1,
        fieldName: 'Hz', srcName: 'Ex',
      });
    }
    // ZN/ZP — Psi layout (ni,nj,nc)
    for (const key of ['zn', 'zp']) {
      const f = cpml[key]; if (!f) continue;
      const nc = f.nc;
      const k_h = f.k_m_start; // src k = k_h+ci, neighbour k+1
      // comp Hx: Psi_hxz (nxp1,ny,nc); src Ey (nxp1,ny,nzp1); field Hx (nxp1,ny,nz)
      addOp(this.cpmlMag, key, 'hx', {
        nc, nU: nxp1, nV: ny,
        b: f.b_m, a: f.a_m, Psi: f.Psi_hxz, CPsi: f.CPsi_hxz,
        sPi_ci: 1, sPi_u: ny * nc, sPi_v: nc,
        // Src Ey index: i*ny*nzp1 + j*nzp1 + (k_h+ci) ; neighbour +1 in k
        base0: k_h, src_ci: 1, src_u: ny * nzp1, src_v: nzp1, srcDelta: 1,
        // Field Hx: i*ny*nz + j*nz + (k_h+ci)
        fbase: k_h, fld_ci: 1, fld_u: ny * nz, fld_v: nz,
        fieldName: 'Hx', srcName: 'Ey',
      });
      // comp Hy: Psi_hyz (nx,nyp1,nc); src Ex (nx,nyp1,nzp1); field Hy (nx,nyp1,nz)
      addOp(this.cpmlMag, key, 'hy', {
        nc, nU: nx, nV: nyp1,
        b: f.b_m, a: f.a_m, Psi: f.Psi_hyz, CPsi: f.CPsi_hyz,
        sPi_ci: 1, sPi_u: nyp1 * nc, sPi_v: nc,
        base0: k_h, src_ci: 1, src_u: nyp1 * nzp1, src_v: nzp1, srcDelta: 1,
        fbase: k_h, fld_ci: 1, fld_u: nyp1 * nz, fld_v: nz,
        fieldName: 'Hy', srcName: 'Ex',
      });
    }

    // ── Electric CPML ─────────────────────────────────────────────────────────
    // XN/XP (modify Ey from Hz, Ez from Hy). Field E start differs from src H start.
    for (const key of ['xn', 'xp']) {
      const f = cpml[key]; if (!f) continue;
      const nc = f.nc;
      // src index: xn -> ei = ci ; xp -> ei = (nx-nc)+ci-1   (see cpml.js electric XP)
      const srcStart = f.ascending ? (this.grid.nx - nc - 1) : 0;
      const eFieldStart = f.i_e_start; // xn:1, xp:nx-nc
      // comp Ey: Psi_eyx (nc,ny,nzp1); src Hz (nx,ny,nzp1); field Ey (nxp1,ny,nzp1)
      addOp(this.cpmlEle, key, 'ey', {
        nc, nU: ny, nV: nzp1,
        b: f.b_e, a: f.a_e, Psi: f.Psi_eyx, CPsi: f.CPsi_eyx,
        sPi_ci: ny * nzp1, sPi_u: nzp1, sPi_v: 1,
        base0: srcStart * (ny * nzp1), src_ci: ny * nzp1, src_u: nzp1, src_v: 1, srcDelta: ny * nzp1,
        fbase: eFieldStart * (ny * nzp1), fld_ci: ny * nzp1, fld_u: nzp1, fld_v: 1,
        fieldName: 'Ey', srcName: 'Hz',
      });
      // comp Ez: Psi_ezx (nc,nyp1,nz); src Hy (nx,nyp1,nz); field Ez (nxp1,nyp1,nz)
      addOp(this.cpmlEle, key, 'ez', {
        nc, nU: nyp1, nV: nz,
        b: f.b_e, a: f.a_e, Psi: f.Psi_ezx, CPsi: f.CPsi_ezx,
        sPi_ci: nyp1 * nz, sPi_u: nz, sPi_v: 1,
        base0: srcStart * (nyp1 * nz), src_ci: nyp1 * nz, src_u: nz, src_v: 1, srcDelta: nyp1 * nz,
        fbase: eFieldStart * (nyp1 * nz), fld_ci: nyp1 * nz, fld_u: nz, fld_v: 1,
        fieldName: 'Ez', srcName: 'Hy',
      });
    }
    // YN/YP (modify Ez from Hx, Ex from Hz). Psi layout (ni,nc,nk)
    for (const key of ['yn', 'yp']) {
      const f = cpml[key]; if (!f) continue;
      const nc = f.nc;
      const srcStart = f.ascending ? (this.grid.ny - nc - 1) : 0; // src j = srcStart+ci, neighbour +1
      const eFieldStart = f.j_e_start; // yn:1, yp:ny-nc
      // comp Ez: Psi_ezy (nxp1,nc,nz); src Hx (nxp1,ny,nz); field Ez (nxp1,nyp1,nz)
      addOp(this.cpmlEle, key, 'ez', {
        nc, nU: nxp1, nV: nz,
        b: f.b_e, a: f.a_e, Psi: f.Psi_ezy, CPsi: f.CPsi_ezy,
        sPi_ci: nz, sPi_u: nc * nz, sPi_v: 1,
        base0: srcStart * nz, src_ci: nz, src_u: ny * nz, src_v: 1, srcDelta: nz,
        fbase: eFieldStart * nz, fld_ci: nz, fld_u: nyp1 * nz, fld_v: 1,
        fieldName: 'Ez', srcName: 'Hx',
      });
      // comp Ex: Psi_exy (nx,nc,nzp1); src Hz (nx,ny,nzp1); field Ex (nx,nyp1,nzp1)
      addOp(this.cpmlEle, key, 'ex', {
        nc, nU: nx, nV: nzp1,
        b: f.b_e, a: f.a_e, Psi: f.Psi_exy, CPsi: f.CPsi_exy,
        sPi_ci: nzp1, sPi_u: nc * nzp1, sPi_v: 1,
        base0: srcStart * nzp1, src_ci: nzp1, src_u: ny * nzp1, src_v: 1, srcDelta: nzp1,
        fbase: eFieldStart * nzp1, fld_ci: nzp1, fld_u: nyp1 * nzp1, fld_v: 1,
        fieldName: 'Ex', srcName: 'Hz',
      });
    }
    // ZN/ZP (modify Ex from Hy, Ey from Hx). Psi layout (ni,nj,nc)
    for (const key of ['zn', 'zp']) {
      const f = cpml[key]; if (!f) continue;
      const nc = f.nc;
      const srcStart = f.ascending ? (this.grid.nz - nc - 1) : 0; // src k = srcStart+ci, neighbour +1
      const eFieldStart = f.k_e_start; // zn:1, zp:nz-nc
      // comp Ex: Psi_exz (nx,nyp1,nc); src Hy (nx,nyp1,nz); field Ex (nx,nyp1,nzp1)
      addOp(this.cpmlEle, key, 'ex', {
        nc, nU: nx, nV: nyp1,
        b: f.b_e, a: f.a_e, Psi: f.Psi_exz, CPsi: f.CPsi_exz,
        sPi_ci: 1, sPi_u: nyp1 * nc, sPi_v: nc,
        base0: srcStart, src_ci: 1, src_u: nyp1 * nz, src_v: nz, srcDelta: 1,
        fbase: eFieldStart, fld_ci: 1, fld_u: nyp1 * nzp1, fld_v: nzp1,
        fieldName: 'Ex', srcName: 'Hy',
      });
      // comp Ey: Psi_eyz (nxp1,ny,nc); src Hx (nxp1,ny,nz); field Ey (nxp1,ny,nzp1)
      addOp(this.cpmlEle, key, 'ey', {
        nc, nU: nxp1, nV: ny,
        b: f.b_e, a: f.a_e, Psi: f.Psi_eyz, CPsi: f.CPsi_eyz,
        sPi_ci: 1, sPi_u: ny * nc, sPi_v: nc,
        base0: srcStart, src_ci: 1, src_u: ny * nz, src_v: nz, srcDelta: 1,
        fbase: eFieldStart, fld_ci: 1, fld_u: ny * nzp1, fld_v: nzp1,
        fieldName: 'Ey', srcName: 'Hx',
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  _setupVoltageSource(sources) {
    this.voltageOps = [];
    const vss = (sources && sources.voltageSources) || [];
    for (let s = 0; s < vss.length; s++) {
      const vs = vss[s];
      const dir = vs.direction[0];
      const fieldName = dir === 'x' ? 'Ex' : dir === 'y' ? 'Ey' : 'Ez';
      const Cs = dir === 'x' ? vs.Cexs : dir === 'y' ? vs.Ceys : vs.Cezs;
      const fiBuf = this._storageU32(`vsrc_fi_${s}`, Uint32Array.from(vs.field_indices));
      const csBuf = this._storageF32(`vsrc_cs_${s}`, Cs);
      const params = this._uniform(`vsrc_P_${s}`, 4 * U32); // count, v(f32), pad, pad
      const bind = this.device.createBindGroup({
        layout: this.pipelines.injectVoltage.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: fiBuf } },
          { binding: 2, resource: { buffer: csBuf } },
          { binding: 3, resource: { buffer: this.buffers[fieldName] } },
        ],
      });
      this.voltageOps.push({
        bind, params, count: vs.field_indices.length,
        voltage_per_e_field: vs.voltage_per_e_field,
      });
    }
  }

  _setupVoltageSamplers(samplers, grid) {
    this.voltageSamplers = [];
    const svs = (samplers && samplers.sampledVoltages) || [];
    const N = grid.numberOfTimeSteps;
    for (let s = 0; s < svs.length; s++) {
      const obs = svs[s];
      const dir = obs.direction[0];
      const fieldName = dir === 'x' ? 'Ex' : dir === 'y' ? 'Ey' : 'Ez';
      const fiBuf = this._storageU32(`vsamp_fi_${s}`, Uint32Array.from(obs.field_indices));
      // output buffer for the trace (also COPY_SRC for readback)
      const outBuf = this.device.createBuffer({
        label: `vsamp_out_${s}`,
        size: align4(N * F32),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const params = this._uniform(`vsamp_P_${s}`, 4 * U32); // count, csvf(f32), tsIndex, pad
      const bind = this.device.createBindGroup({
        layout: this.pipelines.sampleVoltage.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: fiBuf } },
          { binding: 2, resource: { buffer: this.buffers[fieldName] } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      this.voltageSamplers.push({
        bind, params, outBuf, count: obs.field_indices.length, csvf: obs.Csvf, obs, N,
      });
    }
  }

  // ── Far-field via CPU fallback: prepare readback staging for the 6 fields ───
  _setupFarfieldReadback() {
    this._ffReadback = null;
    if (!this.ff || this.ff.nFreq === 0) return;
    // Staging (MAP_READ) buffers, one per field, reused each step.
    const dev = this.device;
    const mkStage = (name, len) => dev.createBuffer({
      label: `stage_${name}`, size: align4(len * F32),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const { nhx, nhy, nhz, nex, ney, nez } = this.sizes;
    this._ffReadback = {
      Hx: mkStage('Hx', nhx), Hy: mkStage('Hy', nhy), Hz: mkStage('Hz', nhz),
      Ex: mkStage('Ex', nex), Ey: mkStage('Ey', ney), Ez: mkStage('Ez', nez),
      host: {
        Hx: new Float64Array(nhx), Hy: new Float64Array(nhy), Hz: new Float64Array(nhz),
        Ex: new Float64Array(nex), Ey: new Float64Array(ney), Ez: new Float64Array(nez),
      },
    };
  }

  // ─── run: the time loop ─────────────────────────────────────────────────────
  async *run(problem) {
    if (!this._initialized) await this.init(problem);
    const dev = this.device;
    const grid = this.grid;
    const N = grid.numberOfTimeSteps;
    const batchSize = (problem.options && problem.options.batchSize) || 50;
    const startTime = Date.now();

    const wgH = [
      Math.ceil(this.sizes.nhx / 64),
      Math.ceil(this.sizes.nhy / 64),
      Math.ceil(this.sizes.nhz / 64),
    ];
    const wgE = [
      Math.ceil(this.sizes.nex / 64),
      Math.ceil(this.sizes.ney / 64),
      Math.ceil(this.sizes.nez / 64),
    ];

    const needFF = this.ff && this.ff.nFreq > 0;

    for (let ts = 0; ts < N; ts++) {
      const enc = dev.createCommandEncoder();

      // ── 1. H bulk (3 passes) ────────────────────────────────────────────────
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.updateH);
        for (let p = 0; p < 3; p++) {
          pass.setBindGroup(0, this.hBind[p]);
          pass.dispatchWorkgroups(wgH[p]);
        }
        // ── 2. H CPML (two passes: all Psi, then all Field) ───────────────────
        pass.setPipeline(this.pipelines.cpmlPsi);
        for (const op of this.cpmlMag) {
          pass.setBindGroup(0, op.psiBind);
          pass.dispatchWorkgroups(Math.ceil(op.total / 64));
        }
        pass.setPipeline(this.pipelines.cpmlField);
        for (const op of this.cpmlMag) {
          pass.setBindGroup(0, op.fieldBind);
          pass.dispatchWorkgroups(Math.ceil(op.total / 64));
        }
        pass.end();
      }

      // ── 3. H captures: sampled currents (default problem: none) ─────────────
      // (current sampling kernel wired in _setupCurrentSamplers — none by default)

      // ── 4. E bulk (3 passes) ────────────────────────────────────────────────
      {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.updateE);
        for (let p = 0; p < 3; p++) {
          pass.setBindGroup(0, this.eBind[p]);
          pass.dispatchWorkgroups(wgE[p]);
        }
        // ── 5. E CPML (two passes: all Psi, then all Field) ───────────────────
        pass.setPipeline(this.pipelines.cpmlPsi);
        for (const op of this.cpmlEle) {
          pass.setBindGroup(0, op.psiBind);
          pass.dispatchWorkgroups(Math.ceil(op.total / 64));
        }
        pass.setPipeline(this.pipelines.cpmlField);
        for (const op of this.cpmlEle) {
          pass.setBindGroup(0, op.fieldBind);
          pass.dispatchWorkgroups(Math.ceil(op.total / 64));
        }
        pass.end();
      }

      // ── 6. Source injection (voltage). Write the per-step v into params. ─────
      for (const vop of this.voltageOps) {
        const v = vop.voltage_per_e_field[ts];
        // params: [count(u32), v(f32), pad, pad]; build a mixed buffer.
        const u = new Uint32Array([vop.count, 0, 0, 0]);
        new Float32Array(u.buffer)[1] = v;
        dev.queue.writeBuffer(vop.params, 0, u);
      }
      if (this.voltageOps.length) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.injectVoltage);
        for (const vop of this.voltageOps) {
          pass.setBindGroup(0, vop.bind);
          pass.dispatchWorkgroups(Math.ceil(vop.count / 64));
        }
        pass.end();
      }

      // ── 7. E captures: sampled voltages. Set tsIndex in params, reduce. ──────
      for (const samp of this.voltageSamplers) {
        const u = new Uint32Array([samp.count, 0, ts, 0]);
        new Float32Array(u.buffer)[1] = samp.csvf;
        dev.queue.writeBuffer(samp.params, 0, u);
      }
      if (this.voltageSamplers.length) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.sampleVoltage);
        for (const samp of this.voltageSamplers) {
          pass.setBindGroup(0, samp.bind);
          pass.dispatchWorkgroups(1); // single-workgroup reduction
        }
        pass.end();
      }

      // ── 8. Far-field DFT (CPU fallback): copy fields to staging this step. ───
      if (needFF) this._encodeFieldCopyToStage(enc);

      dev.queue.submit([enc.finish()]);

      // Far-field on CPU: map staging, copy to f64 host views, accumulate.
      if (needFF) {
        await this._readFieldsAndAccumulateFF(ts);
      }

      // ── Progress snapshot (per batch + last step) ───────────────────────────
      if ((ts + 1) % batchSize === 0 || ts === N - 1) {
        // Read the latest sampled voltage scalar for the snapshot (and to fill
        // the trace incrementally). We read the whole trace at the end; here we
        // read just the one value for live progress.
        let voltage = null;
        if (this.voltageSamplers.length) {
          voltage = await this._readVoltageScalar(this.voltageSamplers[0], ts);
        }
        yield {
          step: ts + 1,
          total: N,
          elapsed: (Date.now() - startTime) / 1000,
          percent: ((ts + 1) / N * 100).toFixed(1),
          voltage,
          time_ns: (ts + 0.5) * grid.dt * 1e9,
          backend: 'webgpu',
        };
      }
    }

    // ── Finalize: copy full voltage traces back into the sampler objects. ──────
    await this._finalizeSamplers();
  }

  // copy all 6 field storage buffers into their MAP_READ staging buffers
  _encodeFieldCopyToStage(enc) {
    const rb = this._ffReadback;
    const { nhx, nhy, nhz, nex, ney, nez } = this.sizes;
    const cp = (name, len) => enc.copyBufferToBuffer(this.buffers[name], 0, rb[name], 0, align4(len * F32));
    cp('Hx', nhx); cp('Hy', nhy); cp('Hz', nhz);
    cp('Ex', nex); cp('Ey', ney); cp('Ez', nez);
  }

  async _readFieldsAndAccumulateFF(ts) {
    const rb = this._ffReadback;
    const names = ['Hx', 'Hy', 'Hz', 'Ex', 'Ey', 'Ez'];
    await Promise.all(names.map((n) => rb[n].mapAsync(GPUMapMode.READ)));
    for (const n of names) {
      const src = new Float32Array(rb[n].getMappedRange());
      const dst = rb.host[n];
      dst.set(src); // f32 -> f64 widen
      rb[n].unmap();
    }
    // accumulate identically to the reference, using f64 host views.
    accumulateFarfieldDFT(this.ff, rb.host, this.grid, ts);
  }

  async _readVoltageScalar(samp, ts) {
    const dev = this.device;
    const tmp = dev.createBuffer({ size: align4(samp.N * F32), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(samp.outBuf, 0, tmp, 0, align4(samp.N * F32));
    dev.queue.submit([enc.finish()]);
    await tmp.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(tmp.getMappedRange())[ts];
    tmp.unmap(); tmp.destroy();
    return v;
  }

  async _finalizeSamplers() {
    const dev = this.device;
    for (const samp of this.voltageSamplers) {
      const tmp = dev.createBuffer({ size: align4(samp.N * F32), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = dev.createCommandEncoder();
      enc.copyBufferToBuffer(samp.outBuf, 0, tmp, 0, align4(samp.N * F32));
      dev.queue.submit([enc.finish()]);
      await tmp.mapAsync(GPUMapMode.READ);
      const arr = new Float32Array(tmp.getMappedRange());
      for (let i = 0; i < samp.N; i++) samp.obs.sampled_value[i] = arr[i];
      tmp.unmap(); tmp.destroy();
    }
  }
}
