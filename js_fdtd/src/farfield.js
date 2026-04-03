// Farfield: Huygens surface DFT accumulation, radiated power, and directivity.
// Mirrors MATLAB lines 1778-1846 (init), 2335-2427 (per-step J/M + DFT),
// 2660-2686 (radiated power), 2706-2893 (directivity per plane cut).
//
// J/M surface current DFT arrays are stored as interleaved real/imag Float64Arrays.
// Layout for x-face arrays (nj_ff × nk_ff cells): flat index = (nj_idx)*nk_ff + nk_idx
// Layout for y-face arrays (ni_ff × nk_ff cells): flat index = ni_idx*nk_ff + nk_idx
// Layout for z-face arrays (ni_ff × nj_ff cells): flat index = ni_idx*nj_ff + nj_idx
// For frequency dimension: freq*faceSize + spatialIdx

import { CONSTANTS } from './constants.js';

// ─── Index helpers (inlined for performance) ─────────────────────────────────
const idxHx = (i,j,k,ny,nz)    => i*ny*nz   + j*nz   + k;
const idxHy = (i,j,k,nyp1,nz)  => i*nyp1*nz + j*nz   + k;
const idxHz = (i,j,k,ny,nzp1)  => i*ny*nzp1 + j*nzp1 + k;
const idxEx = (i,j,k,nyp1,nzp1)=> i*nyp1*nzp1 + j*nzp1 + k;
const idxEy = (i,j,k,ny,nzp1)  => i*ny*nzp1   + j*nzp1 + k;
const idxEz = (i,j,k,nyp1,nz)  => i*nyp1*nz   + j*nz   + k;

/**
 * Initialize farfield DFT accumulation arrays.
 * Returns a farfield state object with all pre-allocated arrays.
 *
 * @param {object} farfieldParams - { frequencies, number_of_cells_from_outer_boundary }
 * @param {object} grid           - grid object from buildGrid()
 */
export function initFarfield(farfieldParams, grid) {
  const { nx, ny, nz } = grid;
  const { frequencies, number_of_cells_from_outer_boundary: nc } = farfieldParams;
  const nFreq = frequencies.length;
  if (nFreq === 0) return { nFreq: 0 };

  // Huygens surface box boundaries (1-based MATLAB indices, like li/ui in MATLAB)
  const li = nc + 1;
  const lj = nc + 1;
  const lk = nc + 1;
  const ui = nx - nc + 1;
  const uj = ny - nc + 1;
  const uk = nz - nc + 1;

  // Surface dimensions
  const ni_ff = ui - li;   // # cells along x on x-faces
  const nj_ff = uj - lj;   // # cells along y on y-faces
  const nk_ff = uk - lk;   // # cells along z on z-faces

  // Sizes for each face type
  const sz_xface = nj_ff * nk_ff;   // x-faces: vary j,k
  const sz_yface = ni_ff * nk_ff;   // y-faces: vary i,k
  const sz_zface = ni_ff * nj_ff;   // z-faces: vary i,j

  // Allocate complex DFT accumulators (re+im paired): 2 * nFreq * faceSize
  // J (electric surface current), M (magnetic surface current)
  // Naming: c[j/m][comp][face][+/-]  where comp = x/y/z component, face = x/y/z face axis
  const mk = (sz) => ({ re: new Float64Array(nFreq * sz), im: new Float64Array(nFreq * sz) });

  return {
    nFreq, li, lj, lk, ui, uj, uk,
    ni_ff, nj_ff, nk_ff,
    sz_xface, sz_yface, sz_zface,
    frequencies,
    w: Float64Array.from(frequencies, f => 2 * Math.PI * f),
    // J components on each face
    cjyxp: mk(sz_xface), cjzxp: mk(sz_xface),
    cjyxn: mk(sz_xface), cjzxn: mk(sz_xface),
    cjxyp: mk(sz_yface), cjzyp: mk(sz_yface),
    cjxyn: mk(sz_yface), cjzyn: mk(sz_yface),
    cjxzp: mk(sz_zface), cjyzp: mk(sz_zface),
    cjxzn: mk(sz_zface), cjyzn: mk(sz_zface),
    // M components on each face
    cmyxp: mk(sz_xface), cmzxp: mk(sz_xface),
    cmyxn: mk(sz_xface), cmzxn: mk(sz_xface),
    cmxyp: mk(sz_yface), cmzyp: mk(sz_yface),
    cmxyn: mk(sz_yface), cmzyn: mk(sz_yface),
    cmxzp: mk(sz_zface), cmyzp: mk(sz_zface),
    cmxzn: mk(sz_zface), cmyzn: mk(sz_zface),
  };
}

/**
 * Per-step: compute J and M on the Huygens surface, then DFT-accumulate.
 * Called at the END of each time step, after E-field update.
 * time_step is 0-based (MATLAB: 1-based).
 *
 * MATLAB DFT phase for H: exp(-j*w*(time_step-0.5)*dt)   (1-based ts → (ts-0.5)*dt)
 * In 0-based JS:           exp(-j*w*(ts+0.5)*dt)
 * MATLAB DFT phase for E: exp(-j*w*time_step*dt)          (1-based ts → ts*dt)
 * In 0-based JS:           exp(-j*w*(ts+1)*dt)
 */
export function accumulateFarfieldDFT(ff, fields, grid, ts) {
  if (ff.nFreq === 0) return;
  const { Hx, Hy, Hz, Ex, Ey, Ez } = fields;
  const { ny, nz, nxp1, nyp1, nzp1, dt } = grid;
  const { li, lj, lk, ui, uj, uk, ni_ff, nj_ff, nk_ff, nFreq, w } = ff;

  // Convert MATLAB 1-based to 0-based
  const LI = li - 1, LJ = lj - 1, LK = lk - 1;
  const UI = ui - 1, UJ = uj - 1, UK = uk - 1;

  // ── Accumulate DFT for all frequencies ────────────────────────────────────
  for (let mi = 0; mi < nFreq; mi++) {
    const wdt = w[mi] * dt;
    // H-field phase: (ts + 0.5)*dt  (H is sampled at half-integer time steps)
    const ph_h = wdt * (ts + 0.5);
    const cos_h =  Math.cos(ph_h);
    const sin_h = -Math.sin(ph_h);
    // E-field phase: (ts + 1)*dt
    const ph_e = wdt * (ts + 1);
    const cos_e =  Math.cos(ph_e);
    const sin_e = -Math.sin(ph_e);
    const freqOff_x = mi * ff.sz_xface;
    const freqOff_y = mi * ff.sz_yface;
    const freqOff_z = mi * ff.sz_zface;

    // ── xp face (i = UI, normal = +x) ──────────────────────────────────────
    // My = +0.5*(Ez(UI,lj:uj-1,lk:uk-1)+Ez(UI,lj+1:uj,lk:uk-1))
    // Mz = -0.5*(Ey(UI,lj:uj-1,lk:uk-1)+Ey(UI,lj:uj-1,lk+1:uk))
    // Jy = -0.25*(Hz4-average)
    // Jz = +0.25*(Hy4-average)
    for (let nj = 0; nj < nj_ff; nj++) {
      for (let nk = 0; nk < nk_ff; nk++) {
        const p = freqOff_x + nj * nk_ff + nk;
        const j0 = LJ + nj, k0 = LK + nk;
        // My_xp
        const tmyxp = 0.5 * (Ez[idxEz(UI, j0, k0, nyp1, nz)] + Ez[idxEz(UI, j0+1, k0, nyp1, nz)]);
        ff.cmyxp.re[p] += cos_e * tmyxp;  ff.cmyxp.im[p] += sin_e * tmyxp;
        // Mz_xp
        const tmzxp = -0.5 * (Ey[idxEy(UI, j0, k0, ny, nzp1)] + Ey[idxEy(UI, j0, k0+1, ny, nzp1)]);
        ff.cmzxp.re[p] += cos_e * tmzxp;  ff.cmzxp.im[p] += sin_e * tmzxp;
        // Jy_xp
        const tjyxp = -0.25 * (Hz[idxHz(UI, j0, k0, ny, nzp1)] + Hz[idxHz(UI, j0, k0+1, ny, nzp1)]
                              + Hz[idxHz(UI-1, j0, k0, ny, nzp1)] + Hz[idxHz(UI-1, j0, k0+1, ny, nzp1)]);
        ff.cjyxp.re[p] += cos_h * tjyxp;  ff.cjyxp.im[p] += sin_h * tjyxp;
        // Jz_xp
        const tjzxp = 0.25 * (Hy[idxHy(UI, j0, k0, nyp1, nz)] + Hy[idxHy(UI, j0+1, k0, nyp1, nz)]
                             + Hy[idxHy(UI-1, j0, k0, nyp1, nz)] + Hy[idxHy(UI-1, j0+1, k0, nyp1, nz)]);
        ff.cjzxp.re[p] += cos_h * tjzxp;  ff.cjzxp.im[p] += sin_h * tjzxp;
      }
    }

    // ── xn face (i = LI, normal = -x) ─────────────────────────────────────
    for (let nj = 0; nj < nj_ff; nj++) {
      for (let nk = 0; nk < nk_ff; nk++) {
        const p = freqOff_x + nj * nk_ff + nk;
        const j0 = LJ + nj, k0 = LK + nk;
        const tmyxn = -0.5 * (Ez[idxEz(LI, j0, k0, nyp1, nz)] + Ez[idxEz(LI, j0+1, k0, nyp1, nz)]);
        ff.cmyxn.re[p] += cos_e * tmyxn;  ff.cmyxn.im[p] += sin_e * tmyxn;
        const tmzxn = 0.5 * (Ey[idxEy(LI, j0, k0, ny, nzp1)] + Ey[idxEy(LI, j0, k0+1, ny, nzp1)]);
        ff.cmzxn.re[p] += cos_e * tmzxn;  ff.cmzxn.im[p] += sin_e * tmzxn;
        const tjyxn = 0.25 * (Hz[idxHz(LI, j0, k0, ny, nzp1)] + Hz[idxHz(LI, j0, k0+1, ny, nzp1)]
                             + Hz[idxHz(LI-1, j0, k0, ny, nzp1)] + Hz[idxHz(LI-1, j0, k0+1, ny, nzp1)]);
        ff.cjyxn.re[p] += cos_h * tjyxn;  ff.cjyxn.im[p] += sin_h * tjyxn;
        const tjzxn = -0.25 * (Hy[idxHy(LI, j0, k0, nyp1, nz)] + Hy[idxHy(LI, j0+1, k0, nyp1, nz)]
                              + Hy[idxHy(LI-1, j0, k0, nyp1, nz)] + Hy[idxHy(LI-1, j0+1, k0, nyp1, nz)]);
        ff.cjzxn.re[p] += cos_h * tjzxn;  ff.cjzxn.im[p] += sin_h * tjzxn;
      }
    }

    // ── yp face (j = UJ, normal = +y) ─────────────────────────────────────
    for (let ni = 0; ni < ni_ff; ni++) {
      for (let nk = 0; nk < nk_ff; nk++) {
        const p = freqOff_y + ni * nk_ff + nk;
        const i0 = LI + ni, k0 = LK + nk;
        // Mx = -0.5*(Ez(li:ui-1,uj,lk:uk-1)+Ez(li+1:ui,uj,lk:uk-1))
        const tmxyp = -0.5 * (Ez[idxEz(i0, UJ, k0, nyp1, nz)] + Ez[idxEz(i0+1, UJ, k0, nyp1, nz)]);
        ff.cmxyp.re[p] += cos_e * tmxyp;  ff.cmxyp.im[p] += sin_e * tmxyp;
        // Mz = +0.5*(Ex(li:ui-1,uj,lk:uk-1)+Ex(li:ui-1,uj,lk+1:uk))
        const tmzyp = 0.5 * (Ex[idxEx(i0, UJ, k0, nyp1, nzp1)] + Ex[idxEx(i0, UJ, k0+1, nyp1, nzp1)]);
        ff.cmzyp.re[p] += cos_e * tmzyp;  ff.cmzyp.im[p] += sin_e * tmzyp;
        // Jz = -0.25*(Hx4-avg)
        const tjzyp = -0.25 * (Hx[idxHx(i0, UJ, k0, ny, nz)] + Hx[idxHx(i0+1, UJ, k0, ny, nz)]
                              + Hx[idxHx(i0, UJ-1, k0, ny, nz)] + Hx[idxHx(i0+1, UJ-1, k0, ny, nz)]);
        ff.cjzyp.re[p] += cos_h * tjzyp;  ff.cjzyp.im[p] += sin_h * tjzyp;
        // Jx = +0.25*(Hz4-avg)
        const tjxyp = 0.25 * (Hz[idxHz(i0, UJ, k0, ny, nzp1)] + Hz[idxHz(i0, UJ, k0+1, ny, nzp1)]
                             + Hz[idxHz(i0, UJ-1, k0, ny, nzp1)] + Hz[idxHz(i0, UJ-1, k0+1, ny, nzp1)]);
        ff.cjxyp.re[p] += cos_h * tjxyp;  ff.cjxyp.im[p] += sin_h * tjxyp;
      }
    }

    // ── yn face (j = LJ, normal = -y) ─────────────────────────────────────
    for (let ni = 0; ni < ni_ff; ni++) {
      for (let nk = 0; nk < nk_ff; nk++) {
        const p = freqOff_y + ni * nk_ff + nk;
        const i0 = LI + ni, k0 = LK + nk;
        const tmxyn = 0.5 * (Ez[idxEz(i0, LJ, k0, nyp1, nz)] + Ez[idxEz(i0+1, LJ, k0, nyp1, nz)]);
        ff.cmxyn.re[p] += cos_e * tmxyn;  ff.cmxyn.im[p] += sin_e * tmxyn;
        const tmzyn = -0.5 * (Ex[idxEx(i0, LJ, k0, nyp1, nzp1)] + Ex[idxEx(i0, LJ, k0+1, nyp1, nzp1)]);
        ff.cmzyn.re[p] += cos_e * tmzyn;  ff.cmzyn.im[p] += sin_e * tmzyn;
        const tjzyn = 0.25 * (Hx[idxHx(i0, LJ, k0, ny, nz)] + Hx[idxHx(i0+1, LJ, k0, ny, nz)]
                            + Hx[idxHx(i0, LJ-1, k0, ny, nz)] + Hx[idxHx(i0+1, LJ-1, k0, ny, nz)]);
        ff.cjzyn.re[p] += cos_h * tjzyn;  ff.cjzyn.im[p] += sin_h * tjzyn;
        const tjxyn = -0.25 * (Hz[idxHz(i0, LJ, k0, ny, nzp1)] + Hz[idxHz(i0, LJ, k0+1, ny, nzp1)]
                              + Hz[idxHz(i0, LJ-1, k0, ny, nzp1)] + Hz[idxHz(i0, LJ-1, k0+1, ny, nzp1)]);
        ff.cjxyn.re[p] += cos_h * tjxyn;  ff.cjxyn.im[p] += sin_h * tjxyn;
      }
    }

    // ── zp face (k = UK, normal = +z) ─────────────────────────────────────
    for (let ni = 0; ni < ni_ff; ni++) {
      for (let nj = 0; nj < nj_ff; nj++) {
        const p = freqOff_z + ni * nj_ff + nj;
        const i0 = LI + ni, j0 = LJ + nj;
        // Mx = +0.5*(Ey(li:ui-1,lj:uj-1,uk)+Ey(li+1:ui,lj:uj-1,uk))
        const tmxzp = 0.5 * (Ey[idxEy(i0, j0, UK, ny, nzp1)] + Ey[idxEy(i0+1, j0, UK, ny, nzp1)]);
        ff.cmxzp.re[p] += cos_e * tmxzp;  ff.cmxzp.im[p] += sin_e * tmxzp;
        // My = -0.5*(Ex(li:ui-1,lj:uj-1,uk)+Ex(li:ui-1,lj+1:uj,uk))
        const tmyzp = -0.5 * (Ex[idxEx(i0, j0, UK, nyp1, nzp1)] + Ex[idxEx(i0, j0+1, UK, nyp1, nzp1)]);
        ff.cmyzp.re[p] += cos_e * tmyzp;  ff.cmyzp.im[p] += sin_e * tmyzp;
        // Jx = +0.25*(Hx4-avg)
        const tjxzp = 0.25 * (Hx[idxHx(i0, j0, UK, ny, nz)] + Hx[idxHx(i0+1, j0, UK, ny, nz)]
                             + Hx[idxHx(i0, j0, UK-1, ny, nz)] + Hx[idxHx(i0+1, j0, UK-1, ny, nz)]);
        ff.cjxzp.re[p] += cos_h * tjxzp;  ff.cjxzp.im[p] += sin_h * tjxzp;
        // Jy = +0.25*(Hy4-avg)  — MATLAB: tjyzp = +0.25*(Hx...)
        const tjyzp = 0.25 * (Hx[idxHx(i0, j0, UK, ny, nz)] + Hx[idxHx(i0+1, j0, UK, ny, nz)]
                            + Hx[idxHx(i0, j0, UK-1, ny, nz)] + Hx[idxHx(i0+1, j0, UK-1, ny, nz)]);
        // Note: MATLAB line 2362: tjyzp = +0.25*(Hx(li:ui-1,lj:uj-1,uk)+ Hx(li+1:ui,...))
        // Jy uses Hx averaged over i and k:
        const tjyzp2 = 0.25 * (Hx[idxHx(i0, j0, UK, ny, nz)] + Hx[idxHx(i0+1, j0, UK, ny, nz)]
                              + Hx[idxHx(i0, j0, UK-1, ny, nz)] + Hx[idxHx(i0+1, j0, UK-1, ny, nz)]);
        // tjxzp uses Hy averaged: MATLAB: tjxzp = -0.25*(Hy(li:ui-1,lj:uj-1,uk)+Hy(li:ui-1,lj+1:uj,uk)+...uk-1...)
        const tjxzp2 = -0.25 * (Hy[idxHy(i0, j0, UK, nyp1, nz)] + Hy[idxHy(i0, j0+1, UK, nyp1, nz)]
                               + Hy[idxHy(i0, j0, UK-1, nyp1, nz)] + Hy[idxHy(i0, j0+1, UK-1, nyp1, nz)]);
        ff.cjxzp.re[p] = ff.cjxzp.re[p] - cos_h * tjxzp + cos_h * tjxzp2;
        ff.cjxzp.im[p] = ff.cjxzp.im[p] - sin_h * tjxzp + sin_h * tjxzp2;
        ff.cjyzp.re[p] += cos_h * tjyzp2;  ff.cjyzp.im[p] += sin_h * tjyzp2;
      }
    }

    // ── zn face (k = LK, normal = -z) ─────────────────────────────────────
    for (let ni = 0; ni < ni_ff; ni++) {
      for (let nj = 0; nj < nj_ff; nj++) {
        const p = freqOff_z + ni * nj_ff + nj;
        const i0 = LI + ni, j0 = LJ + nj;
        const tmxzn = -0.5 * (Ey[idxEy(i0, j0, LK, ny, nzp1)] + Ey[idxEy(i0+1, j0, LK, ny, nzp1)]);
        ff.cmxzn.re[p] += cos_e * tmxzn;  ff.cmxzn.im[p] += sin_e * tmxzn;
        const tmyzn = 0.5 * (Ex[idxEx(i0, j0, LK, nyp1, nzp1)] + Ex[idxEx(i0, j0+1, LK, nyp1, nzp1)]);
        ff.cmyzn.re[p] += cos_e * tmyzn;  ff.cmyzn.im[p] += sin_e * tmyzn;
        const tjxzn = 0.25 * (Hy[idxHy(i0, j0, LK, nyp1, nz)] + Hy[idxHy(i0, j0+1, LK, nyp1, nz)]
                            + Hy[idxHy(i0, j0, LK-1, nyp1, nz)] + Hy[idxHy(i0, j0+1, LK-1, nyp1, nz)]);
        ff.cjxzn.re[p] += cos_h * tjxzn;  ff.cjxzn.im[p] += sin_h * tjxzn;
        const tjyzn = -0.25 * (Hx[idxHx(i0, j0, LK, ny, nz)] + Hx[idxHx(i0+1, j0, LK, ny, nz)]
                              + Hx[idxHx(i0, j0, LK-1, ny, nz)] + Hx[idxHx(i0+1, j0, LK-1, ny, nz)]);
        ff.cjyzn.re[p] += cos_h * tjyzn;  ff.cjyzn.im[p] += sin_h * tjyzn;
      }
    }
  }
}

/**
 * Calculate total radiated power on the Huygens surface.
 * Uses the Poynting vector integral: P = 0.5*Re(∮ E×H* · dA)
 * Mirrors MATLAB lines 2663-2686.
 *
 * @returns {Float64Array} radiated_power[nFreq]
 */
export function calcRadiatedPower(ff, grid) {
  const { dx, dy, dz } = grid;
  const { nFreq, ni_ff, nj_ff, nk_ff, sz_xface, sz_yface, sz_zface } = ff;
  const radiated_power = new Float64Array(nFreq);

  for (let mi = 0; mi < nFreq; mi++) {
    const ox = mi * sz_xface;
    const oy = mi * sz_yface;
    const oz = mi * sz_zface;
    let powr = 0;

    // zp face: My*conj(Jx) - Mx*conj(Jy)  * dx*dy
    for (let p = 0; p < sz_zface; p++) {
      // Re(cmyzp * conj(cjxzp)) = re*re + im*im  (conj flips sign of im part of cj)
      powr += dx*dy * (
          ff.cmyzp.re[oz+p]*ff.cjxzp.re[oz+p] + ff.cmyzp.im[oz+p]*ff.cjxzp.im[oz+p]
        - ff.cmxzp.re[oz+p]*ff.cjyzp.re[oz+p] - ff.cmxzp.im[oz+p]*ff.cjyzp.im[oz+p]
      );
    }
    // zn face: subtract
    for (let p = 0; p < sz_zface; p++) {
      powr -= dx*dy * (
          ff.cmyzn.re[oz+p]*ff.cjxzn.re[oz+p] + ff.cmyzn.im[oz+p]*ff.cjxzn.im[oz+p]
        - ff.cmxzn.re[oz+p]*ff.cjyzn.re[oz+p] - ff.cmxzn.im[oz+p]*ff.cjyzn.im[oz+p]
      );
    }
    // yp face: Mx*conj(Jz) - Mz*conj(Jx)  * dx*dz
    for (let p = 0; p < sz_yface; p++) {
      powr += dx*dz * (
          ff.cmxyp.re[oy+p]*ff.cjzyp.re[oy+p] + ff.cmxyp.im[oy+p]*ff.cjzyp.im[oy+p]
        - ff.cmzyp.re[oy+p]*ff.cjxyp.re[oy+p] - ff.cmzyp.im[oy+p]*ff.cjxyp.im[oy+p]
      );
    }
    // yn face: subtract
    for (let p = 0; p < sz_yface; p++) {
      powr -= dx*dz * (
          ff.cmxyn.re[oy+p]*ff.cjzyn.re[oy+p] + ff.cmxyn.im[oy+p]*ff.cjzyn.im[oy+p]
        - ff.cmzyn.re[oy+p]*ff.cjxyn.re[oy+p] - ff.cmzyn.im[oy+p]*ff.cjxyn.im[oy+p]
      );
    }
    // xp face: Mz*conj(Jy) - My*conj(Jz)  * dy*dz
    for (let p = 0; p < sz_xface; p++) {
      powr += dy*dz * (
          ff.cmzxp.re[ox+p]*ff.cjyxp.re[ox+p] + ff.cmzxp.im[ox+p]*ff.cjyxp.im[ox+p]
        - ff.cmyxp.re[ox+p]*ff.cjzxp.re[ox+p] - ff.cmyxp.im[ox+p]*ff.cjzxp.im[ox+p]
      );
    }
    // xn face: subtract
    for (let p = 0; p < sz_xface; p++) {
      powr -= dy*dz * (
          ff.cmzxn.re[ox+p]*ff.cjyxn.re[ox+p] + ff.cmzxn.im[ox+p]*ff.cjyxn.im[ox+p]
        - ff.cmyxn.re[ox+p]*ff.cjzxn.re[ox+p] - ff.cmyxn.im[ox+p]*ff.cjzxn.im[ox+p]
      );
    }
    radiated_power[mi] = 0.5 * powr;
  }
  return radiated_power;
}

/**
 * Calculate directivity pattern for a plane cut.
 * Mirrors MATLAB lines 2755-2893 (called for xy, xz, yz planes).
 *
 * @param {object} ff              - farfield state from initFarfield()
 * @param {object} grid            - grid object
 * @param {Float64Array} radPower  - radiated_power[nFreq] from calcRadiatedPower()
 * @param {Float64Array} theta     - theta angles (radians), length nAngles
 * @param {Float64Array} phi       - phi angles (radians), length nAngles
 * @returns {{ dataTheta: Float64Array[], dataPhi: Float64Array[] }}
 *          dataTheta[mi] and dataPhi[mi] are nAngles directivity arrays per frequency
 */
export function calcDirectivity(ff, grid, radPower, theta, phi) {
  const { dx, dy, dz } = grid;
  const { nFreq, li, lj, lk, ui, uj, uk, ni_ff, nj_ff, nk_ff, sz_xface, sz_yface, sz_zface, frequencies } = ff;
  const { eta_0 } = CONSTANTS;
  const nAngles = theta.length;

  const dataTheta = Array.from({ length: nFreq }, () => new Float64Array(nAngles));
  const dataPhi   = Array.from({ length: nFreq }, () => new Float64Array(nAngles));

  // Pre-compute angle-dependent quantities (same for all frequencies)
  const sinth = new Float64Array(nAngles);
  const costh = new Float64Array(nAngles);
  const sinph = new Float64Array(nAngles);
  const cosph = new Float64Array(nAngles);
  for (let a = 0; a < nAngles; a++) {
    sinth[a] = Math.sin(theta[a]);
    costh[a] = Math.cos(theta[a]);
    sinph[a] = Math.sin(phi[a]);
    cosph[a] = Math.cos(phi[a]);
  }

  const ci = 0.5 * (ui + li);
  const cj = 0.5 * (uj + lj);
  const ck = 0.5 * (uk + lk);

  for (let mi = 0; mi < nFreq; mi++) {
    const k_wave = 2 * Math.PI * frequencies[mi] / CONSTANTS.c;
    const scale = k_wave * k_wave / (8 * Math.PI * eta_0 * radPower[mi]);
    const ox = mi * sz_xface;
    const oy = mi * sz_yface;
    const oz = mi * sz_zface;

    // Accumulators for N and L vectors (theta/phi components), real and imag
    const NtRe = new Float64Array(nAngles);
    const NtIm = new Float64Array(nAngles);
    const NpRe = new Float64Array(nAngles);
    const NpIm = new Float64Array(nAngles);
    const LtRe = new Float64Array(nAngles);
    const LtIm = new Float64Array(nAngles);
    const LpRe = new Float64Array(nAngles);
    const LpIm = new Float64Array(nAngles);

    // ── x-faces (vary j, k) ────────────────────────────────────────────────
    for (let nj = 0; nj < nj_ff; nj++) {
      for (let nk = 0; nk < nk_ff; nk++) {
        const p = ox + nj * nk_ff + nk;
        const jpos = (lj - 1) + nj;
        const kpos = (lk - 1) + nk;
        const Jyr = ff.cjyxp.re[p]; const Jyi = ff.cjyxp.im[p];
        const Jzr = ff.cjzxp.re[p]; const Jzi = ff.cjzxp.im[p];
        const Myr = ff.cmyxp.re[p]; const Myi = ff.cmyxp.im[p];
        const Mzr = ff.cmzxp.re[p]; const Mzi = ff.cmzxp.im[p];
        const JyrN = ff.cjyxn.re[p]; const JyiN = ff.cjyxn.im[p];
        const JzrN = ff.cjzxn.re[p]; const JziN = ff.cjzxn.im[p];
        const MyrN = ff.cmyxn.re[p]; const MyiN = ff.cmyxn.im[p];
        const MzrN = ff.cmzxn.re[p]; const MziN = ff.cmzxn.im[p];
        const dydz = dy * dz;

        for (let a = 0; a < nAngles; a++) {
          // Phase for +x face
          const rpr_p = (ui - ci) * dx * sinth[a] * cosph[a]
                      + (jpos - cj + 0.5) * dy * sinth[a] * sinph[a]
                      + (kpos - ck + 0.5) * dz * costh[a];
          const cos_rpr_p = Math.cos(k_wave * rpr_p);
          const sin_rpr_p = Math.sin(k_wave * rpr_p);

          const costh_sinph = costh[a] * sinph[a];
          const dydz_costh_sinph = dydz * costh_sinph;
          const dydz_sinth = dydz * sinth[a];
          const dydz_cosph = dydz * cosph[a];

          // Ntheta = Jy*costh*sinph - Jz*sinth
          const NT_r = (Jyr * dydz_costh_sinph - Jzr * dydz_sinth);
          const NT_i = (Jyi * dydz_costh_sinph - Jzi * dydz_sinth);
          NtRe[a] += NT_r * cos_rpr_p - NT_i * sin_rpr_p;
          NtIm[a] += NT_r * sin_rpr_p + NT_i * cos_rpr_p;
          // Nphi = Jy*cosphi
          const NP_r = Jyr * dydz_cosph;
          const NP_i = Jyi * dydz_cosph;
          NpRe[a] += NP_r * cos_rpr_p - NP_i * sin_rpr_p;
          NpIm[a] += NP_r * sin_rpr_p + NP_i * cos_rpr_p;
          // Ltheta = My*costh*sinph - Mz*sinth
          const LT_r = (Myr * dydz_costh_sinph - Mzr * dydz_sinth);
          const LT_i = (Myi * dydz_costh_sinph - Mzi * dydz_sinth);
          LtRe[a] += LT_r * cos_rpr_p - LT_i * sin_rpr_p;
          LtIm[a] += LT_r * sin_rpr_p + LT_i * cos_rpr_p;
          // Lphi = My*cosphi
          const LP_r = Myr * dydz_cosph;
          const LP_i = Myi * dydz_cosph;
          LpRe[a] += LP_r * cos_rpr_p - LP_i * sin_rpr_p;
          LpIm[a] += LP_r * sin_rpr_p + LP_i * cos_rpr_p;

          // Phase for -x face
          const rpr_n = (li - ci) * dx * sinth[a] * cosph[a]
                      + (jpos - cj + 0.5) * dy * sinth[a] * sinph[a]
                      + (kpos - ck + 0.5) * dz * costh[a];
          const cos_rpr_n = Math.cos(k_wave * rpr_n);
          const sin_rpr_n = Math.sin(k_wave * rpr_n);

          const NT_rN = (JyrN * dydz_costh_sinph - JzrN * dydz_sinth);
          const NT_iN = (JyiN * dydz_costh_sinph - JziN * dydz_sinth);
          NtRe[a] += NT_rN * cos_rpr_n - NT_iN * sin_rpr_n;
          NtIm[a] += NT_rN * sin_rpr_n + NT_iN * cos_rpr_n;
          const NP_rN = JyrN * dydz_cosph;
          const NP_iN = JyiN * dydz_cosph;
          NpRe[a] += NP_rN * cos_rpr_n - NP_iN * sin_rpr_n;
          NpIm[a] += NP_rN * sin_rpr_n + NP_iN * cos_rpr_n;
          const LT_rN = (MyrN * dydz_costh_sinph - MzrN * dydz_sinth);
          const LT_iN = (MyiN * dydz_costh_sinph - MziN * dydz_sinth);
          LtRe[a] += LT_rN * cos_rpr_n - LT_iN * sin_rpr_n;
          LtIm[a] += LT_rN * sin_rpr_n + LT_iN * cos_rpr_n;
          const LP_rN = MyrN * dydz_cosph;
          const LP_iN = MyiN * dydz_cosph;
          LpRe[a] += LP_rN * cos_rpr_n - LP_iN * sin_rpr_n;
          LpIm[a] += LP_rN * sin_rpr_n + LP_iN * cos_rpr_n;
        }
      }
    }

    // ── y-faces (vary i, k) ────────────────────────────────────────────────
    for (let ni = 0; ni < ni_ff; ni++) {
      for (let nk = 0; nk < nk_ff; nk++) {
        const p = oy + ni * nk_ff + nk;
        const ipos = (li - 1) + ni;
        const kpos = (lk - 1) + nk;
        const Jxr = ff.cjxyp.re[p]; const Jxi = ff.cjxyp.im[p];
        const Jzr = ff.cjzyp.re[p]; const Jzi = ff.cjzyp.im[p];
        const Mxr = ff.cmxyp.re[p]; const Mxi = ff.cmxyp.im[p];
        const Mzr = ff.cmzyp.re[p]; const Mzi = ff.cmzyp.im[p];
        const JxrN = ff.cjxyn.re[p]; const JxiN = ff.cjxyn.im[p];
        const JzrN = ff.cjzyn.re[p]; const JziN = ff.cjzyn.im[p];
        const MxrN = ff.cmxyn.re[p]; const MxiN = ff.cmxyn.im[p];
        const MzrN = ff.cmzyn.re[p]; const MziN = ff.cmzyn.im[p];
        const dxdz = dx * dz;

        for (let a = 0; a < nAngles; a++) {
          const rpr_p = (ipos - ci + 0.5) * dx * sinth[a] * cosph[a]
                      + (uj - cj) * dy * sinth[a] * sinph[a]
                      + (kpos - ck + 0.5) * dz * costh[a];
          const cos_rpr_p = Math.cos(k_wave * rpr_p);
          const sin_rpr_p = Math.sin(k_wave * rpr_p);

          const costh_cosph = costh[a] * cosph[a];
          const dxdz_costh_cosph = dxdz * costh_cosph;
          const dxdz_sinth = dxdz * sinth[a];
          const dxdz_sinph = dxdz * sinph[a];

          const NT_r = Jxr * dxdz_costh_cosph - Jzr * dxdz_sinth;
          const NT_i = Jxi * dxdz_costh_cosph - Jzi * dxdz_sinth;
          NtRe[a] += NT_r * cos_rpr_p - NT_i * sin_rpr_p;
          NtIm[a] += NT_r * sin_rpr_p + NT_i * cos_rpr_p;
          const NP_r = -Jxr * dxdz_sinph;
          const NP_i = -Jxi * dxdz_sinph;
          NpRe[a] += NP_r * cos_rpr_p - NP_i * sin_rpr_p;
          NpIm[a] += NP_r * sin_rpr_p + NP_i * cos_rpr_p;
          const LT_r = Mxr * dxdz_costh_cosph - Mzr * dxdz_sinth;
          const LT_i = Mxi * dxdz_costh_cosph - Mzi * dxdz_sinth;
          LtRe[a] += LT_r * cos_rpr_p - LT_i * sin_rpr_p;
          LtIm[a] += LT_r * sin_rpr_p + LT_i * cos_rpr_p;
          const LP_r = -Mxr * dxdz_sinph;
          const LP_i = -Mxi * dxdz_sinph;
          LpRe[a] += LP_r * cos_rpr_p - LP_i * sin_rpr_p;
          LpIm[a] += LP_r * sin_rpr_p + LP_i * cos_rpr_p;

          const rpr_n = (ipos - ci + 0.5) * dx * sinth[a] * cosph[a]
                      + (lj - cj) * dy * sinth[a] * sinph[a]
                      + (kpos - ck + 0.5) * dz * costh[a];
          const cos_rpr_n = Math.cos(k_wave * rpr_n);
          const sin_rpr_n = Math.sin(k_wave * rpr_n);

          const NT_rN = JxrN * dxdz_costh_cosph - JzrN * dxdz_sinth;
          const NT_iN = JxiN * dxdz_costh_cosph - JziN * dxdz_sinth;
          NtRe[a] += NT_rN * cos_rpr_n - NT_iN * sin_rpr_n;
          NtIm[a] += NT_rN * sin_rpr_n + NT_iN * cos_rpr_n;
          const NP_rN = -JxrN * dxdz_sinph;
          const NP_iN = -JxiN * dxdz_sinph;
          NpRe[a] += NP_rN * cos_rpr_n - NP_iN * sin_rpr_n;
          NpIm[a] += NP_rN * sin_rpr_n + NP_iN * cos_rpr_n;
          const LT_rN = MxrN * dxdz_costh_cosph - MzrN * dxdz_sinth;
          const LT_iN = MxiN * dxdz_costh_cosph - MziN * dxdz_sinth;
          LtRe[a] += LT_rN * cos_rpr_n - LT_iN * sin_rpr_n;
          LtIm[a] += LT_rN * sin_rpr_n + LT_iN * cos_rpr_n;
          const LP_rN = -MxrN * dxdz_sinph;
          const LP_iN = -MxiN * dxdz_sinph;
          LpRe[a] += LP_rN * cos_rpr_n - LP_iN * sin_rpr_n;
          LpIm[a] += LP_rN * sin_rpr_n + LP_iN * cos_rpr_n;
        }
      }
    }

    // ── z-faces (vary i, j) ────────────────────────────────────────────────
    for (let ni = 0; ni < ni_ff; ni++) {
      for (let nj = 0; nj < nj_ff; nj++) {
        const p = oz + ni * nj_ff + nj;
        const ipos = (li - 1) + ni;
        const jpos = (lj - 1) + nj;
        const Jxr = ff.cjxzp.re[p]; const Jxi = ff.cjxzp.im[p];
        const Jyr = ff.cjyzp.re[p]; const Jyi = ff.cjyzp.im[p];
        const Mxr = ff.cmxzp.re[p]; const Mxi = ff.cmxzp.im[p];
        const Myr = ff.cmyzp.re[p]; const Myi = ff.cmyzp.im[p];
        const JxrN = ff.cjxzn.re[p]; const JxiN = ff.cjxzn.im[p];
        const JyrN = ff.cjyzn.re[p]; const JyiN = ff.cjyzn.im[p];
        const MxrN = ff.cmxzn.re[p]; const MxiN = ff.cmxzn.im[p];
        const MyrN = ff.cmyzn.re[p]; const MyiN = ff.cmyzn.im[p];
        const dxdy = dx * dy;

        for (let a = 0; a < nAngles; a++) {
          const rpr_p = (ipos - ci + 0.5) * dx * sinth[a] * cosph[a]
                      + (jpos - cj + 0.5) * dy * sinth[a] * sinph[a]
                      + (uk - ck) * dz * costh[a];
          const cos_rpr_p = Math.cos(k_wave * rpr_p);
          const sin_rpr_p = Math.sin(k_wave * rpr_p);

          const costh_cosph = costh[a] * cosph[a];
          const costh_sinph = costh[a] * sinph[a];
          const dxdy_costh_cosph = dxdy * costh_cosph;
          const dxdy_costh_sinph = dxdy * costh_sinph;
          const dxdy_sinph = dxdy * sinph[a];
          const dxdy_cosph = dxdy * cosph[a];

          // Ntheta = Jx*costh*cosphi + Jy*costh*sinphi
          const NT_r = Jxr * dxdy_costh_cosph + Jyr * dxdy_costh_sinph;
          const NT_i = Jxi * dxdy_costh_cosph + Jyi * dxdy_costh_sinph;
          NtRe[a] += NT_r * cos_rpr_p - NT_i * sin_rpr_p;
          NtIm[a] += NT_r * sin_rpr_p + NT_i * cos_rpr_p;
          // Nphi = -Jx*sinphi + Jy*cosphi
          const NP_r = -Jxr * dxdy_sinph + Jyr * dxdy_cosph;
          const NP_i = -Jxi * dxdy_sinph + Jyi * dxdy_cosph;
          NpRe[a] += NP_r * cos_rpr_p - NP_i * sin_rpr_p;
          NpIm[a] += NP_r * sin_rpr_p + NP_i * cos_rpr_p;
          const LT_r = Mxr * dxdy_costh_cosph + Myr * dxdy_costh_sinph;
          const LT_i = Mxi * dxdy_costh_cosph + Myi * dxdy_costh_sinph;
          LtRe[a] += LT_r * cos_rpr_p - LT_i * sin_rpr_p;
          LtIm[a] += LT_r * sin_rpr_p + LT_i * cos_rpr_p;
          const LP_r = -Mxr * dxdy_sinph + Myr * dxdy_cosph;
          const LP_i = -Mxi * dxdy_sinph + Myi * dxdy_cosph;
          LpRe[a] += LP_r * cos_rpr_p - LP_i * sin_rpr_p;
          LpIm[a] += LP_r * sin_rpr_p + LP_i * cos_rpr_p;

          const rpr_n = (ipos - ci + 0.5) * dx * sinth[a] * cosph[a]
                      + (jpos - cj + 0.5) * dy * sinth[a] * sinph[a]
                      + (lk - ck) * dz * costh[a];
          const cos_rpr_n = Math.cos(k_wave * rpr_n);
          const sin_rpr_n = Math.sin(k_wave * rpr_n);

          const NT_rN = JxrN * dxdy_costh_cosph + JyrN * dxdy_costh_sinph;
          const NT_iN = JxiN * dxdy_costh_cosph + JyiN * dxdy_costh_sinph;
          NtRe[a] += NT_rN * cos_rpr_n - NT_iN * sin_rpr_n;
          NtIm[a] += NT_rN * sin_rpr_n + NT_iN * cos_rpr_n;
          const NP_rN = -JxrN * dxdy_sinph + JyrN * dxdy_cosph;
          const NP_iN = -JxiN * dxdy_sinph + JyiN * dxdy_cosph;
          NpRe[a] += NP_rN * cos_rpr_n - NP_iN * sin_rpr_n;
          NpIm[a] += NP_rN * sin_rpr_n + NP_iN * cos_rpr_n;
          const LT_rN = MxrN * dxdy_costh_cosph + MyrN * dxdy_costh_sinph;
          const LT_iN = MxiN * dxdy_costh_cosph + MyiN * dxdy_costh_sinph;
          LtRe[a] += LT_rN * cos_rpr_n - LT_iN * sin_rpr_n;
          LtIm[a] += LT_rN * sin_rpr_n + LT_iN * cos_rpr_n;
          const LP_rN = -MxrN * dxdy_sinph + MyrN * dxdy_cosph;
          const LP_iN = -MxiN * dxdy_sinph + MyiN * dxdy_cosph;
          LpRe[a] += LP_rN * cos_rpr_n - LP_iN * sin_rpr_n;
          LpIm[a] += LP_rN * sin_rpr_n + LP_iN * cos_rpr_n;
        }
      }
    }

    // ── Directivity = |Lphi + eta0*Ntheta|^2 * k^2/(8*pi*eta0*P_rad) ─────
    for (let a = 0; a < nAngles; a++) {
      // D_theta = (k^2 / (8*pi*eta0*P)) * |Lphi + eta0*Ntheta|^2
      const LpNt_r = LpRe[a] + eta_0 * NtRe[a];
      const LpNt_i = LpIm[a] + eta_0 * NtIm[a];
      dataTheta[mi][a] = scale * (LpNt_r * LpNt_r + LpNt_i * LpNt_i);
      // D_phi = (k^2 / (8*pi*eta0*P)) * |Ltheta - eta0*Nphi|^2
      const LtNp_r = LtRe[a] - eta_0 * NpRe[a];
      const LtNp_i = LtIm[a] - eta_0 * NpIm[a];
      dataPhi[mi][a] = scale * (LtNp_r * LtNp_r + LtNp_i * LtNp_i);
    }
  }

  return { dataTheta, dataPhi };
}
