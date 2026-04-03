// FDTD update coefficient computation.
// Mirrors MATLAB lines 970-1292 of fdtd_solve_Ch9_ex9a_12032026.m
//
// General electric update coefficients (lines 975-1009):
//   Cexe  =  (2*eps_r_x*eps0 - dt*sigma_e_x) / (2*eps_r_x*eps0 + dt*sigma_e_x)
//   Cexhz =  (2*dt/dy) / (2*eps_r_x*eps0 + dt*sigma_e_x)
//   Cexhy = -(2*dt/dz) / (2*eps_r_x*eps0 + dt*sigma_e_x)
//   (similar for Ey, Ez, Hx, Hy, Hz)
//
// Lumped element modifications then overwrite subsets of these arrays
// using createLinearIndexList() indices.

import { CONSTANTS } from './constants.js';
import { createLinearIndexList, coordToIndex } from './grid.js';

/**
 * Compute the 18 general FDTD update coefficient arrays.
 * @param {object} mc   - material components from computeMaterialComponents()
 * @param {object} grid
 * @returns {object} coeffs
 */
export function computeGeneralCoefficients(mc, grid) {
  const { eps_0, mu_0 } = CONSTANTS;
  const { nx, ny, nz, nxp1, nyp1, nzp1, dx, dy, dz, dt, buffer, pointers } = grid;
  const {
    eps_r_x, eps_r_y, eps_r_z,
    mu_r_x,  mu_r_y,  mu_r_z,
    sigma_e_x, sigma_e_y, sigma_e_z,
    sigma_m_x, sigma_m_y, sigma_m_z,
  } = mc;

  const two_dt_dy = 2 * dt / dy;
  const two_dt_dz = 2 * dt / dz;
  const two_dt_dx = 2 * dt / dx;

  // ─── Ex coefficients: size (nx, nyp1, nzp1) ──────────────────────────────────
  const nex = nx * nyp1 * nzp1;
  const Cexe  = new Float64Array(buffer, pointers.Cexe, nex);
  const Cexhz = new Float64Array(buffer, pointers.Cexhz, nex);
  const Cexhy = new Float64Array(buffer, pointers.Cexhy, nex);
  for (let n = 0; n < nex; n++) {
    const denom = 2 * eps_r_x[n] * eps_0 + dt * sigma_e_x[n];
    Cexe[n]  =  (2 * eps_r_x[n] * eps_0 - dt * sigma_e_x[n]) / denom;
    Cexhz[n] =  two_dt_dy / denom;
    Cexhy[n] = -two_dt_dz / denom;
  }

  // ─── Ey coefficients: size (nxp1, ny, nzp1) ──────────────────────────────────
  const ney = nxp1 * ny * nzp1;
  const Ceye  = new Float64Array(buffer, pointers.Ceye, ney);
  const Ceyhx = new Float64Array(buffer, pointers.Ceyhx, ney);
  const Ceyhz = new Float64Array(buffer, pointers.Ceyhz, ney);
  for (let n = 0; n < ney; n++) {
    const denom = 2 * eps_r_y[n] * eps_0 + dt * sigma_e_y[n];
    Ceye[n]  =  (2 * eps_r_y[n] * eps_0 - dt * sigma_e_y[n]) / denom;
    Ceyhx[n] =  two_dt_dz / denom;
    Ceyhz[n] = -two_dt_dx / denom;
  }

  // ─── Ez coefficients: size (nxp1, nyp1, nz) ──────────────────────────────────
  const nez = nxp1 * nyp1 * nz;
  const Ceze  = new Float64Array(buffer, pointers.Ceze, nez);
  const Cezhy = new Float64Array(buffer, pointers.Cezhy, nez);
  const Cezhx = new Float64Array(buffer, pointers.Cezhx, nez);
  for (let n = 0; n < nez; n++) {
    const denom = 2 * eps_r_z[n] * eps_0 + dt * sigma_e_z[n];
    Ceze[n]  =  (2 * eps_r_z[n] * eps_0 - dt * sigma_e_z[n]) / denom;
    Cezhy[n] =  two_dt_dx / denom;
    Cezhx[n] = -two_dt_dy / denom;
  }

  // ─── Hx coefficients: size (nxp1, ny, nz) ────────────────────────────────────
  const nhx = nxp1 * ny * nz;
  const Chxh  = new Float64Array(buffer, pointers.Chxh, nhx);
  const Chxez = new Float64Array(buffer, pointers.Chxez, nhx);
  const Chxey = new Float64Array(buffer, pointers.Chxey, nhx);
  for (let n = 0; n < nhx; n++) {
    const denom = 2 * mu_r_x[n] * mu_0 + dt * sigma_m_x[n];
    Chxh[n]  =  (2 * mu_r_x[n] * mu_0 - dt * sigma_m_x[n]) / denom;
    Chxez[n] = -two_dt_dy / denom;
    Chxey[n] =  two_dt_dz / denom;
  }

  // ─── Hy coefficients: size (nx, nyp1, nz) ────────────────────────────────────
  const nhy = nx * nyp1 * nz;
  const Chyh  = new Float64Array(buffer, pointers.Chyh, nhy);
  const Chyex = new Float64Array(buffer, pointers.Chyex, nhy);
  const Chyez = new Float64Array(buffer, pointers.Chyez, nhy);
  for (let n = 0; n < nhy; n++) {
    const denom = 2 * mu_r_y[n] * mu_0 + dt * sigma_m_y[n];
    Chyh[n]  =  (2 * mu_r_y[n] * mu_0 - dt * sigma_m_y[n]) / denom;
    Chyex[n] = -two_dt_dz / denom;
    Chyez[n] =  two_dt_dx / denom;
  }

  // ─── Hz coefficients: size (nx, ny, nzp1) ────────────────────────────────────
  const nhz = nx * ny * nzp1;
  const Chzh  = new Float64Array(buffer, pointers.Chzh, nhz);
  const Chzey = new Float64Array(buffer, pointers.Chzey, nhz);
  const Chzex = new Float64Array(buffer, pointers.Chzex, nhz);
  for (let n = 0; n < nhz; n++) {
    const denom = 2 * mu_r_z[n] * mu_0 + dt * sigma_m_z[n];
    Chzh[n]  =  (2 * mu_r_z[n] * mu_0 - dt * sigma_m_z[n]) / denom;
    Chzey[n] = -two_dt_dx / denom;
    Chzex[n] =  two_dt_dy / denom;
  }

  return {
    Cexe, Cexhz, Cexhy,
    Ceye, Ceyhx, Ceyhz,
    Ceze, Cezhy, Cezhx,
    Chxh, Chxez, Chxey,
    Chyh, Chyex, Chyez,
    Chzh, Chzey, Chzex,
  };
}

/**
 * Modify update coefficients for all lumped elements (voltage/current sources,
 * resistors, capacitors, inductors, diodes).
 * Mirrors MATLAB lines 1015-1292.
 *
 * Modifies coeffs in-place and attaches computed per-element fields
 * (field_indices, Cexs/Ceys/Cezs, etc.) to each element object.
 *
 * @param {object} coeffs  - from computeGeneralCoefficients() (modified in-place)
 * @param {object} mc      - material components (eps_r_x, sigma_e_x, etc.)
 * @param {object} grid
 * @param {Array}  voltageSources
 * @param {Array}  currentSources
 * @param {Array}  resistors
 * @param {Array}  capacitors
 * @param {Array}  inductors
 * @param {Array}  diodes
 */
export function applyLumpedElementCoefficients(
  coeffs, mc, grid,
  voltageSources, currentSources, resistors, capacitors, inductors, diodes
) {
  const { eps_0 } = CONSTANTS;
  const { nx, ny, nz, nxp1, nyp1, nzp1, dx, dy, dz, dt, min_x, min_y, min_z } = grid;
  const { Cexe, Cexhz, Cexhy, Ceye, Ceyhx, Ceyhz, Ceze, Cezhy, Cezhx } = coeffs;
  const { eps_r_x, eps_r_y, eps_r_z, sigma_e_x, sigma_e_y, sigma_e_z } = mc;

  // Helper: get 1-based indices of an element's bounding box
  function getIndices(el) {
    return {
      is: Math.round((el.min_x - min_x) / dx) + 1,
      js: Math.round((el.min_y - min_y) / dy) + 1,
      ks: Math.round((el.min_z - min_z) / dz) + 1,
      ie: Math.round((el.max_x - min_x) / dx) + 1,
      je: Math.round((el.max_y - min_y) / dy) + 1,
      ke: Math.round((el.max_z - min_z) / dz) + 1,
    };
  }

  // ─── Voltage sources (lines 1015-1064) ───────────────────────────────────────
  for (const vs of voltageSources) {
    const { is, js, ks, ie, je, ke } = getIndices(vs);
    let R = vs.resistance_per_component;
    if (R === 0) R = 1e-20;
    const dir = vs.direction[0];

    if (dir === 'x') {
      const fi = createLinearIndexList(is, ie-1, js, je, ks, ke, nyp1, nzp1);
      const a_term = (dt * dx) / (R * dy * dz);
      const Cexs = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_x[idx] + dt*sigma_e_x[idx] + a_term;
        Cexe[idx]  = (2*eps_0*eps_r_x[idx] - dt*sigma_e_x[idx] - a_term) / denom;
        Cexhz[idx] = (2*dt/dy) / denom;
        Cexhy[idx] = -(2*dt/dz) / denom;
        Cexs[n]    = -(2*dt/(R*dy*dz)) / denom;
      }
      vs.field_indices = fi;
      vs.Cexs = Cexs;
    } else if (dir === 'y') {
      const fi = createLinearIndexList(is, ie, js, je-1, ks, ke, ny, nzp1);
      const a_term = (dt * dy) / (R * dz * dx);
      const Ceys = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_y[idx] + dt*sigma_e_y[idx] + a_term;
        Ceye[idx]  = (2*eps_0*eps_r_y[idx] - dt*sigma_e_y[idx] - a_term) / denom;
        Ceyhx[idx] = (2*dt/dz) / denom;
        Ceyhz[idx] = -(2*dt/dx) / denom;
        Ceys[n]    = -(2*dt/(R*dz*dx)) / denom;
      }
      vs.field_indices = fi;
      vs.Ceys = Ceys;
    } else { // z
      const fi = createLinearIndexList(is, ie, js, je, ks, ke-1, nyp1, nz);
      const a_term = (dt * dz) / (R * dx * dy);
      const Cezs = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_z[idx] + dt*sigma_e_z[idx] + a_term;
        Ceze[idx]  = (2*eps_0*eps_r_z[idx] - dt*sigma_e_z[idx] - a_term) / denom;
        Cezhy[idx] = (2*dt/dx) / denom;
        Cezhx[idx] = -(2*dt/dy) / denom;
        Cezs[n]    = -(2*dt/(R*dx*dy)) / denom;
      }
      vs.field_indices = fi;
      vs.Cezs = Cezs;
    }
  }

  // ─── Current sources (lines 1068-1117) ───────────────────────────────────────
  for (const cs of currentSources) {
    const { is, js, ks, ie, je, ke } = getIndices(cs);
    let R = cs.resistance_per_component;
    if (R === 0) R = 1e-20;
    const dir = cs.direction[0];

    if (dir === 'x') {
      const fi = createLinearIndexList(is, ie-1, js, je, ks, ke, nyp1, nzp1);
      const a_term = (dt * dx) / (R * dy * dz);
      const Cexs = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_x[idx] + dt*sigma_e_x[idx] + a_term;
        Cexe[idx]  = (2*eps_0*eps_r_x[idx] - dt*sigma_e_x[idx] - a_term) / denom;
        Cexhz[idx] = (2*dt/dy) / denom;
        Cexhy[idx] = -(2*dt/dz) / denom;
        Cexs[n]    = -(2*dt/(dy*dz)) / denom;
      }
      cs.field_indices = fi;
      cs.Cexs = Cexs;
    } else if (dir === 'y') {
      const fi = createLinearIndexList(is, ie, js, je-1, ks, ke, ny, nzp1);
      const a_term = (dt * dy) / (R * dz * dx);
      const Ceys = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_y[idx] + dt*sigma_e_y[idx] + a_term;
        Ceye[idx]  = (2*eps_0*eps_r_y[idx] - dt*sigma_e_y[idx] - a_term) / denom;
        Ceyhx[idx] = (2*dt/dz) / denom;
        Ceyhz[idx] = -(2*dt/dx) / denom;
        Ceys[n]    = -(2*dt/(dz*dx)) / denom;
      }
      cs.field_indices = fi;
      cs.Ceys = Ceys;
    } else {
      const fi = createLinearIndexList(is, ie, js, je, ks, ke-1, nyp1, nz);
      const a_term = (dt * dz) / (R * dx * dy);
      const Cezs = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_z[idx] + dt*sigma_e_z[idx] + a_term;
        Ceze[idx]  = (2*eps_0*eps_r_z[idx] - dt*sigma_e_z[idx] - a_term) / denom;
        Cezhy[idx] = (2*dt/dx) / denom;
        Cezhx[idx] = -(2*dt/dy) / denom;
        Cezs[n]    = -(2*dt/(dx*dy)) / denom;
      }
      cs.field_indices = fi;
      cs.Cezs = Cezs;
    }
  }

  // ─── Resistors (lines 1121-1164) ─────────────────────────────────────────────
  for (const res of resistors) {
    const { is, js, ks, ie, je, ke } = getIndices(res);
    const R = res.resistance_per_component;
    const dir = res.direction[0];

    if (dir === 'x') {
      const fi = createLinearIndexList(is, ie-1, js, je, ks, ke, nyp1, nzp1);
      const a_term = (dt * dx) / (R * dy * dz);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_x[idx] + dt*sigma_e_x[idx] + a_term;
        Cexe[idx]  = (2*eps_0*eps_r_x[idx] - dt*sigma_e_x[idx] - a_term) / denom;
        Cexhz[idx] = (2*dt/dy) / denom;
        Cexhy[idx] = -(2*dt/dz) / denom;
      }
      res.field_indices = fi;
    } else if (dir === 'y') {
      const fi = createLinearIndexList(is, ie, js, je-1, ks, ke, ny, nzp1);
      const a_term = (dt * dy) / (R * dz * dx);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_y[idx] + dt*sigma_e_y[idx] + a_term;
        Ceye[idx]  = (2*eps_0*eps_r_y[idx] - dt*sigma_e_y[idx] - a_term) / denom;
        Ceyhx[idx] = (2*dt/dz) / denom;
        Ceyhz[idx] = -(2*dt/dx) / denom;
      }
      res.field_indices = fi;
    } else {
      const fi = createLinearIndexList(is, ie, js, je, ks, ke-1, nyp1, nz);
      const a_term = (dt * dz) / (R * dx * dy);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_z[idx] + dt*sigma_e_z[idx] + a_term;
        Ceze[idx]  = (2*eps_0*eps_r_z[idx] - dt*sigma_e_z[idx] - a_term) / denom;
        Cezhy[idx] = (2*dt/dx) / denom;
        Cezhx[idx] = -(2*dt/dy) / denom;
      }
      res.field_indices = fi;
    }
  }

  // ─── Capacitors (lines 1168-1211) ────────────────────────────────────────────
  for (const cap of capacitors) {
    const { is, js, ks, ie, je, ke } = getIndices(cap);
    const C = cap.capacitance_per_component;
    const dir = cap.direction[0];

    if (dir === 'x') {
      const fi = createLinearIndexList(is, ie-1, js, je, ks, ke, nyp1, nzp1);
      const a_term = (2 * C * dx) / (dy * dz);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_x[idx] + dt*sigma_e_x[idx] + a_term;
        Cexe[idx]  = (2*eps_0*eps_r_x[idx] - dt*sigma_e_x[idx] + a_term) / denom;
        Cexhz[idx] = (2*dt/dy) / denom;
        Cexhy[idx] = -(2*dt/dz) / denom;
      }
      cap.field_indices = fi;
    } else if (dir === 'y') {
      const fi = createLinearIndexList(is, ie, js, je-1, ks, ke, ny, nzp1);
      const a_term = (2 * C * dy) / (dz * dx);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_y[idx] + dt*sigma_e_y[idx] + a_term;
        Ceye[idx]  = (2*eps_0*eps_r_y[idx] - dt*sigma_e_y[idx] + a_term) / denom;
        Ceyhx[idx] = (2*dt/dz) / denom;
        Ceyhz[idx] = -(2*dt/dx) / denom;
      }
      cap.field_indices = fi;
    } else {
      const fi = createLinearIndexList(is, ie, js, je, ks, ke-1, nyp1, nz);
      const a_term = (2 * C * dz) / (dx * dy);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        const denom = 2*eps_0*eps_r_z[idx] + dt*sigma_e_z[idx] + a_term;
        Ceze[idx]  = (2*eps_0*eps_r_z[idx] - dt*sigma_e_z[idx] + a_term) / denom;
        Cezhy[idx] = (2*dt/dx) / denom;
        Cezhx[idx] = -(2*dt/dy) / denom;
      }
      cap.field_indices = fi;
    }
  }

  // ─── Inductors (lines 1215-1246) ─────────────────────────────────────────────
  for (const ind of inductors) {
    const { is, js, ks, ie, je, ke } = getIndices(ind);
    const L = ind.inductance_per_component;
    const dir = ind.direction[0];

    if (dir === 'x') {
      const fi = createLinearIndexList(is, ie-1, js, je, ks, ke, nyp1, nzp1);
      const Cexj = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        Cexj[n] = -(2*dt) / (2*eps_r_x[idx]*eps_0 + dt*sigma_e_x[idx]);
      }
      ind.field_indices = fi;
      ind.Cexj = Cexj;
      ind.Cjex = (dt * dx) / (L * dy * dz);
      ind.Jix  = new Float64Array(fi.length);
    } else if (dir === 'y') {
      const fi = createLinearIndexList(is, ie, js, je-1, ks, ke, ny, nzp1);
      const Ceyj = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        Ceyj[n] = -(2*dt) / (2*eps_r_y[idx]*eps_0 + dt*sigma_e_y[idx]);
      }
      ind.field_indices = fi;
      ind.Ceyj = Ceyj;
      ind.Cjey = (dt * dy) / (L * dz * dx);
      ind.Jiy  = new Float64Array(fi.length);
    } else {
      const fi = createLinearIndexList(is, ie, js, je, ks, ke-1, nyp1, nz);
      const Cezj = new Float64Array(fi.length);
      for (let n = 0; n < fi.length; n++) {
        const idx = fi[n];
        Cezj[n] = -(2*dt) / (2*eps_r_z[idx]*eps_0 + dt*sigma_e_z[idx]);
      }
      ind.field_indices = fi;
      ind.Cezj = Cezj;
      ind.Cjez = (dt * dz) / (L * dx * dy);
      ind.Jiz  = new Float64Array(fi.length);
    }
  }

  // ─── Diodes (lines 1250-1292) ─────────────────────────────────────────────────
  const { q, k_B } = CONSTANTS;
  const T = 273 + 27; // room temperature K
  const I_0 = 1e-14;  // saturation current A

  for (const d of diodes) {
    const { is, js, ks } = getIndices(d);
    const dir = d.direction[0];
    const sgn = d.direction[1] === 'n' ? -1 : 1;

    if (dir === 'x') {
      const fi = createLinearIndexList(is, is, js, js, ks, ks, nyp1, nzp1);
      const idx = fi[0];
      d.B    = sgn * q * dx / (2 * k_B * T);
      d.Cexd = -sgn * (2*dt*I_0/(dy*dz)) * Math.exp(d.B) /
               (2*eps_r_x[idx]*eps_0 + dt*sigma_e_x[idx]);
      d.Exn  = 0;
      d.field_indices = fi;
    } else if (dir === 'y') {
      const fi = createLinearIndexList(is, is, js, js, ks, ks, ny, nzp1);
      const idx = fi[0];
      d.B    = sgn * q * dy / (2 * k_B * T);
      d.Ceyd = -sgn * (2*dt*I_0/(dz*dx)) * Math.exp(d.B) /
               (2*eps_r_y[idx]*eps_0 + dt*sigma_e_y[idx]);
      d.Eyn  = 0;
      d.field_indices = fi;
    } else {
      const fi = createLinearIndexList(is, is, js, js, ks, ks, nyp1, nz);
      const idx = fi[0];
      d.B    = sgn * q * dz / (2 * k_B * T);
      d.Cezd = -sgn * (2*dt*I_0/(dx*dy)) * Math.exp(d.B) /
               (2*eps_r_z[idx]*eps_0 + dt*sigma_e_z[idx]);
      d.Ezn  = 0;
      d.field_indices = fi;
    }
  }
}
