// Sampling: observer setup and per-step field/voltage/current capture.
// Mirrors MATLAB lines 1686-1908 (init) and 2022-2333 (per-step capture).
//
// Observers are initialized before the time loop; capture* functions are
// called at specific points inside the loop.

import { CONSTANTS } from './constants.js';

// ─── Index helpers (inlined for hot-path speed) ─────────────────────────────
// All 0-based, row-major.
const idxHx = (i, j, k, ny, nz)   => i*ny*nz   + j*nz   + k;
const idxHy = (i, j, k, nyp1, nz) => i*nyp1*nz + j*nz   + k;
const idxHz = (i, j, k, ny, nzp1) => i*ny*nzp1 + j*nzp1 + k;
const idxEx = (i, j, k, nyp1, nzp1) => i*nyp1*nzp1 + j*nzp1 + k;
const idxEy = (i, j, k, ny, nzp1)   => i*ny*nzp1   + j*nzp1 + k;
const idxEz = (i, j, k, nyp1, nz)   => i*nyp1*nz   + j*nz   + k;

/**
 * Convert physical coordinate to 1-based MATLAB grid index.
 */
function coordIdx(coord, domainMin, cellSize) {
  return Math.round((coord - domainMin) / cellSize) + 1;
}

/**
 * Build flat JS index list for a volume region.
 * Returns an Int32Array of 0-based flat indices into a (d1×d2×d3) array.
 * is..ie, js..je, ks..ke are 1-based inclusive MATLAB indices.
 * JS array size is (d1, d2, d3) in row-major: idx = (i-1)*d2*d3 + (j-1)*d3 + (k-1)
 */
function makeIndices(is, ie, js, je, ks, ke, d2, d3) {
  const count = (ie-is+1) * (je-js+1) * (ke-ks+1);
  const fi = new Int32Array(count);
  let p = 0;
  for (let mk = ks; mk <= ke; mk++) {
    for (let mj = js; mj <= je; mj++) {
      for (let mi = is; mi <= ie; mi++) {
        fi[p++] = (mi-1)*d2*d3 + (mj-1)*d3 + (mk-1);
      }
    }
  }
  return fi;
}

// ─── Initialization ──────────────────────────────────────────────────────────

/**
 * Initialize sampled electric field observers.
 * Adds .is/.js/.ks, .sampled_value Float64Array, .time Float64Array.
 */
export function initSampledElectricFields(sampledEFields, grid) {
  const { min_x, min_y, min_z, dx, dy, dz, numberOfTimeSteps, dt } = grid;
  for (const obs of sampledEFields) {
    obs.is = coordIdx(obs.x, min_x, dx);
    obs.js = coordIdx(obs.y, min_y, dy);
    obs.ks = coordIdx(obs.z, min_z, dz);
    obs.sampled_value = new Float64Array(numberOfTimeSteps);
    obs.time = Float64Array.from({ length: numberOfTimeSteps }, (_, n) => (n + 1) * dt);
  }
}

/**
 * Initialize sampled magnetic field observers.
 * H-fields are time-shifted by -dt/2 relative to E-fields.
 */
export function initSampledMagneticFields(sampledHFields, grid) {
  const { min_x, min_y, min_z, dx, dy, dz, numberOfTimeSteps, dt } = grid;
  for (const obs of sampledHFields) {
    obs.is = coordIdx(obs.x, min_x, dx);
    obs.js = coordIdx(obs.y, min_y, dy);
    obs.ks = coordIdx(obs.z, min_z, dz);
    obs.sampled_value = new Float64Array(numberOfTimeSteps);
    obs.time = Float64Array.from({ length: numberOfTimeSteps }, (_, n) => (n + 0.5) * dt);
  }
}

/**
 * Initialize sampled voltage observers.
 * Computes field index arrays and the averaging coefficient Csvf.
 */
export function initSampledVoltages(sampledVoltages, grid) {
  const { min_x, min_y, min_z, dx, dy, dz,
          nx, ny, nz, nxp1, nyp1, nzp1, numberOfTimeSteps, dt } = grid;
  for (const obs of sampledVoltages) {
    const is = coordIdx(obs.min_x, min_x, dx);
    const js = coordIdx(obs.min_y, min_y, dy);
    const ks = coordIdx(obs.min_z, min_z, dz);
    const ie = coordIdx(obs.max_x, min_x, dx);
    const je = coordIdx(obs.max_y, min_y, dy);
    const ke = coordIdx(obs.max_z, min_z, dz);
    obs.is = is; obs.js = js; obs.ks = ks;
    obs.ie = ie; obs.je = je; obs.ke = ke;

    const dir = obs.direction[0];
    let fi;
    switch (dir) {
      case 'x':
        // Ex(i,j,k): size (nx, nyp1, nzp1) — average over j,k
        fi = makeIndices(is, ie-1, js, je, ks, ke, nyp1, nzp1);
        obs.Csvf = -dx / ((je-js+1) * (ke-ks+1));
        break;
      case 'y':
        // Ey(i,j,k): size (nxp1, ny, nzp1)
        fi = makeIndices(is, ie, js, je-1, ks, ke, ny, nzp1);
        obs.Csvf = -dy / ((ke-ks+1) * (ie-is+1));
        break;
      case 'z':
        // Ez(i,j,k): size (nxp1, nyp1, nz)
        fi = makeIndices(is, ie, js, je, ks, ke-1, nyp1, nz);
        obs.Csvf = -dz / ((ie-is+1) * (je-js+1));
        break;
    }
    if (obs.direction[1] === 'n') obs.Csvf *= -1;
    obs.field_indices = fi;
    obs.sampled_value = new Float64Array(numberOfTimeSteps);
    obs.time = Float64Array.from({ length: numberOfTimeSteps }, (_, n) => (n + 1) * dt);
  }
}

/**
 * Initialize sampled current observers.
 * Currents use H-field time stagger (time - 0.5*dt).
 */
export function initSampledCurrents(sampledCurrents, grid) {
  const { min_x, min_y, min_z, dx, dy, dz, numberOfTimeSteps, dt } = grid;
  for (const obs of sampledCurrents) {
    obs.is = coordIdx(obs.min_x, min_x, dx);
    obs.js = coordIdx(obs.min_y, min_y, dy);
    obs.ks = coordIdx(obs.min_z, min_z, dz);
    obs.ie = coordIdx(obs.max_x, min_x, dx);
    obs.je = coordIdx(obs.max_y, min_y, dy);
    obs.ke = coordIdx(obs.max_z, min_z, dz);
    obs.sampled_value = new Float64Array(numberOfTimeSteps);
    obs.time = Float64Array.from({ length: numberOfTimeSteps }, (_, n) => (n + 0.5) * dt);
  }
}

// ─── Per-step capture (called inside the time loop) ─────────────────────────

/**
 * Capture magnetic fields AFTER H-field update (including CPML corrections).
 * Called at time_step (0-based index ts).
 */
export function captureMagneticFields(sampledHFields, fields, grid, ts) {
  const { ny, nz, nxp1, nyp1, nzp1 } = grid;
  const { Hx, Hy, Hz } = fields;

  for (const obs of sampledHFields) {
    const is = obs.is - 1;  // convert to 0-based
    const js = obs.js - 1;
    const ks = obs.ks - 1;
    let sv;
    switch (obs.component) {
      case 'x':
        // 0.25 * sum(Hx(is, js-1:js, ks-1:ks)) — 4-point average
        sv = 0.25 * (
          Hx[idxHx(is, js-1, ks-1, ny, nz)] +
          Hx[idxHx(is, js,   ks-1, ny, nz)] +
          Hx[idxHx(is, js-1, ks,   ny, nz)] +
          Hx[idxHx(is, js,   ks,   ny, nz)]
        );
        break;
      case 'y':
        sv = 0.25 * (
          Hy[idxHy(is-1, js, ks-1, nyp1, nz)] +
          Hy[idxHy(is,   js, ks-1, nyp1, nz)] +
          Hy[idxHy(is-1, js, ks,   nyp1, nz)] +
          Hy[idxHy(is,   js, ks,   nyp1, nz)]
        );
        break;
      case 'z':
        sv = 0.25 * (
          Hz[idxHz(is-1, js-1, ks, ny, nzp1)] +
          Hz[idxHz(is,   js-1, ks, ny, nzp1)] +
          Hz[idxHz(is-1, js,   ks, ny, nzp1)] +
          Hz[idxHz(is,   js,   ks, ny, nzp1)]
        );
        break;
      case 'm': {
        const svx = 0.25 * (
          Hx[idxHx(is, js-1, ks-1, ny, nz)] + Hx[idxHx(is, js, ks-1, ny, nz)] +
          Hx[idxHx(is, js-1, ks,   ny, nz)] + Hx[idxHx(is, js, ks,   ny, nz)]
        );
        const svy = 0.25 * (
          Hy[idxHy(is-1, js, ks-1, nyp1, nz)] + Hy[idxHy(is, js, ks-1, nyp1, nz)] +
          Hy[idxHy(is-1, js, ks,   nyp1, nz)] + Hy[idxHy(is, js, ks,   nyp1, nz)]
        );
        const svz = 0.25 * (
          Hz[idxHz(is-1, js-1, ks, ny, nzp1)] + Hz[idxHz(is, js-1, ks, ny, nzp1)] +
          Hz[idxHz(is-1, js,   ks, ny, nzp1)] + Hz[idxHz(is, js,   ks, ny, nzp1)]
        );
        sv = Math.sqrt(svx*svx + svy*svy + svz*svz);
        break;
      }
    }
    obs.sampled_value[ts] = sv;
  }
}

/**
 * Capture sampled currents via Ampere's law path integral around the loop.
 * Called AFTER H-field update (H is at time step + 0.5).
 */
export function captureSampledCurrents(sampledCurrents, fields, grid, ts) {
  const { dx, dy, dz, ny, nz, nxp1, nyp1, nzp1 } = grid;
  const { Hx, Hy, Hz } = fields;

  for (const obs of sampledCurrents) {
    const { is, js, ks, ie, je, ke } = obs;
    // Convert to 0-based
    const I = is-1, J = js-1, K = ks-1;
    const IE = ie-1, JE = je-1, KE = ke-1;
    let sv = 0;

    switch (obs.direction[0]) {
      case 'x': {
        // Loop in yz-plane at i=ie-1 (0-based: IE-1)
        const ix = IE - 1;
        for (let j = J; j <= JE-1; j++) sv += dy * Hy[idxHy(ix, j, K-1, nyp1, nz)];
        for (let k = K; k <= KE-1; k++) sv += dz * Hz[idxHz(ix, JE-1, k, ny, nzp1)];
        for (let j = J; j <= JE-1; j++) sv -= dy * Hy[idxHy(ix, j, KE-1, nyp1, nz)];
        for (let k = K; k <= KE-1; k++) sv -= dz * Hz[idxHz(ix, J-1, k, ny, nzp1)];
        break;
      }
      case 'y': {
        const jx = JE - 1;
        for (let k = K; k <= KE-1; k++) sv += dz * Hz[idxHz(I-1, jx, k, ny, nzp1)];
        for (let i = I; i <= IE-1; i++) sv += dx * Hx[idxHx(i, jx, KE-1, ny, nz)];
        for (let k = K; k <= KE-1; k++) sv -= dz * Hz[idxHz(IE, jx, k, ny, nzp1)];
        for (let i = I; i <= IE-1; i++) sv -= dx * Hx[idxHx(i, jx, K-1, ny, nz)];
        break;
      }
      case 'z': {
        const kx = KE - 1;
        for (let i = I; i <= IE-1; i++) sv += dx * Hx[idxHx(i, J-1, kx, ny, nz)];
        for (let j = J; j <= JE-1; j++) sv += dy * Hy[idxHy(IE, j, kx, nyp1, nz)];
        for (let i = I; i <= IE-1; i++) sv -= dx * Hx[idxHx(i, JE, kx, ny, nz)];
        for (let j = J; j <= JE-1; j++) sv -= dy * Hy[idxHy(I-1, j, kx, nyp1, nz)];
        break;
      }
    }
    if (obs.direction[1] === 'n') sv = -sv;
    obs.sampled_value[ts] = sv;
  }
}

/**
 * Capture electric fields AFTER E-field update and source injection.
 * Called at time_step ts (0-based).
 */
export function captureElectricFields(sampledEFields, fields, grid, ts) {
  const { ny, nz, nxp1, nyp1, nzp1 } = grid;
  const { Ex, Ey, Ez } = fields;

  for (const obs of sampledEFields) {
    const is = obs.is - 1;
    const js = obs.js - 1;
    const ks = obs.ks - 1;
    let sv;
    switch (obs.component) {
      case 'x':
        // 0.5*(Ex(is-1,js,ks) + Ex(is,js,ks)) — 2-point average
        sv = 0.5 * (
          Ex[idxEx(is-1, js, ks, nyp1, nzp1)] +
          Ex[idxEx(is,   js, ks, nyp1, nzp1)]
        );
        break;
      case 'y':
        sv = 0.5 * (
          Ey[idxEy(is, js-1, ks, ny, nzp1)] +
          Ey[idxEy(is, js,   ks, ny, nzp1)]
        );
        break;
      case 'z':
        sv = 0.5 * (
          Ez[idxEz(is, js, ks-1, nyp1, nz)] +
          Ez[idxEz(is, js, ks,   nyp1, nz)]
        );
        break;
      case 'm': {
        const svx = 0.5 * (Ex[idxEx(is-1, js, ks, nyp1, nzp1)] + Ex[idxEx(is, js, ks, nyp1, nzp1)]);
        const svy = 0.5 * (Ey[idxEy(is, js-1, ks, ny, nzp1)]   + Ey[idxEy(is, js, ks, ny, nzp1)]);
        const svz = 0.5 * (Ez[idxEz(is, js, ks-1, nyp1, nz)]   + Ez[idxEz(is, js, ks, nyp1, nz)]);
        sv = Math.sqrt(svx*svx + svy*svy + svz*svz);
        break;
      }
    }
    obs.sampled_value[ts] = sv;
  }
}

/**
 * Capture sampled voltages AFTER E-field update and source injection.
 */
export function captureSampledVoltages(sampledVoltages, fields, grid, ts) {
  const { Ex, Ey, Ez } = fields;
  const allFields = { x: Ex, y: Ey, z: Ez };

  for (const obs of sampledVoltages) {
    const dir = obs.direction[0];
    const F = allFields[dir];
    const fi = obs.field_indices;
    let sum = 0;
    for (let n = 0; n < fi.length; n++) sum += F[fi[n]];
    obs.sampled_value[ts] = obs.Csvf * sum;
  }
}
