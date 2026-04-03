// Grid: domain sizing, cell coordinates, flat-array index helpers.
// Mirrors MATLAB lines 260-382 of fdtd_solve_Ch9_ex9a_12032026.m
//
// Indexing convention (critical):
//   MATLAB arrays are 1-based and column-major: A(i,j,k) → fast index = i
//   JS arrays are 0-based and row-major: A[i*ny*nz + j*nz + k] → fast index = k
//   All JS index functions below use 0-based i,j,k.

import { CONSTANTS } from './constants.js';

/**
 * Compute domain size and allocate all field + coefficient arrays.
 * @param {object} params - simulation parameters (see index.js)
 * @returns {object} grid - all dimensions, strides, and preallocated arrays
 */
export function buildGrid(params) {
  const { dx, dy, dz, boundary, bricks, spheres, courantFactor, numberOfTimeSteps } = params;
  const { eps_0, mu_0, c } = CONSTANTS;

  // ─── Find bounding box of all objects ───────────────────────────────────────
  let min_x = Infinity, min_y = Infinity, min_z = Infinity;
  let max_x = -Infinity, max_y = -Infinity, max_z = -Infinity;

  for (const s of spheres) {
    min_x = Math.min(min_x, s.center_x - s.radius);
    min_y = Math.min(min_y, s.center_y - s.radius);
    min_z = Math.min(min_z, s.center_z - s.radius);
    max_x = Math.max(max_x, s.center_x + s.radius);
    max_y = Math.max(max_y, s.center_y + s.radius);
    max_z = Math.max(max_z, s.center_z + s.radius);
  }
  for (const b of bricks) {
    min_x = Math.min(min_x, b.min_x);
    min_y = Math.min(min_y, b.min_y);
    min_z = Math.min(min_z, b.min_z);
    max_x = Math.max(max_x, b.max_x);
    max_y = Math.max(max_y, b.max_y);
    max_z = Math.max(max_z, b.max_z);
  }

  // ─── Add air buffers ─────────────────────────────────────────────────────────
  min_x -= dx * boundary.air_buffer_xn;
  min_y -= dy * boundary.air_buffer_yn;
  min_z -= dz * boundary.air_buffer_zn;
  max_x += dx * boundary.air_buffer_xp;
  max_y += dy * boundary.air_buffer_yp;
  max_z += dz * boundary.air_buffer_zp;

  // ─── Add CPML layers ─────────────────────────────────────────────────────────
  if (boundary.type_xn === 'cpml' && boundary.cpml_cells_xn > 0) min_x -= dx * boundary.cpml_cells_xn;
  if (boundary.type_xp === 'cpml' && boundary.cpml_cells_xp > 0) max_x += dx * boundary.cpml_cells_xp;
  if (boundary.type_yn === 'cpml' && boundary.cpml_cells_yn > 0) min_y -= dy * boundary.cpml_cells_yn;
  if (boundary.type_yp === 'cpml' && boundary.cpml_cells_yp > 0) max_y += dy * boundary.cpml_cells_yp;
  if (boundary.type_zn === 'cpml' && boundary.cpml_cells_zn > 0) min_z -= dz * boundary.cpml_cells_zn;
  if (boundary.type_zp === 'cpml' && boundary.cpml_cells_zp > 0) max_z += dz * boundary.cpml_cells_zp;

  // ─── Snap to grid ────────────────────────────────────────────────────────────
  const nx = Math.round((max_x - min_x) / dx);
  const ny = Math.round((max_y - min_y) / dy);
  const nz = Math.round((max_z - min_z) / dz);

  // Adjust max to snapped values
  const domainMaxX = min_x + nx * dx;
  const domainMaxY = min_y + ny * dy;
  const domainMaxZ = min_z + nz * dz;

  const nxp1 = nx + 1, nyp1 = ny + 1, nzp1 = nz + 1;

  // ─── Time step ───────────────────────────────────────────────────────────────
  const dt = courantFactor / (c * Math.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2));

  // ─── Cell center coordinate arrays (flat, row-major) ─────────────────────────
  // cell_center_x[i*ny*nz + j*nz + k] = (i + 0.5)*dx + min_x
  const cellX = new Float64Array(nx * ny * nz);
  const cellY = new Float64Array(nx * ny * nz);
  const cellZ = new Float64Array(nx * ny * nz);
  for (let i = 0; i < nx; i++) {
    const cx = (i + 0.5) * dx + min_x;
    for (let j = 0; j < ny; j++) {
      const cy = (j + 0.5) * dy + min_y;
      for (let k = 0; k < nz; k++) {
        const idx = i * ny * nz + j * nz + k;
        cellX[idx] = cx;
        cellY[idx] = cy;
        cellZ[idx] = (k + 0.5) * dz + min_z;
      }
    }
  }

  // ─── Precompute flat strides for each field component ────────────────────────
  // These are used by both JS and WASM kernels.
  const strides = {
    // Hx(i,j,k): size (nxp1, ny, nz)
    hx: { ni: nxp1, nj: ny,   nk: nz,   s1: ny*nz,   s2: nz,   s3: 1 },
    // Hy(i,j,k): size (nx, nyp1, nz)
    hy: { ni: nx,   nj: nyp1, nk: nz,   s1: nyp1*nz, s2: nz,   s3: 1 },
    // Hz(i,j,k): size (nx, ny, nzp1)
    hz: { ni: nx,   nj: ny,   nk: nzp1, s1: ny*nzp1, s2: nzp1, s3: 1 },
    // Ex(i,j,k): size (nx, nyp1, nzp1)
    ex: { ni: nx,   nj: nyp1, nk: nzp1, s1: nyp1*nzp1, s2: nzp1, s3: 1 },
    // Ey(i,j,k): size (nxp1, ny, nzp1)
    ey: { ni: nxp1, nj: ny,   nk: nzp1, s1: ny*nzp1,   s2: nzp1, s3: 1 },
    // Ez(i,j,k): size (nxp1, nyp1, nz)
    ez: { ni: nxp1, nj: nyp1, nk: nz,   s1: nyp1*nz,   s2: nz,   s3: 1 },
  };

  // ─── Unified WebAssembly Memory Allocation ───────────────────────────────────
  // Calculate exact total floats needed for Fields + Coefficients
  const nhx = nxp1 * ny   * nz;
  const nhy = nx   * nyp1 * nz;
  const nhz = nx   * ny   * nzp1;
  const nex = nx   * nyp1 * nzp1;
  const ney = nxp1 * ny   * nzp1;
  const nez = nxp1 * nyp1 * nz;

  // 6 Fields + 18 Coefficients = 24 Arrays
  // 3 * nex arrays (Cexe, Cexhz, Cexhy), etc.
  const totalFieldFloats = nhx + nhy + nhz + nex + ney + nez;
  const totalCoeffFloats = 3*nex + 3*ney + 3*nez + 3*nhx + 3*nhy + 3*nhz;
  const totalFloats = totalFieldFloats + totalCoeffFloats;

  const totalBytes = totalFloats * Float64Array.BYTES_PER_ELEMENT;
  const wasmPages = Math.ceil(totalBytes / 65536) + 100; // +100 pages overhead padding

  const wasmMemory = new WebAssembly.Memory({
    initial: wasmPages,
    maximum: 65536,
    shared: true
  });
  const buffer = wasmMemory.buffer;

  // Distribute pointers (byte offsets)
  const pointers = {};
  let currentOffset = 0;

  pointers.Hx = currentOffset; currentOffset += nhx * 8;
  pointers.Hy = currentOffset; currentOffset += nhy * 8;
  pointers.Hz = currentOffset; currentOffset += nhz * 8;
  pointers.Ex = currentOffset; currentOffset += nex * 8;
  pointers.Ey = currentOffset; currentOffset += ney * 8;
  pointers.Ez = currentOffset; currentOffset += nez * 8;
  
  // Coefficient pointers
  pointers.Cexe = currentOffset; currentOffset += nex * 8;
  pointers.Cexhz = currentOffset; currentOffset += nex * 8;
  pointers.Cexhy = currentOffset; currentOffset += nex * 8;
  pointers.Ceye = currentOffset; currentOffset += ney * 8;
  pointers.Ceyhx = currentOffset; currentOffset += ney * 8;
  pointers.Ceyhz = currentOffset; currentOffset += ney * 8;
  pointers.Ceze = currentOffset; currentOffset += nez * 8;
  pointers.Cezhy = currentOffset; currentOffset += nez * 8;
  pointers.Cezhx = currentOffset; currentOffset += nez * 8;

  pointers.Chxh = currentOffset; currentOffset += nhx * 8;
  pointers.Chxez = currentOffset; currentOffset += nhx * 8;
  pointers.Chxey = currentOffset; currentOffset += nhx * 8;
  pointers.Chyh = currentOffset; currentOffset += nhy * 8;
  pointers.Chyex = currentOffset; currentOffset += nhy * 8;
  pointers.Chyez = currentOffset; currentOffset += nhy * 8;
  pointers.Chzh = currentOffset; currentOffset += nhz * 8;
  pointers.Chzey = currentOffset; currentOffset += nhz * 8;
  pointers.Chzex = currentOffset; currentOffset += nhz * 8;

  // Instantiate Field array views natively matching the WASM pointers
  const fields = {
    Hx: new Float64Array(buffer, pointers.Hx, nhx),
    Hy: new Float64Array(buffer, pointers.Hy, nhy),
    Hz: new Float64Array(buffer, pointers.Hz, nhz),
    Ex: new Float64Array(buffer, pointers.Ex, nex),
    Ey: new Float64Array(buffer, pointers.Ey, ney),
    Ez: new Float64Array(buffer, pointers.Ez, nez),
  };

  return {
    dx, dy, dz, dt,
    nx, ny, nz, nxp1, nyp1, nzp1,
    min_x, min_y, min_z,
    max_x: domainMaxX, max_y: domainMaxY, max_z: domainMaxZ,
    cellX, cellY, cellZ,
    strides,
    wasmMemory,
    buffer,
    pointers,
    fields,
    numberOfTimeSteps: params.numberOfTimeSteps,
  };
}

// ─── Inline index functions ─────────────────────────────────────────────────
// All arguments are 0-based.

/** Hx(i,j,k): size (nxp1, ny, nz) */
export const idxHx = (i, j, k, ny, nz) => i * ny * nz + j * nz + k;
/** Hy(i,j,k): size (nx, nyp1, nz) */
export const idxHy = (i, j, k, nyp1, nz) => i * nyp1 * nz + j * nz + k;
/** Hz(i,j,k): size (nx, ny, nzp1) */
export const idxHz = (i, j, k, ny, nzp1) => i * ny * nzp1 + j * nzp1 + k;
/** Ex(i,j,k): size (nx, nyp1, nzp1) */
export const idxEx = (i, j, k, nyp1, nzp1) => i * nyp1 * nzp1 + j * nzp1 + k;
/** Ey(i,j,k): size (nxp1, ny, nzp1) */
export const idxEy = (i, j, k, ny, nzp1) => i * ny * nzp1 + j * nzp1 + k;
/** Ez(i,j,k): size (nxp1, nyp1, nz) */
export const idxEz = (i, j, k, nyp1, nz) => i * nyp1 * nz + j * nz + k;

/**
 * JS equivalent of MATLAB sub2ind for a 3D array of size (dim1, dim2, dim3).
 * MATLAB is 1-indexed column-major; JS is 0-indexed row-major.
 * Accepts 1-based i,j,k (matching MATLAB) and returns 0-based flat JS index.
 * @param {number} i  - 1-based i index (MATLAB convention)
 * @param {number} j  - 1-based j index
 * @param {number} k  - 1-based k index
 * @param {number} d2 - size of second dimension
 * @param {number} d3 - size of third dimension
 */
export const sub2ind = (i, j, k, d2, d3) => (i - 1) * d2 * d3 + (j - 1) * d3 + (k - 1);

/**
 * JS equivalent of MATLAB create_linear_index_list.
 * Generates flat JS indices for a 3D sub-region [is..ie] × [js..je] × [ks..ke]
 * using 1-based MATLAB indices converted to 0-based JS row-major.
 *
 * Ordering: k outermost, j middle, i innermost (matches MATLAB loop order).
 *
 * @param {number} is,ie,js,je,ks,ke - 1-based inclusive MATLAB index ranges
 * @param {number} d2,d3 - array second and third dimensions (in JS row-major)
 * @returns {Int32Array} flat JS indices (0-based)
 */
export function createLinearIndexList(is, ie, js, je, ks, ke, d2, d3) {
  const iSize = ie - is + 1;
  const jSize = je - js + 1;
  const kSize = ke - ks + 1;
  const count = iSize * jSize * kSize;
  const fi = new Int32Array(count);
  let ind = 0;
  for (let mk = ks; mk <= ke; mk++) {
    for (let mj = js; mj <= je; mj++) {
      for (let mi = is; mi <= ie; mi++) {
        fi[ind++] = sub2ind(mi, mj, mk, d2, d3);
      }
    }
  }
  return fi;
}

/**
 * Convert physical coordinates (meters) to 1-based MATLAB grid indices.
 * Used when initializing sources, samplers, etc.
 */
export function coordToIndex(coord, domainMin, cellSize) {
  return Math.round((coord - domainMin) / cellSize) + 1;
}
