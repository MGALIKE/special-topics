// Material grid construction and property averaging.
// Mirrors MATLAB lines 388-595 of fdtd_solve_Ch9_ex9a_12032026.m
//
// Material properties live on different subgrids (Yee staggered grid):
//   eps_r_x, sigma_e_x : Ex field locations → size (nx, nyp1, nzp1), averaged over 4 cells in yz
//   eps_r_y, sigma_e_y : Ey field locations → size (nxp1, ny, nzp1), averaged over 4 cells in xz
//   eps_r_z, sigma_e_z : Ez field locations → size (nxp1, nyp1, nz), averaged over 4 cells in xy
//   mu_r_x,  sigma_m_x : Hx field locations → size (nxp1, ny, nz),  averaged over 2 cells in x
//   mu_r_y,  sigma_m_y : Hy field locations → size (nx, nyp1, nz),  averaged over 2 cells in y
//   mu_r_z,  sigma_m_z : Hz field locations → size (nx, ny, nzp1), averaged over 2 cells in z

/**
 * Build the 3D material index array (Uint8Array, row-major).
 * material[i*ny*nz + j*nz + k] = materialTypeIndex (1-based, matching materialTypes array)
 *
 * @param {object} grid  - from buildGrid()
 * @param {Array}  bricks  - array of brick geometry objects
 * @param {Array}  spheres - array of sphere geometry objects
 * @returns {Uint8Array} material_3d_space (0-indexed flat, values are 1-based type indices)
 */
export function buildMaterialGrid(grid, bricks, spheres) {
  const { nx, ny, nz, dx, dy, dz, min_x, min_y, min_z, cellX, cellY, cellZ } = grid;
  const mat = new Uint8Array(nx * ny * nz).fill(1); // default: air (type index 1)

  // ─── Spheres ─────────────────────────────────────────────────────────────────
  for (const s of spheres) {
    const r2 = s.radius * s.radius;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          const idx = i * ny * nz + j * nz + k;
          const dx2 = cellX[idx] - s.center_x;
          const dy2 = cellY[idx] - s.center_y;
          const dz2 = cellZ[idx] - s.center_z;
          if (dx2*dx2 + dy2*dy2 + dz2*dz2 <= r2) {
            mat[idx] = s.material_type;
          }
        }
      }
    }
  }

  // ─── Bricks ──────────────────────────────────────────────────────────────────
  // MATLAB: blx = round((min_x - domain.min_x)/dx)+1 (1-based)
  //         material_3d_space(blx:bux-1, bly:buy-1, blz:buz-1) = type
  // In JS (0-based): i in [blx-1 .. bux-2], etc.
  for (const b of bricks) {
    const blx = Math.round((b.min_x - min_x) / dx);     // 0-based start
    const bly = Math.round((b.min_y - min_y) / dy);
    const blz = Math.round((b.min_z - min_z) / dz);
    const bux = Math.round((b.max_x - min_x) / dx);     // 0-based exclusive end
    const buy = Math.round((b.max_y - min_y) / dy);
    const buz = Math.round((b.max_z - min_z) / dz);

    for (let i = blx; i < bux; i++) {
      if (i < 0 || i >= nx) continue;
      for (let j = bly; j < buy; j++) {
        if (j < 0 || j >= ny) continue;
        for (let k = blz; k < buz; k++) {
          if (k < 0 || k >= nz) continue;
          mat[i * ny * nz + j * nz + k] = b.material_type;
        }
      }
    }
  }

  return mat;
}

/**
 * Compute all material component arrays from the 3D material index grid.
 * Returns Float64Arrays for eps_r_x/y/z, mu_r_x/y/z, sigma_e_x/y/z, sigma_m_x/y/z.
 *
 * Averaging formulas (match MATLAB exactly):
 *   eps_r_x(i,j,k):  0.25-average of cells (i,j,k),(i,j-1,k),(i,j,k-1),(i,j-1,k-1)   [1-based]
 *   eps_r_y(i,j,k):  0.25-average of cells (i,j,k),(i-1,j,k),(i,j,k-1),(i-1,j,k-1)
 *   eps_r_z(i,j,k):  0.25-average of cells (i,j,k),(i-1,j,k),(i,j-1,k),(i-1,j-1,k)
 *   mu_r_x(i,j,k):   harmonic mean of cells (i,j,k) and (i-1,j,k)
 *   mu_r_y(i,j,k):   harmonic mean of cells (i,j,k) and (i,j-1,k)
 *   mu_r_z(i,j,k):   harmonic mean of cells (i,j,k) and (i,j,k-1)
 *
 * @param {Uint8Array} mat - material_3d_space from buildMaterialGrid()
 * @param {Array} materialTypes - array of {eps_r, mu_r, sigma_e, sigma_m}
 * @param {object} grid
 * @returns {object} materialComponents
 */
export function computeMaterialComponents(mat, materialTypes, grid) {
  const { nx, ny, nz, nxp1, nyp1, nzp1 } = grid;

  // Expand material type arrays for fast lookup
  const nmt = materialTypes.length;
  const t_eps_r   = new Float64Array(nmt + 1);
  const t_mu_r    = new Float64Array(nmt + 1);
  const t_sigma_e = new Float64Array(nmt + 1);
  const t_sigma_m = new Float64Array(nmt + 1);
  for (let ind = 0; ind < nmt; ind++) {
    t_eps_r[ind + 1]   = materialTypes[ind].eps_r;
    t_mu_r[ind + 1]    = materialTypes[ind].mu_r    || 1e-20; // avoid div-by-zero
    t_sigma_e[ind + 1] = materialTypes[ind].sigma_e;
    t_sigma_m[ind + 1] = materialTypes[ind].sigma_m || 1e-20;
  }

  // Helper: get material type index (1-based) from 0-based cell (i,j,k)
  // Clamps to boundary cell to handle edge indices.
  const mtype = (i, j, k) => {
    const ii = Math.max(0, Math.min(nx - 1, i));
    const jj = Math.max(0, Math.min(ny - 1, j));
    const kk = Math.max(0, Math.min(nz - 1, k));
    return mat[ii * ny * nz + jj * nz + kk];
  };

  // ─── Allocate ─────────────────────────────────────────────────────────────────
  // All initialized to 1.0 (air) for eps_r/mu_r, 0 for sigma
  const eps_r_x   = new Float64Array(nx   * nyp1 * nzp1).fill(1.0);
  const eps_r_y   = new Float64Array(nxp1 * ny   * nzp1).fill(1.0);
  const eps_r_z   = new Float64Array(nxp1 * nyp1 * nz  ).fill(1.0);
  const mu_r_x    = new Float64Array(nxp1 * ny   * nz  ).fill(1.0);
  const mu_r_y    = new Float64Array(nx   * nyp1 * nz  ).fill(1.0);
  const mu_r_z    = new Float64Array(nx   * ny   * nzp1).fill(1.0);
  const sigma_e_x = new Float64Array(nx   * nyp1 * nzp1);
  const sigma_e_y = new Float64Array(nxp1 * ny   * nzp1);
  const sigma_e_z = new Float64Array(nxp1 * nyp1 * nz  );
  const sigma_m_x = new Float64Array(nxp1 * ny   * nz  );
  const sigma_m_y = new Float64Array(nx   * nyp1 * nz  );
  const sigma_m_z = new Float64Array(nx   * ny   * nzp1);

  // ─── eps_r_x(i,j,k): 1-based i∈[1..nx], j∈[2..ny], k∈[2..nz]
  //   = 0.25*(mat(i,j,k) + mat(i,j-1,k) + mat(i,j,k-1) + mat(i,j-1,k-1))
  //   JS: 0-based i∈[0..nx-1], j∈[1..ny-1], k∈[1..nz-1]
  //   Array size: (nx, nyp1, nzp1), strides: nyp1*nzp1, nzp1, 1
  for (let i = 0; i < nx; i++) {
    for (let j = 1; j < ny; j++) {       // j: 1..ny-1 (MATLAB 2..ny)
      for (let k = 1; k < nz; k++) {     // k: 1..nz-1 (MATLAB 2..nz)
        const idx = i * nyp1 * nzp1 + j * nzp1 + k;
        const m1 = mtype(i, j,   k);
        const m2 = mtype(i, j-1, k);
        const m3 = mtype(i, j,   k-1);
        const m4 = mtype(i, j-1, k-1);
        eps_r_x[idx]   = 0.25 * (t_eps_r[m1]   + t_eps_r[m2]   + t_eps_r[m3]   + t_eps_r[m4]);
        sigma_e_x[idx] = 0.25 * (t_sigma_e[m1] + t_sigma_e[m2] + t_sigma_e[m3] + t_sigma_e[m4]);
      }
    }
  }

  // ─── eps_r_y(i,j,k): 1-based i∈[2..nx], j∈[1..ny], k∈[2..nz]
  //   = 0.25*(mat(i,j,k) + mat(i-1,j,k) + mat(i,j,k-1) + mat(i-1,j,k-1))
  //   JS: 0-based i∈[1..nx-1], j∈[0..ny-1], k∈[1..nz-1]
  //   Array size: (nxp1, ny, nzp1), strides: ny*nzp1, nzp1, 1
  for (let i = 1; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 1; k < nz; k++) {
        const idx = i * ny * nzp1 + j * nzp1 + k;
        const m1 = mtype(i,   j, k);
        const m2 = mtype(i-1, j, k);
        const m3 = mtype(i,   j, k-1);
        const m4 = mtype(i-1, j, k-1);
        eps_r_y[idx]   = 0.25 * (t_eps_r[m1]   + t_eps_r[m2]   + t_eps_r[m3]   + t_eps_r[m4]);
        sigma_e_y[idx] = 0.25 * (t_sigma_e[m1] + t_sigma_e[m2] + t_sigma_e[m3] + t_sigma_e[m4]);
      }
    }
  }

  // ─── eps_r_z(i,j,k): 1-based i∈[2..nx], j∈[2..ny], k∈[1..nz]
  //   = 0.25*(mat(i,j,k) + mat(i-1,j,k) + mat(i,j-1,k) + mat(i-1,j-1,k))
  //   JS: 0-based i∈[1..nx-1], j∈[1..ny-1], k∈[0..nz-1]
  //   Array size: (nxp1, nyp1, nz), strides: nyp1*nz, nz, 1
  for (let i = 1; i < nx; i++) {
    for (let j = 1; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        const idx = i * nyp1 * nz + j * nz + k;
        const m1 = mtype(i,   j,   k);
        const m2 = mtype(i-1, j,   k);
        const m3 = mtype(i,   j-1, k);
        const m4 = mtype(i-1, j-1, k);
        eps_r_z[idx]   = 0.25 * (t_eps_r[m1]   + t_eps_r[m2]   + t_eps_r[m3]   + t_eps_r[m4]);
        sigma_e_z[idx] = 0.25 * (t_sigma_e[m1] + t_sigma_e[m2] + t_sigma_e[m3] + t_sigma_e[m4]);
      }
    }
  }

  // ─── mu_r_x(i,j,k): harmonic mean of cells (i,j,k) and (i-1,j,k) (1-based i∈[2..nx])
  //   JS: 0-based i∈[1..nx-1]
  //   Array size: (nxp1, ny, nz), strides: ny*nz, nz, 1
  for (let i = 1; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        const idx = i * ny * nz + j * nz + k;
        const mu1 = t_mu_r[mtype(i,   j, k)];
        const mu2 = t_mu_r[mtype(i-1, j, k)];
        mu_r_x[idx] = 2 * mu1 * mu2 / (mu1 + mu2);
        const sm1 = t_sigma_m[mtype(i,   j, k)];
        const sm2 = t_sigma_m[mtype(i-1, j, k)];
        sigma_m_x[idx] = 2 * sm1 * sm2 / (sm1 + sm2);
      }
    }
  }

  // ─── mu_r_y(i,j,k): harmonic mean of cells (i,j,k) and (i,j-1,k) (1-based j∈[2..ny])
  //   JS: 0-based j∈[1..ny-1]
  //   Array size: (nx, nyp1, nz), strides: nyp1*nz, nz, 1
  for (let i = 0; i < nx; i++) {
    for (let j = 1; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        const idx = i * nyp1 * nz + j * nz + k;
        const mu1 = t_mu_r[mtype(i, j,   k)];
        const mu2 = t_mu_r[mtype(i, j-1, k)];
        mu_r_y[idx] = 2 * mu1 * mu2 / (mu1 + mu2);
        const sm1 = t_sigma_m[mtype(i, j,   k)];
        const sm2 = t_sigma_m[mtype(i, j-1, k)];
        sigma_m_y[idx] = 2 * sm1 * sm2 / (sm1 + sm2);
      }
    }
  }

  // ─── mu_r_z(i,j,k): harmonic mean of cells (i,j,k) and (i,j,k-1) (1-based k∈[2..nz])
  //   JS: 0-based k∈[1..nz-1]
  //   Array size: (nx, ny, nzp1), strides: ny*nzp1, nzp1, 1
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 1; k < nz; k++) {
        const idx = i * ny * nzp1 + j * nzp1 + k;
        const mu1 = t_mu_r[mtype(i, j, k)];
        const mu2 = t_mu_r[mtype(i, j, k-1)];
        mu_r_z[idx] = 2 * mu1 * mu2 / (mu1 + mu2);
        const sm1 = t_sigma_m[mtype(i, j, k)];
        const sm2 = t_sigma_m[mtype(i, j, k-1)];
        sigma_m_z[idx] = 2 * sm1 * sm2 / (sm1 + sm2);
      }
    }
  }

  return {
    eps_r_x, eps_r_y, eps_r_z,
    mu_r_x,  mu_r_y,  mu_r_z,
    sigma_e_x, sigma_e_y, sigma_e_z,
    sigma_m_x, sigma_m_y, sigma_m_z,
  };
}

/**
 * Apply PEC plate conductivity from zero-thickness bricks.
 * Mirrors MATLAB lines 568-595.
 *
 * A brick with min_x == max_x applies conductivity to Ey and Ez on that face.
 * A brick with min_y == max_y applies to Ex and Ez.
 * A brick with min_z == max_z applies to Ex and Ey.
 *
 * @param {Array} bricks
 * @param {Array} materialTypes
 * @param {object} mc  - material components from computeMaterialComponents
 * @param {object} grid
 */
export function applyPECPlates(bricks, materialTypes, mc, grid) {
  const { nx, ny, nz, nxp1, nyp1, nzp1, dx, dy, dz, min_x, min_y, min_z } = grid;
  const { sigma_e_x, sigma_e_y, sigma_e_z } = mc;

  for (const b of bricks) {
    const sigma_pec = materialTypes[b.material_type - 1].sigma_e;
    if (sigma_pec === 0) continue; // skip non-conductive bricks

    const blx = Math.round((b.min_x - min_x) / dx) + 1; // 1-based
    const bly = Math.round((b.min_y - min_y) / dy) + 1;
    const blz = Math.round((b.min_z - min_z) / dz) + 1;
    const bux = Math.round((b.max_x - min_x) / dx) + 1;
    const buy = Math.round((b.max_y - min_y) / dy) + 1;
    const buz = Math.round((b.max_z - min_z) / dz) + 1;

    if (blx === bux) {
      // Zero-thickness in x: apply to Ey(blx, bly:buy-1, blz:buz) and Ez(blx, bly:buy, blz:buz-1)
      // MATLAB: sigma_e_y(blx, bly:buy-1, blz:buz) = sigma_pec
      //         sigma_e_z(blx, bly:buy,   blz:buz-1) = sigma_pec
      // JS: 0-based i=blx-1, j=[bly-1..buy-2], k=[blz-1..buz-1]
      const i = blx - 1;
      if (i >= 0 && i < nxp1) {
        // sigma_e_y: size (nxp1, ny, nzp1)
        for (let j = bly - 1; j < buy - 1; j++) {
          for (let k = blz - 1; k < buz; k++) {
            if (j >= 0 && j < ny && k >= 0 && k < nzp1)
              sigma_e_y[i * ny * nzp1 + j * nzp1 + k] = sigma_pec;
          }
        }
        // sigma_e_z: size (nxp1, nyp1, nz)
        for (let j = bly - 1; j < buy; j++) {
          for (let k = blz - 1; k < buz - 1; k++) {
            if (j >= 0 && j < nyp1 && k >= 0 && k < nz)
              sigma_e_z[i * nyp1 * nz + j * nz + k] = sigma_pec;
          }
        }
      }
    }
    if (bly === buy) {
      // Zero-thickness in y
      const j = bly - 1;
      if (j >= 0 && j < nyp1) {
        // sigma_e_x: size (nx, nyp1, nzp1)
        for (let i = blx - 1; i < bux - 1; i++) {
          for (let k = blz - 1; k < buz; k++) {
            if (i >= 0 && i < nx && k >= 0 && k < nzp1)
              sigma_e_x[i * nyp1 * nzp1 + j * nzp1 + k] = sigma_pec;
          }
        }
        // sigma_e_z: size (nxp1, nyp1, nz)
        for (let i = blx - 1; i < bux; i++) {
          for (let k = blz - 1; k < buz - 1; k++) {
            if (i >= 0 && i < nxp1 && k >= 0 && k < nz)
              sigma_e_z[i * nyp1 * nz + j * nz + k] = sigma_pec;
          }
        }
      }
    }
    if (blz === buz) {
      // Zero-thickness in z
      const k = blz - 1;
      if (k >= 0 && k < nzp1) {
        // sigma_e_x: size (nx, nyp1, nzp1)
        for (let i = blx - 1; i < bux - 1; i++) {
          for (let j = bly - 1; j < buy; j++) {
            if (i >= 0 && i < nx && j >= 0 && j < nyp1)
              sigma_e_x[i * nyp1 * nzp1 + j * nzp1 + k] = sigma_pec;
          }
        }
        // sigma_e_y: size (nxp1, ny, nzp1)
        for (let i = blx - 1; i < bux; i++) {
          for (let j = bly - 1; j < buy - 1; j++) {
            if (i >= 0 && i < nxp1 && j >= 0 && j < ny)
              sigma_e_y[i * ny * nzp1 + j * nzp1 + k] = sigma_pec;
          }
        }
      }
    }
  }
}
