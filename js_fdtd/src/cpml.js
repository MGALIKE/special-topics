// CPML (Convolutional Perfectly Matched Layer) absorbing boundary condition.
// Mirrors MATLAB lines 1294-1676 (init) and 1935-2190 (per-step update).
//
// For each of 6 faces (xn, xp, yn, yp, zn, zp):
//   1. Compute 1D b/a coefficient arrays from polynomial conductivity profile.
//   2. Allocate 3D Psi (convolution field) arrays, stored as flat Float64Array.
//   3. In the time loop: update Psi, then correct H/E fields.
//
// Psi array layout: (ncells, nj, nk) → flat index = ci*nj*nk + j*nk + k
//   (cell index outermost, matching the i-loop over ncells in MATLAB)
//
// CRITICAL: After CPML init, the FDTD coefficients Ceyhz, Cezhy, Chyez, Chzey
// (and their analogues in y/z faces) are divided by kappa at each affected index.
// This is done HERE and modifies the coeffs arrays in-place.

import { CONSTANTS } from './constants.js';

/**
 * Initialize all CPML faces.
 * @param {object} boundary - boundary parameters
 * @param {object} coeffs   - FDTD update coefficients (modified in-place)
 * @param {object} grid
 * @returns {object} cpml - state for all 6 faces (or null if face is PEC)
 */
export function initCPML(boundary, coeffs, grid) {
  const { eps_0, mu_0 } = CONSTANTS;
  const { nx, ny, nz, nxp1, nyp1, nzp1, dx, dy, dz, dt } = grid;
  const p_order    = boundary.cpml_order;
  const sigma_ratio = boundary.cpml_sigma_factor;
  const kappa_max  = boundary.cpml_kappa_max;
  const alpha_min  = boundary.cpml_alpha_min;
  const alpha_max  = boundary.cpml_alpha_max;

  const cpml = {};

  // ─── XN face ─────────────────────────────────────────────────────────────────
  if (boundary.type_xn === 'cpml') {
    const nc = boundary.cpml_cells_xn;
    cpml.xn = _initFace_x(
      'xn', nc, nx, ny, nz, nxp1, nyp1, nzp1, dx, dt,
      p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, mu_0,
      coeffs, /* direction: from inside outward (rho descending) */ false
    );
  }

  // ─── XP face ─────────────────────────────────────────────────────────────────
  if (boundary.type_xp === 'cpml') {
    const nc = boundary.cpml_cells_xp;
    cpml.xp = _initFace_x(
      'xp', nc, nx, ny, nz, nxp1, nyp1, nzp1, dx, dt,
      p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, mu_0,
      coeffs, /* direction: from inside outward (rho ascending) */ true
    );
  }

  // ─── YN face ─────────────────────────────────────────────────────────────────
  if (boundary.type_yn === 'cpml') {
    const nc = boundary.cpml_cells_yn;
    cpml.yn = _initFace_y(
      'yn', nc, nx, ny, nz, nxp1, nyp1, nzp1, dy, dt,
      p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, mu_0,
      coeffs, false
    );
  }

  // ─── YP face ─────────────────────────────────────────────────────────────────
  if (boundary.type_yp === 'cpml') {
    const nc = boundary.cpml_cells_yp;
    cpml.yp = _initFace_y(
      'yp', nc, nx, ny, nz, nxp1, nyp1, nzp1, dy, dt,
      p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, mu_0,
      coeffs, true
    );
  }

  // ─── ZN face ─────────────────────────────────────────────────────────────────
  if (boundary.type_zn === 'cpml') {
    const nc = boundary.cpml_cells_zn;
    cpml.zn = _initFace_z(
      'zn', nc, nx, ny, nz, nxp1, nyp1, nzp1, dz, dt,
      p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, mu_0,
      coeffs, false
    );
  }

  // ─── ZP face ─────────────────────────────────────────────────────────────────
  if (boundary.type_zp === 'cpml') {
    const nc = boundary.cpml_cells_zp;
    cpml.zp = _initFace_z(
      'zp', nc, nx, ny, nz, nxp1, nyp1, nzp1, dz, dt,
      p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, mu_0,
      coeffs, true
    );
  }

  return cpml;
}

// ─── Private: compute 1D b/a arrays for a face ───────────────────────────────
function _cpml1D(ncells, cellSize, dt, p_order, sigma_ratio,
                 kappa_max, alpha_min, alpha_max, eps_scale, ascending) {
  const sigma_max = sigma_ratio * (p_order + 1) / (150 * Math.PI * cellSize);

  const b_e = new Float64Array(ncells);
  const a_e = new Float64Array(ncells);
  const b_m = new Float64Array(ncells);
  const a_m = new Float64Array(ncells);
  const kappa_e = new Float64Array(ncells);
  const kappa_m = new Float64Array(ncells);

  for (let ci = 0; ci < ncells; ci++) {
    // MATLAB: for xn, rho_e = ([ncells:-1:1]-0.75)/ncells, i.e. largest at ci=0
    //         for xp, rho_e = ([1:ncells]-0.75)/ncells,   i.e. smallest at ci=0
    const rho_e = ascending
      ? (ci + 1 - 0.75) / ncells
      : (ncells - ci - 0.75) / ncells;
    const rho_m = ascending
      ? (ci + 1 - 0.25) / ncells
      : (ncells - ci - 0.25) / ncells;

    const se = sigma_max * Math.pow(rho_e, p_order);
    const sm = (mu_0_local / eps_scale) * sigma_max * Math.pow(rho_m, p_order);

    const ke = 1 + (kappa_max - 1) * Math.pow(rho_e, p_order);
    const km = 1 + (kappa_max - 1) * Math.pow(rho_m, p_order);

    const ae = alpha_min + (alpha_max - alpha_min) * (1 - rho_e);
    const am = (mu_0_local / eps_scale) * (alpha_min + (alpha_max - alpha_min) * (1 - rho_m));

    b_e[ci] = Math.exp(-(dt / eps_scale) * (se / ke + ae));
    a_e[ci] = (1 / cellSize) * (b_e[ci] - 1) * se / (ke * (se + ke * ae));

    b_m[ci] = Math.exp(-(dt / mu_0_local) * (sm / km + am));
    a_m[ci] = (1 / cellSize) * (b_m[ci] - 1) * sm / (km * (sm + km * am));

    kappa_e[ci] = ke;
    kappa_m[ci] = km;
  }

  return { b_e, a_e, b_m, a_m, kappa_e, kappa_m };
}

// Capture mu_0 at module scope for use in _cpml1D
const mu_0_local = 4 * Math.PI * 1e-7;

// ─── XN/XP face init ─────────────────────────────────────────────────────────
function _initFace_x(face, nc, nx, ny, nz, nxp1, nyp1, nzp1,
                     dx, dt, p_order, sigma_ratio, kappa_max, alpha_min, alpha_max,
                     eps_0, mu_0, coeffs, ascending) {
  const { b_e, a_e, b_m, a_m, kappa_e, kappa_m } =
    _cpml1D(nc, dx, dt, p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, ascending);

  // Psi arrays: (nc, ny_or_nyp1, nzp1_or_nz)
  // Psi_hyx: size (nc, nyp1, nz) - modifies Hy
  // Psi_hzx: size (nc, ny, nzp1) - modifies Hz
  const Psi_hyx = new Float64Array(nc * nyp1 * nz);
  const Psi_hzx = new Float64Array(nc * ny  * nzp1);
  // Psi_eyx: size (nc, ny, nzp1) - modifies Ey
  // Psi_ezx: size (nc, nyp1, nz) - modifies Ez
  const Psi_eyx = new Float64Array(nc * ny   * nzp1);
  const Psi_ezx = new Float64Array(nc * nyp1 * nz);

  // CPsi coefficients: slices of Hy/Hz/Ey/Ez update coefficients × dx
  // MATLAB: CPsi_hyx_xn = Chyez(1:nc,:,:)*dx  (for xn)
  //         CPsi_hyx_xp = Chyez(nxp1-nc:nx,:,:)*dx  (for xp)
  const CPsi_hyx = new Float64Array(nc * nyp1 * nz);
  const CPsi_hzx = new Float64Array(nc * ny   * nzp1);
  const CPsi_eyx = new Float64Array(nc * ny   * nzp1);
  const CPsi_ezx = new Float64Array(nc * nyp1 * nz);

  // Which i-indices in the full arrays correspond to this CPML face?
  // For xn: i_start = 0 (MATLAB 1), but Ey/Ez start at index 1 (MATLAB 2)
  // For xp: n_st = nx - nc, field Ey/Ez at i_start = n_st (MATLAB n_st+1 onwards)
  const i_m_start = ascending ? (nx - nc) : 0;        // H-field (Hy, Hz) start
  const i_e_start = ascending ? (nx - nc) : 0;        // E-field correction start (Ey at i+1..nc+1 for xn)

  for (let ci = 0; ci < nc; ci++) {
    // For xn: MATLAB Chyez(ci+1,:,:) → JS Chyez[ci*nyp1*nz + ...]
    //         CPsi_hyx_xn = Chyez(1:nc)*dx
    // For xp: MATLAB Chyez(nx-nc+ci+1,:,:)
    const i_h = i_m_start + ci; // 0-based i into Hy/Hz array (size nx, nyp1, nz)
    const i_e_adj = ascending ? i_e_start + ci : i_e_start + ci + 1; // Ey/Ez adjusted (xn: skip boundary)

    for (let j = 0; j < nyp1; j++) {
      for (let k = 0; k < nz; k++) {
        CPsi_hyx[ci * nyp1 * nz + j * nz + k] =
          coeffs.Chyez[i_h * nyp1 * nz + j * nz + k] * dx;
      }
    }
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nzp1; k++) {
        CPsi_hzx[ci * ny * nzp1 + j * nzp1 + k] =
          coeffs.Chzey[i_h * ny * nzp1 + j * nzp1 + k] * dx;
      }
    }
    if (i_e_adj >= 0 && i_e_adj < nxp1) {
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          CPsi_eyx[ci * ny * nzp1 + j * nzp1 + k] =
            coeffs.Ceyhz[i_e_adj * ny * nzp1 + j * nzp1 + k] * dx;
        }
      }
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          CPsi_ezx[ci * nyp1 * nz + j * nz + k] =
            coeffs.Cezhy[i_e_adj * nyp1 * nz + j * nz + k] * dx;
        }
      }
    }
  }

  // Divide FDTD coefficients by kappa in CPML region (in-place)
  for (let ci = 0; ci < nc; ci++) {
    const i_h = i_m_start + ci;
    const i_e = ascending ? (nx - nc + ci) : (ci + 1); // xp: nx-nc+ci, xn: ci+1 (MATLAB ci+1, 0-based)
    const ke = kappa_e[ci];
    const km = kappa_m[ci];

    // H-field coefficients: Chyez, Chzey (Hy at i_h, Hz at i_h)
    for (let j = 0; j < nyp1; j++) {
      for (let k = 0; k < nz; k++) {
        coeffs.Chyez[i_h * nyp1 * nz + j * nz + k] /= km;
      }
    }
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nzp1; k++) {
        coeffs.Chzey[i_h * ny * nzp1 + j * nzp1 + k] /= km;
      }
    }
    // E-field coefficients: Ceyhz, Cezhy at i_e (MATLAB i+1, so 0-based = ci+1 for xn)
    if (i_e >= 0 && i_e < nxp1) {
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          coeffs.Ceyhz[i_e * ny * nzp1 + j * nzp1 + k] /= ke;
        }
      }
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          coeffs.Cezhy[i_e * nyp1 * nz + j * nz + k] /= ke;
        }
      }
    }
  }

  return {
    face, nc, b_e, a_e, b_m, a_m,
    Psi_hyx, Psi_hzx, Psi_eyx, Psi_ezx,
    CPsi_hyx, CPsi_hzx, CPsi_eyx, CPsi_ezx,
    i_m_start, i_e_start: ascending ? (nx - nc) : 1, // 0-based start for E corrections
    ascending,
    ny, nz, nxp1, nyp1, nzp1, nx,
  };
}

// ─── YN/YP face init ─────────────────────────────────────────────────────────
function _initFace_y(face, nc, nx, ny, nz, nxp1, nyp1, nzp1,
                     dy, dt, p_order, sigma_ratio, kappa_max, alpha_min, alpha_max,
                     eps_0, mu_0, coeffs, ascending) {
  const { b_e, a_e, b_m, a_m, kappa_e, kappa_m } =
    _cpml1D(nc, dy, dt, p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, ascending);

  // MATLAB: for yn, fields are indexed (:, ci+1, :) for E and (:, ci, :) for H
  // Psi_ezy: (nxp1, nc, nz), Psi_exy: (nx, nc, nzp1)
  // Psi_hzy: (nx, nc, nzp1), Psi_hxy: (nxp1, nc, nz)
  // Layout: (ni, nc, nk) - flat index = i*nc*nk + ci*nk + k

  const Psi_ezy = new Float64Array(nxp1 * nc * nz);
  const Psi_exy = new Float64Array(nx   * nc * nzp1);
  const Psi_hzy = new Float64Array(nx   * nc * nzp1);
  const Psi_hxy = new Float64Array(nxp1 * nc * nz);

  const CPsi_ezy = new Float64Array(nxp1 * nc * nz);
  const CPsi_exy = new Float64Array(nx   * nc * nzp1);
  const CPsi_hzy = new Float64Array(nx   * nc * nzp1);
  const CPsi_hxy = new Float64Array(nxp1 * nc * nz);

  const j_m_start = ascending ? (ny - nc) : 0;
  const j_e_start = ascending ? (ny - nc) : 1;

  for (let ci = 0; ci < nc; ci++) {
    const j_h = j_m_start + ci;
    const j_e = ascending ? (ny - nc + ci) : (ci + 1);

    for (let i = 0; i < nxp1; i++) {
      for (let k = 0; k < nz; k++) {
        CPsi_hxy[i * nc * nz + ci * nz + k] =
          coeffs.Chxez[i * ny * nz + j_h * nz + k] * dy;
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let k = 0; k < nzp1; k++) {
        CPsi_hzy[i * nc * nzp1 + ci * nzp1 + k] =
          coeffs.Chzex[i * nyp1 * nzp1 + j_h * nzp1 + k] * dy;
      }
    }
    if (j_e >= 0 && j_e < nyp1) {
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          CPsi_ezy[i * nc * nz + ci * nz + k] =
            coeffs.Cezhx[i * nyp1 * nz + j_e * nz + k] * dy;
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          CPsi_exy[i * nc * nzp1 + ci * nzp1 + k] =
            coeffs.Cexhz[i * nyp1 * nzp1 + j_e * nzp1 + k] * dy;
        }
      }
    }
  }

  // Divide FDTD coefficients by kappa
  for (let ci = 0; ci < nc; ci++) {
    const j_h = j_m_start + ci;
    const j_e = ascending ? (ny - nc + ci) : (ci + 1);
    const ke = kappa_e[ci];
    const km = kappa_m[ci];

    for (let i = 0; i < nxp1; i++) {
      for (let k = 0; k < nz; k++) {
        coeffs.Chxez[i * ny * nz + j_h * nz + k] /= km;
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let k = 0; k < nzp1; k++) {
        coeffs.Chzex[i * nyp1 * nzp1 + j_h * nzp1 + k] /= km;
      }
    }
    if (j_e >= 0 && j_e < nyp1) {
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          coeffs.Cezhx[i * nyp1 * nz + j_e * nz + k] /= ke;
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          coeffs.Cexhz[i * nyp1 * nzp1 + j_e * nzp1 + k] /= ke;
        }
      }
    }
  }

  return {
    face, nc, b_e, a_e, b_m, a_m,
    Psi_ezy, Psi_exy, Psi_hzy, Psi_hxy,
    CPsi_ezy, CPsi_exy, CPsi_hzy, CPsi_hxy,
    j_m_start, j_e_start,
    ascending,
    nx, ny, nz, nxp1, nyp1, nzp1,
  };
}

// ─── ZN/ZP face init ─────────────────────────────────────────────────────────
function _initFace_z(face, nc, nx, ny, nz, nxp1, nyp1, nzp1,
                     dz, dt, p_order, sigma_ratio, kappa_max, alpha_min, alpha_max,
                     eps_0, mu_0, coeffs, ascending) {
  const { b_e, a_e, b_m, a_m, kappa_e, kappa_m } =
    _cpml1D(nc, dz, dt, p_order, sigma_ratio, kappa_max, alpha_min, alpha_max, eps_0, ascending);

  // MATLAB: Psi_exz: (nx, nyp1, nc), Psi_eyz: (nxp1, ny, nc)
  //         Psi_hxz: (nxp1, ny, nc), Psi_hyz: (nx, nyp1, nc)
  // Layout: (ni, nj, nc) - flat index = i*nj*nc + j*nc + ci

  const Psi_exz = new Float64Array(nx   * nyp1 * nc);
  const Psi_eyz = new Float64Array(nxp1 * ny   * nc);
  const Psi_hxz = new Float64Array(nxp1 * ny   * nc);
  const Psi_hyz = new Float64Array(nx   * nyp1 * nc);

  const CPsi_exz = new Float64Array(nx   * nyp1 * nc);
  const CPsi_eyz = new Float64Array(nxp1 * ny   * nc);
  const CPsi_hxz = new Float64Array(nxp1 * ny   * nc);
  const CPsi_hyz = new Float64Array(nx   * nyp1 * nc);

  const k_m_start = ascending ? (nz - nc) : 0;
  const k_e_start = ascending ? (nz - nc) : 1;

  for (let ci = 0; ci < nc; ci++) {
    const k_h = k_m_start + ci;
    const k_e = ascending ? (nz - nc + ci) : (ci + 1);

    for (let i = 0; i < nxp1; i++) {
      for (let j = 0; j < ny; j++) {
        CPsi_hxz[i * ny * nc + j * nc + ci] =
          coeffs.Chxey[i * ny * nzp1 + j * nzp1 + k_h] * dz;
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nyp1; j++) {
        CPsi_hyz[i * nyp1 * nc + j * nc + ci] =
          coeffs.Chyex[i * nyp1 * nz + j * nz + k_h] * dz;
      }
    }
    if (k_e >= 0 && k_e < nzp1) {
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          CPsi_exz[i * nyp1 * nc + j * nc + ci] =
            coeffs.Cexhy[i * nyp1 * nzp1 + j * nzp1 + k_e] * dz;
        }
      }
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          CPsi_eyz[i * ny * nc + j * nc + ci] =
            coeffs.Ceyhx[i * ny * nzp1 + j * nzp1 + k_e] * dz;
        }
      }
    }
  }

  // Divide FDTD coefficients by kappa
  for (let ci = 0; ci < nc; ci++) {
    const k_h = k_m_start + ci;
    const k_e = ascending ? (nz - nc + ci) : (ci + 1);
    const ke = kappa_e[ci];
    const km = kappa_m[ci];

    for (let i = 0; i < nxp1; i++) {
      for (let j = 0; j < ny; j++) {
        coeffs.Chxey[i * ny * nzp1 + j * nzp1 + k_h] /= km;
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nyp1; j++) {
        coeffs.Chyex[i * nyp1 * nz + j * nz + k_h] /= km;
      }
    }
    if (k_e >= 0 && k_e < nzp1) {
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          coeffs.Cexhy[i * nyp1 * nzp1 + j * nzp1 + k_e] /= ke;
        }
      }
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          coeffs.Ceyhx[i * ny * nzp1 + j * nzp1 + k_e] /= ke;
        }
      }
    }
  }

  return {
    face, nc, b_e, a_e, b_m, a_m,
    Psi_exz, Psi_eyz, Psi_hxz, Psi_hyz,
    CPsi_exz, CPsi_eyz, CPsi_hxz, CPsi_hyz,
    k_m_start, k_e_start,
    ascending,
    nx, ny, nz, nxp1, nyp1, nzp1,
  };
}

// ─── Per-step: update magnetic CPML (called AFTER H-field bulk update) ───────
/**
 * @param {object} cpml    - from initCPML()
 * @param {object} fields  - {Hx, Hy, Hz, Ex, Ey, Ez}
 * @param {object} grid
 */
export function updateMagneticCPML(cpml, fields, grid) {
  const { Hx, Hy, Hz, Ex, Ey, Ez } = fields;
  const { nx, ny, nz, nxp1, nyp1, nzp1 } = grid;

  // ── XN ──────────────────────────────────────────────────────────────────────
  if (cpml.xn) {
    const { nc, b_m, a_m, Psi_hyx, Psi_hzx, CPsi_hyx, CPsi_hzx } = cpml.xn;
    for (let ci = 0; ci < nc; ci++) {
      const bm = b_m[ci], am = a_m[ci];
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          const pi = ci * nyp1 * nz + j * nz + k;
          const ez1 = Ez[(ci + 1) * nyp1 * nz + j * nz + k];
          const ez0 = Ez[ci       * nyp1 * nz + j * nz + k];
          Psi_hyx[pi] = bm * Psi_hyx[pi] + am * (ez1 - ez0);
        }
      }
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = ci * ny * nzp1 + j * nzp1 + k;
          const ey1 = Ey[(ci + 1) * ny * nzp1 + j * nzp1 + k];
          const ey0 = Ey[ci       * ny * nzp1 + j * nzp1 + k];
          Psi_hzx[pi] = bm * Psi_hzx[pi] + am * (ey1 - ey0);
        }
      }
    }
    // Hy(0..nc-1,:,:) += CPsi_hyx * Psi_hyx
    for (let n = 0; n < nc * nyp1 * nz; n++) {
      Hy[n] += CPsi_hyx[n] * Psi_hyx[n];
    }
    // Hz(0..nc-1,:,:) += CPsi_hzx * Psi_hzx
    for (let n = 0; n < nc * ny * nzp1; n++) {
      Hz[n] += CPsi_hzx[n] * Psi_hzx[n];
    }
  }

  // ── XP ──────────────────────────────────────────────────────────────────────
  if (cpml.xp) {
    const { nc, b_m, a_m, Psi_hyx, Psi_hzx, CPsi_hyx, CPsi_hzx, i_m_start } = cpml.xp;
    const n_st = i_m_start; // = nx - nc
    for (let ci = 0; ci < nc; ci++) {
      const bm = b_m[ci], am = a_m[ci];
      const ei = n_st + ci; // 0-based Ez/Ey index
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          const pi = ci * nyp1 * nz + j * nz + k;
          const ez1 = Ez[(ei + 1) * nyp1 * nz + j * nz + k];
          const ez0 = Ez[ei       * nyp1 * nz + j * nz + k];
          Psi_hyx[pi] = bm * Psi_hyx[pi] + am * (ez1 - ez0);
        }
      }
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = ci * ny * nzp1 + j * nzp1 + k;
          const ey1 = Ey[(ei + 1) * ny * nzp1 + j * nzp1 + k];
          const ey0 = Ey[ei       * ny * nzp1 + j * nzp1 + k];
          Psi_hzx[pi] = bm * Psi_hzx[pi] + am * (ey1 - ey0);
        }
      }
    }
    // Hy(n_st..nx-1,:,:) += CPsi_hyx * Psi_hyx
    for (let ci = 0; ci < nc; ci++) {
      const i = n_st + ci;
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          Hy[i * nyp1 * nz + j * nz + k] +=
            CPsi_hyx[ci * nyp1 * nz + j * nz + k] *
            Psi_hyx[ci * nyp1 * nz + j * nz + k];
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const i = n_st + ci;
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          Hz[i * ny * nzp1 + j * nzp1 + k] +=
            CPsi_hzx[ci * ny * nzp1 + j * nzp1 + k] *
            Psi_hzx[ci * ny * nzp1 + j * nzp1 + k];
        }
      }
    }
  }

  // ── YN ──────────────────────────────────────────────────────────────────────
  if (cpml.yn) {
    const { nc, b_m, a_m, Psi_hzy, Psi_hxy, CPsi_hzy, CPsi_hxy, j_m_start } = cpml.yn;
    for (let ci = 0; ci < nc; ci++) {
      const bm = b_m[ci], am = a_m[ci];
      const j_h = j_m_start + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          const pi = i * nc * nz + ci * nz + k;
          const hx1 = Hx[i * ny * nz + (j_h + 1) * nz + k];
          const hx0 = Hx[i * ny * nz + j_h       * nz + k];
          Psi_hxy[pi] = bm * Psi_hxy[pi] + am * (hx1 - hx0);
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = i * nc * nzp1 + ci * nzp1 + k;
          const hz1 = Hz[i * ny * nzp1 + (j_h + 1) * nzp1 + k];
          const hz0 = Hz[i * ny * nzp1 + j_h        * nzp1 + k];
          Psi_hzy[pi] = bm * Psi_hzy[pi] + am * (hz1 - hz0);
        }
      }
    }
    // Hz(:,0..nc-1,:) += CPsi_hzy * Psi_hzy
    for (let i = 0; i < nx; i++) {
      for (let ci = 0; ci < nc; ci++) {
        const j = j_m_start + ci;
        for (let k = 0; k < nzp1; k++) {
          Hz[i * ny * nzp1 + j * nzp1 + k] +=
            CPsi_hzy[i * nc * nzp1 + ci * nzp1 + k] *
            Psi_hzy[i * nc * nzp1 + ci * nzp1 + k];
        }
      }
    }
    // Hx(:,0..nc-1,:) += CPsi_hxy * Psi_hxy
    for (let i = 0; i < nxp1; i++) {
      for (let ci = 0; ci < nc; ci++) {
        const j = j_m_start + ci;
        for (let k = 0; k < nz; k++) {
          Hx[i * ny * nz + j * nz + k] +=
            CPsi_hxy[i * nc * nz + ci * nz + k] *
            Psi_hxy[i * nc * nz + ci * nz + k];
        }
      }
    }
  }

  // ── YP ──────────────────────────────────────────────────────────────────────
  if (cpml.yp) {
    const { nc, b_m, a_m, Psi_hzy, Psi_hxy, CPsi_hzy, CPsi_hxy, j_m_start } = cpml.yp;
    for (let ci = 0; ci < nc; ci++) {
      const bm = b_m[ci], am = a_m[ci];
      const j_h = j_m_start + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          const pi = i * nc * nz + ci * nz + k;
          const hx1 = Hx[i * ny * nz + (j_h + 1) * nz + k];
          const hx0 = Hx[i * ny * nz + j_h       * nz + k];
          Psi_hxy[pi] = bm * Psi_hxy[pi] + am * (hx1 - hx0);
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = i * nc * nzp1 + ci * nzp1 + k;
          const hz1 = Hz[i * ny * nzp1 + (j_h + 1) * nzp1 + k];
          const hz0 = Hz[i * ny * nzp1 + j_h        * nzp1 + k];
          Psi_hzy[pi] = bm * Psi_hzy[pi] + am * (hz1 - hz0);
        }
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let ci = 0; ci < nc; ci++) {
        const j = j_m_start + ci;
        for (let k = 0; k < nzp1; k++) {
          Hz[i * ny * nzp1 + j * nzp1 + k] +=
            CPsi_hzy[i * nc * nzp1 + ci * nzp1 + k] *
            Psi_hzy[i * nc * nzp1 + ci * nzp1 + k];
        }
      }
    }
    for (let i = 0; i < nxp1; i++) {
      for (let ci = 0; ci < nc; ci++) {
        const j = j_m_start + ci;
        for (let k = 0; k < nz; k++) {
          Hx[i * ny * nz + j * nz + k] +=
            CPsi_hxy[i * nc * nz + ci * nz + k] *
            Psi_hxy[i * nc * nz + ci * nz + k];
        }
      }
    }
  }

  // ── ZN ──────────────────────────────────────────────────────────────────────
  if (cpml.zn) {
    const { nc, b_m, a_m, Psi_hxz, Psi_hyz, CPsi_hxz, CPsi_hyz, k_m_start } = cpml.zn;
    for (let ci = 0; ci < nc; ci++) {
      const bm = b_m[ci], am = a_m[ci];
      const k_h = k_m_start + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          const pi = i * ny * nc + j * nc + ci;
          const ey1 = Ey[i * ny * nzp1 + j * nzp1 + (k_h + 1)];
          const ey0 = Ey[i * ny * nzp1 + j * nzp1 + k_h];
          Psi_hxz[pi] = bm * Psi_hxz[pi] + am * (ey1 - ey0);
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          const pi = i * nyp1 * nc + j * nc + ci;
          const ex1 = Ex[i * nyp1 * nzp1 + j * nzp1 + (k_h + 1)];
          const ex0 = Ex[i * nyp1 * nzp1 + j * nzp1 + k_h];
          Psi_hyz[pi] = bm * Psi_hyz[pi] + am * (ex1 - ex0);
        }
      }
    }
    // Hx(:,:,0..nc-1) += CPsi_hxz * Psi_hxz
    for (let i = 0; i < nxp1; i++) {
      for (let j = 0; j < ny; j++) {
        for (let ci = 0; ci < nc; ci++) {
          const k = k_m_start + ci;
          const pi = i * ny * nc + j * nc + ci;
          Hx[i * ny * nz + j * nz + k] += CPsi_hxz[pi] * Psi_hxz[pi];
        }
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nyp1; j++) {
        for (let ci = 0; ci < nc; ci++) {
          const k = k_m_start + ci;
          const pi = i * nyp1 * nc + j * nc + ci;
          Hy[i * nyp1 * nz + j * nz + k] += CPsi_hyz[pi] * Psi_hyz[pi];
        }
      }
    }
  }

  // ── ZP ──────────────────────────────────────────────────────────────────────
  if (cpml.zp) {
    const { nc, b_m, a_m, Psi_hxz, Psi_hyz, CPsi_hxz, CPsi_hyz, k_m_start } = cpml.zp;
    for (let ci = 0; ci < nc; ci++) {
      const bm = b_m[ci], am = a_m[ci];
      const k_h = k_m_start + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          const pi = i * ny * nc + j * nc + ci;
          const ey1 = Ey[i * ny * nzp1 + j * nzp1 + (k_h + 1)];
          const ey0 = Ey[i * ny * nzp1 + j * nzp1 + k_h];
          Psi_hxz[pi] = bm * Psi_hxz[pi] + am * (ey1 - ey0);
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          const pi = i * nyp1 * nc + j * nc + ci;
          const ex1 = Ex[i * nyp1 * nzp1 + j * nzp1 + (k_h + 1)];
          const ex0 = Ex[i * nyp1 * nzp1 + j * nzp1 + k_h];
          Psi_hyz[pi] = bm * Psi_hyz[pi] + am * (ex1 - ex0);
        }
      }
    }
    for (let i = 0; i < nxp1; i++) {
      for (let j = 0; j < ny; j++) {
        for (let ci = 0; ci < nc; ci++) {
          const k = k_m_start + ci;
          const pi = i * ny * nc + j * nc + ci;
          Hx[i * ny * nz + j * nz + k] += CPsi_hxz[pi] * Psi_hxz[pi];
        }
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nyp1; j++) {
        for (let ci = 0; ci < nc; ci++) {
          const k = k_m_start + ci;
          const pi = i * nyp1 * nc + j * nc + ci;
          Hy[i * nyp1 * nz + j * nz + k] += CPsi_hyz[pi] * Psi_hyz[pi];
        }
      }
    }
  }
}

// ─── Per-step: update electric CPML (called AFTER E-field bulk update) ────────
export function updateElectricCPML(cpml, fields, grid) {
  const { Hx, Hy, Hz, Ex, Ey, Ez } = fields;
  const { nx, ny, nz, nxp1, nyp1, nzp1 } = grid;

  // ── XN ──────────────────────────────────────────────────────────────────────
  if (cpml.xn) {
    const { nc, b_e, a_e, Psi_eyx, Psi_ezx, CPsi_eyx, CPsi_ezx, i_e_start } = cpml.xn;
    for (let ci = 0; ci < nc; ci++) {
      const be = b_e[ci], ae = a_e[ci];
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = ci * ny * nzp1 + j * nzp1 + k;
          const hz1 = Hz[(ci + 1) * ny * nzp1 + j * nzp1 + k];
          const hz0 = Hz[ci       * ny * nzp1 + j * nzp1 + k];
          Psi_eyx[pi] = be * Psi_eyx[pi] + ae * (hz1 - hz0);
        }
      }
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          const pi = ci * nyp1 * nz + j * nz + k;
          const hy1 = Hy[(ci + 1) * nyp1 * nz + j * nz + k];
          const hy0 = Hy[ci       * nyp1 * nz + j * nz + k];
          Psi_ezx[pi] = be * Psi_ezx[pi] + ae * (hy1 - hy0);
        }
      }
    }
    // Ey(1..nc+1,:,:) += CPsi_eyx * Psi_eyx (MATLAB index 2..nc+1 = JS 1..nc)
    for (let ci = 0; ci < nc; ci++) {
      const i = i_e_start + ci; // 0-based JS index for Ey (starts at 1 for xn)
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          Ey[i * ny * nzp1 + j * nzp1 + k] +=
            CPsi_eyx[ci * ny * nzp1 + j * nzp1 + k] *
            Psi_eyx[ci * ny * nzp1 + j * nzp1 + k];
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const i = i_e_start + ci;
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          Ez[i * nyp1 * nz + j * nz + k] +=
            CPsi_ezx[ci * nyp1 * nz + j * nz + k] *
            Psi_ezx[ci * nyp1 * nz + j * nz + k];
        }
      }
    }
  }

  // ── XP ──────────────────────────────────────────────────────────────────────
  if (cpml.xp) {
    const { nc, b_e, a_e, Psi_eyx, Psi_ezx, CPsi_eyx, CPsi_ezx, i_e_start } = cpml.xp;
    const n_st = i_e_start; // = nx - nc
    for (let ci = 0; ci < nc; ci++) {
      const be = b_e[ci], ae = a_e[ci];
      const ei = n_st + ci - 1; // Hz/Hy source index: n_st+ci-1..n_st+ci (0-based)
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = ci * ny * nzp1 + j * nzp1 + k;
          const hz1 = Hz[(ei + 1) * ny * nzp1 + j * nzp1 + k];
          const hz0 = Hz[ei       * ny * nzp1 + j * nzp1 + k];
          Psi_eyx[pi] = be * Psi_eyx[pi] + ae * (hz1 - hz0);
        }
      }
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          const pi = ci * nyp1 * nz + j * nz + k;
          const hy1 = Hy[(ei + 1) * nyp1 * nz + j * nz + k];
          const hy0 = Hy[ei       * nyp1 * nz + j * nz + k];
          Psi_ezx[pi] = be * Psi_ezx[pi] + ae * (hy1 - hy0);
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const i = n_st + ci;
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nzp1; k++) {
          Ey[i * ny * nzp1 + j * nzp1 + k] +=
            CPsi_eyx[ci * ny * nzp1 + j * nzp1 + k] *
            Psi_eyx[ci * ny * nzp1 + j * nzp1 + k];
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const i = n_st + ci;
      for (let j = 0; j < nyp1; j++) {
        for (let k = 0; k < nz; k++) {
          Ez[i * nyp1 * nz + j * nz + k] +=
            CPsi_ezx[ci * nyp1 * nz + j * nz + k] *
            Psi_ezx[ci * nyp1 * nz + j * nz + k];
        }
      }
    }
  }

  // ── YN ──────────────────────────────────────────────────────────────────────
  if (cpml.yn) {
    const { nc, b_e, a_e, Psi_ezy, Psi_exy, CPsi_ezy, CPsi_exy, j_e_start } = cpml.yn;
    for (let ci = 0; ci < nc; ci++) {
      const be = b_e[ci], ae = a_e[ci];
      const j_src = ci; // 0-based: source j and j+1
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          const pi = i * nc * nz + ci * nz + k;
          const hx1 = Hx[i * ny * nz + (j_src + 1) * nz + k];
          const hx0 = Hx[i * ny * nz + j_src       * nz + k];
          Psi_ezy[pi] = be * Psi_ezy[pi] + ae * (hx1 - hx0);
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = i * nc * nzp1 + ci * nzp1 + k;
          const hz1 = Hz[i * ny * nzp1 + (j_src + 1) * nzp1 + k];
          const hz0 = Hz[i * ny * nzp1 + j_src        * nzp1 + k];
          Psi_exy[pi] = be * Psi_exy[pi] + ae * (hz1 - hz0);
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const j = j_e_start + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          Ez[i * nyp1 * nz + j * nz + k] +=
            CPsi_ezy[i * nc * nz + ci * nz + k] *
            Psi_ezy[i * nc * nz + ci * nz + k];
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const j = j_e_start + ci;
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          Ex[i * nyp1 * nzp1 + j * nzp1 + k] +=
            CPsi_exy[i * nc * nzp1 + ci * nzp1 + k] *
            Psi_exy[i * nc * nzp1 + ci * nzp1 + k];
        }
      }
    }
  }

  // ── YP ──────────────────────────────────────────────────────────────────────
  if (cpml.yp) {
    const { nc, b_e, a_e, Psi_ezy, Psi_exy, CPsi_ezy, CPsi_exy, j_e_start, j_m_start } = cpml.yp;
    const n_st = j_m_start; // = ny - nc
    for (let ci = 0; ci < nc; ci++) {
      const be = b_e[ci], ae = a_e[ci];
      const j_src = n_st + ci - 1;
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          const pi = i * nc * nz + ci * nz + k;
          const hx1 = Hx[i * ny * nz + (j_src + 1) * nz + k];
          const hx0 = Hx[i * ny * nz + j_src       * nz + k];
          Psi_ezy[pi] = be * Psi_ezy[pi] + ae * (hx1 - hx0);
        }
      }
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          const pi = i * nc * nzp1 + ci * nzp1 + k;
          const hz1 = Hz[i * ny * nzp1 + (j_src + 1) * nzp1 + k];
          const hz0 = Hz[i * ny * nzp1 + j_src        * nzp1 + k];
          Psi_exy[pi] = be * Psi_exy[pi] + ae * (hz1 - hz0);
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const j = n_st + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let k = 0; k < nz; k++) {
          Ez[i * nyp1 * nz + j * nz + k] +=
            CPsi_ezy[i * nc * nz + ci * nz + k] *
            Psi_ezy[i * nc * nz + ci * nz + k];
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const j = n_st + ci;
      for (let i = 0; i < nx; i++) {
        for (let k = 0; k < nzp1; k++) {
          Ex[i * nyp1 * nzp1 + j * nzp1 + k] +=
            CPsi_exy[i * nc * nzp1 + ci * nzp1 + k] *
            Psi_exy[i * nc * nzp1 + ci * nzp1 + k];
        }
      }
    }
  }

  // ── ZN ──────────────────────────────────────────────────────────────────────
  if (cpml.zn) {
    const { nc, b_e, a_e, Psi_exz, Psi_eyz, CPsi_exz, CPsi_eyz, k_e_start } = cpml.zn;
    for (let ci = 0; ci < nc; ci++) {
      const be = b_e[ci], ae = a_e[ci];
      const k_src = ci;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          const pi = i * nyp1 * nc + j * nc + ci;
          const hy1 = Hy[i * nyp1 * nz + j * nz + (k_src + 1)];
          const hy0 = Hy[i * nyp1 * nz + j * nz + k_src];
          Psi_exz[pi] = be * Psi_exz[pi] + ae * (hy1 - hy0);
        }
      }
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          const pi = i * ny * nc + j * nc + ci;
          const hx1 = Hx[i * ny * nz + j * nz + (k_src + 1)];
          const hx0 = Hx[i * ny * nz + j * nz + k_src];
          Psi_eyz[pi] = be * Psi_eyz[pi] + ae * (hx1 - hx0);
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const k = k_e_start + ci;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          Ex[i * nyp1 * nzp1 + j * nzp1 + k] +=
            CPsi_exz[i * nyp1 * nc + j * nc + ci] *
            Psi_exz[i * nyp1 * nc + j * nc + ci];
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const k = k_e_start + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          Ey[i * ny * nzp1 + j * nzp1 + k] +=
            CPsi_eyz[i * ny * nc + j * nc + ci] *
            Psi_eyz[i * ny * nc + j * nc + ci];
        }
      }
    }
  }

  // ── ZP ──────────────────────────────────────────────────────────────────────
  if (cpml.zp) {
    const { nc, b_e, a_e, Psi_exz, Psi_eyz, CPsi_exz, CPsi_eyz, k_e_start, k_m_start } = cpml.zp;
    const n_st = k_m_start; // = nz - nc
    for (let ci = 0; ci < nc; ci++) {
      const be = b_e[ci], ae = a_e[ci];
      const k_src = n_st + ci - 1;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          const pi = i * nyp1 * nc + j * nc + ci;
          const hy1 = Hy[i * nyp1 * nz + j * nz + (k_src + 1)];
          const hy0 = Hy[i * nyp1 * nz + j * nz + k_src];
          Psi_exz[pi] = be * Psi_exz[pi] + ae * (hy1 - hy0);
        }
      }
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          const pi = i * ny * nc + j * nc + ci;
          const hx1 = Hx[i * ny * nz + j * nz + (k_src + 1)];
          const hx0 = Hx[i * ny * nz + j * nz + k_src];
          Psi_eyz[pi] = be * Psi_eyz[pi] + ae * (hx1 - hx0);
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const k = n_st + ci;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nyp1; j++) {
          Ex[i * nyp1 * nzp1 + j * nzp1 + k] +=
            CPsi_exz[i * nyp1 * nc + j * nc + ci] *
            Psi_exz[i * nyp1 * nc + j * nc + ci];
        }
      }
    }
    for (let ci = 0; ci < nc; ci++) {
      const k = n_st + ci;
      for (let i = 0; i < nxp1; i++) {
        for (let j = 0; j < ny; j++) {
          Ey[i * ny * nzp1 + j * nzp1 + k] +=
            CPsi_eyz[i * ny * nc + j * nc + ci] *
            Psi_eyz[i * ny * nc + j * nc + ci];
        }
      }
    }
  }
}
