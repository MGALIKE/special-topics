// Physical constants (SI units)
// All values match MATLAB solver exactly.
export const CONSTANTS = Object.freeze({
  eps_0: 8.854187817e-12,   // permittivity of free space (F/m)
  mu_0:  4 * Math.PI * 1e-7, // permeability of free space (H/m)
  c:     2.99792458e8,       // speed of light (m/s)
  eta_0: 376.73031346177066, // intrinsic impedance of free space (Ω) = sqrt(mu_0/eps_0)
  q:     1.602e-19,          // electron charge (C)
  k_B:   1.38066e-23,        // Boltzmann constant (J/K)
});

// Derived: eta_0 = sqrt(mu_0/eps_0)
// Cross-check: sqrt(4*pi*1e-7 / 8.854187817e-12) ≈ 376.73...
