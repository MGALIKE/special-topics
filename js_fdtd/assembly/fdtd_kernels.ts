// FDTD Hot Loops for WebAssembly optimization.
// Compiled using AssemblyScript.
//
// These functions operate directly on raw WebAssembly Memory via `usize` pointers.
// This executes seamlessly shared across multi-core Node.js Clusters.

@inline
function load_f64(ptr: usize, idx: i32): f64 {
  return load<f64>(ptr + (<usize>idx << 3));
}

@inline
function store_f64(ptr: usize, idx: i32, val: f64): void {
  store<f64>(ptr + (<usize>idx << 3), val);
}

// Computes min and max
@inline function max(a: i32, b: i32): i32 { return a > b ? a : b; }

export function updateH(
  nx: i32, ny: i32, nz: i32, nxp1: i32, nyp1: i32, nzp1: i32,
  p_nxp1_start: i32, p_nxp1_end: i32,
  p_nx_start: i32, p_nx_end: i32,
  Hx: usize, Hy: usize, Hz: usize,
  Ex: usize, Ey: usize, Ez: usize,
  Chxh: usize, Chxey: usize, Chxez: usize,
  Chyh: usize, Chyez: usize, Chyex: usize,
  Chzh: usize, Chzex: usize, Chzey: usize
): void {
  const nynz = ny * nz;
  const nynzp1_ = ny * nzp1;
  const nyp1nz_ = nyp1 * nz;
  const nyp1nz = nyp1 * nz;
  const nyp1nzp1 = nyp1 * nzp1;
  const nynzp1 = ny * nzp1;

  // Hx
  for (let i = p_nxp1_start; i < p_nxp1_end; i++) {
    const hxBase = i * nynz;
    const eyBase = i * nynzp1_;
    const ezBaseJ0 = i * nyp1nz_;
    for (let j = 0; j < ny; j++) {
      const hxOff = hxBase + j * nz;
      const eyOff = eyBase + j * nzp1;
      const ezOff0 = ezBaseJ0 + j * nz;
      const ezOff1 = ezBaseJ0 + (j + 1) * nz;
      for (let k = 0; k < nz; k++) {
        const idx = hxOff + k;
        const eey1 = load_f64(Ey, eyOff + k + 1);
        const eey0 = load_f64(Ey, eyOff + k);
        const eez1 = load_f64(Ez, ezOff1 + k);
        const eez0 = load_f64(Ez, ezOff0 + k);

        let val = load_f64(Chxh, idx) * load_f64(Hx, idx)
                + load_f64(Chxey, idx) * (eey1 - eey0)
                + load_f64(Chxez, idx) * (eez1 - eez0);
        store_f64(Hx, idx, val);
      }
    }
  }

  // Hy
  for (let i = p_nx_start; i < p_nx_end; i++) {
    const hyBase = i * nyp1nz;
    const ezBase0 = i * nyp1nz_;
    const ezBase1 = (i + 1) * nyp1nz_;
    const exBase = i * nyp1nzp1;
    for (let j = 0; j < nyp1; j++) {
      const hyOff = hyBase + j * nz;
      const ezOff0 = ezBase0 + j * nz;
      const ezOff1 = ezBase1 + j * nz;
      const exOff = exBase + j * nzp1;
      for (let k = 0; k < nz; k++) {
        const idx = hyOff + k;
        const eez1 = load_f64(Ez, ezOff1 + k);
        const eez0 = load_f64(Ez, ezOff0 + k);
        const eex1 = load_f64(Ex, exOff + k + 1);
        const eex0 = load_f64(Ex, exOff + k);

        let val = load_f64(Chyh, idx) * load_f64(Hy, idx)
                + load_f64(Chyez, idx) * (eez1 - eez0)
                + load_f64(Chyex, idx) * (eex1 - eex0);
        store_f64(Hy, idx, val);
      }
    }
  }

  // Hz
  for (let i = p_nx_start; i < p_nx_end; i++) {
    const hzBase = i * nynzp1;
    const exBase = i * nyp1nzp1;
    const eyBase0 = i * nynzp1_;
    const eyBase1 = (i + 1) * nynzp1_;
    for (let j = 0; j < ny; j++) {
      const hzOff = hzBase + j * nzp1;
      const exOff0 = exBase + j * nzp1;
      const exOff1 = exBase + (j + 1) * nzp1;
      const eyOff0 = eyBase0 + j * nzp1;
      const eyOff1 = eyBase1 + j * nzp1;
      for (let k = 0; k < nzp1; k++) {
        const idx = hzOff + k;
        const eex1 = load_f64(Ex, exOff1 + k);
        const eex0 = load_f64(Ex, exOff0 + k);
        const eey1 = load_f64(Ey, eyOff1 + k);
        const eey0 = load_f64(Ey, eyOff0 + k);

        let val = load_f64(Chzh, idx) * load_f64(Hz, idx)
                + load_f64(Chzex, idx) * (eex1 - eex0)
                + load_f64(Chzey, idx) * (eey1 - eey0);
        store_f64(Hz, idx, val);
      }
    }
  }
}

export function updateE(
  nx: i32, ny: i32, nz: i32, nxp1: i32, nyp1: i32, nzp1: i32,
  p_nx_start: i32, p_nx_end: i32,
  p_nxp1_start: i32, p_nxp1_end: i32,
  Ex: usize, Ey: usize, Ez: usize,
  Hx: usize, Hy: usize, Hz: usize,
  Cexe: usize, Cexhz: usize, Cexhy: usize,
  Ceye: usize, Ceyhx: usize, Ceyhz: usize,
  Ceze: usize, Cezhy: usize, Cezhx: usize
): void {
  const nynz = ny * nz;
  const nynzp1 = ny * nzp1;
  const nyp1nz = nyp1 * nz;
  const nyp1nzp1 = nyp1 * nzp1;
  const nynzp1_ = ny * nzp1;
  const nyp1nz_ = nyp1 * nz;

  // Ex
  for (let i = p_nx_start; i < p_nx_end; i++) {
    const exBase = i * nyp1nzp1;
    const hzBase = i * nynzp1;
    const hyBase = i * nyp1nz;
    for (let j = 1; j < ny; j++) {
      const exOff = exBase + j * nzp1;
      const hzOff0 = hzBase + (j - 1) * nzp1;
      const hzOff1 = hzBase + j * nzp1;
      const hyOff = hyBase + j * nz;
      for (let k = 1; k < nz; k++) {
        const idx = exOff + k;
        const hhz1 = load_f64(Hz, hzOff1 + k);
        const hhz0 = load_f64(Hz, hzOff0 + k);
        const hhy1 = load_f64(Hy, hyOff + k);
        const hhy0 = load_f64(Hy, hyOff + k - 1);

        let val = load_f64(Cexe, idx) * load_f64(Ex, idx)
                + load_f64(Cexhz, idx) * (hhz1 - hhz0)
                + load_f64(Cexhy, idx) * (hhy1 - hhy0);
        store_f64(Ex, idx, val);
      }
    }
  }

  // Ey
  for (let i = max(1, p_nxp1_start); i < p_nxp1_end; i++) {
    const eyBase = i * nynzp1_;
    const hxBase = i * nynz;
    const hzBase0 = (i - 1) * nynzp1;
    const hzBase1 = i * nynzp1;
    for (let j = 0; j < ny; j++) {
      const eyOff = eyBase + j * nzp1;
      const hxOff = hxBase + j * nz;
      const hzOff0 = hzBase0 + j * nzp1;
      const hzOff1 = hzBase1 + j * nzp1;
      for (let k = 1; k < nz; k++) {
        const idx = eyOff + k;
        const hhx1 = load_f64(Hx, hxOff + k);
        const hhx0 = load_f64(Hx, hxOff + k - 1);
        const hhz1 = load_f64(Hz, hzOff1 + k);
        const hhz0 = load_f64(Hz, hzOff0 + k);

        let val = load_f64(Ceye, idx) * load_f64(Ey, idx)
                + load_f64(Ceyhx, idx) * (hhx1 - hhx0)
                + load_f64(Ceyhz, idx) * (hhz1 - hhz0);
        store_f64(Ey, idx, val);
      }
    }
  }

  // Ez
  for (let i = max(1, p_nxp1_start); i < p_nxp1_end; i++) {
    const ezBase = i * nyp1nz_;
    const hyBase0 = (i - 1) * nyp1nz;
    const hyBase1 = i * nyp1nz;
    const hxBase = i * nynz;
    for (let j = 1; j < ny; j++) {
      const ezOff = ezBase + j * nz;
      const hyOff0 = hyBase0 + j * nz;
      const hyOff1 = hyBase1 + j * nz;
      const hxOff0 = hxBase + (j - 1) * nz;
      const hxOff1 = hxBase + j * nz;
      for (let k = 0; k < nz; k++) {
        const idx = ezOff + k;
        const hhy1 = load_f64(Hy, hyOff1 + k);
        const hhy0 = load_f64(Hy, hyOff0 + k);
        const hhx1 = load_f64(Hx, hxOff1 + k);
        const hhx0 = load_f64(Hx, hxOff0 + k);

        let val = load_f64(Ceze, idx) * load_f64(Ez, idx)
                + load_f64(Cezhy, idx) * (hhy1 - hhy0)
                + load_f64(Cezhx, idx) * (hhx1 - hhx0);
        store_f64(Ez, idx, val);
      }
    }
  }
}
