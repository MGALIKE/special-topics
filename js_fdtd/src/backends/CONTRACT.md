# FDTD Backend Contract

This is the single source of truth that **every** compute backend (wasm-cpu,
webgpu, cuda) must obey. The goal: identical numerical results (to f32/f64
tolerance) regardless of which backend runs the time loop. The existing
multi-threaded WASM engine (`src/fdtdSolver.js`) is the **golden reference** —
all other backends are validated by diffing sampled outputs against it.

> If you are a backend author: do NOT invent new index math. Reproduce exactly
> what is described here and what lives in `src/fdtd_kernels` / `src/cpml.js` /
> `src/sampling.js` / `src/sources.js` / `src/farfield.js`.

---

## 1. Grid & memory layout

All arrays are **flat, row-major, 0-based**. Built in `src/grid.js`.

Dimensions: `nx, ny, nz` and `nxp1=nx+1, nyp1=ny+1, nzp1=nz+1`.

Field component sizes (number of f64/f32 elements):

| Field | dims          | length (`n*`) |
|-------|---------------|---------------|
| Hx    | (nxp1,ny,nz)  | nhx = nxp1*ny*nz     |
| Hy    | (nx,nyp1,nz)  | nhy = nx*nyp1*nz     |
| Hz    | (nx,ny,nzp1)  | nhz = nx*ny*nzp1     |
| Ex    | (nx,nyp1,nzp1)| nex = nx*nyp1*nzp1   |
| Ey    | (nxp1,ny,nzp1)| ney = nxp1*ny*nzp1   |
| Ez    | (nxp1,nyp1,nz)| nez = nxp1*nyp1*nz   |

Flat index for a component of size (d1,d2,d3): `idx = i*d2*d3 + j*d3 + k`.
Helper forms (matching `src/grid.js`):
```
idxHx(i,j,k) = i*ny*nz     + j*nz   + k      // (nxp1,ny,nz)
idxHy(i,j,k) = i*nyp1*nz   + j*nz   + k      // (nx,nyp1,nz)
idxHz(i,j,k) = i*ny*nzp1   + j*nzp1 + k      // (nx,ny,nzp1)
idxEx(i,j,k) = i*nyp1*nzp1 + j*nzp1 + k      // (nx,nyp1,nzp1)
idxEy(i,j,k) = i*ny*nzp1   + j*nzp1 + k      // (nxp1,ny,nzp1)
idxEz(i,j,k) = i*nyp1*nz   + j*nz   + k      // (nxp1,nyp1,nz)
```

There are **18 coefficient arrays** (same sizes as the field they update),
3 per field component:
- Ex: `Cexe, Cexhz, Cexhy`  (size nex)
- Ey: `Ceye, Ceyhx, Ceyhz`  (size ney)
- Ez: `Ceze, Cezhy, Cezhx`  (size nez)
- Hx: `Chxh, Chxey, Chxez`  (size nhx)
- Hy: `Chyh, Chyez, Chyex`  (size nhy)
- Hz: `Chzh, Chzex, Chzey`  (size nhz)

The WASM engine packs all 24 arrays (6 fields + 18 coeffs) into one shared
`WebAssembly.Memory` at byte offsets in `grid.pointers` (see `src/grid.js`
lines 132-162). GPU backends instead receive plain `Float64Array`/`Float32Array`
views via `grid.fields` and the `coeffs` object; upload each to its own GPU
storage buffer. **Precision: GPU backends use f32** (cast on upload). f32 is
physically fine for FDTD; validate S11 against the f64 reference within ~0.1 dB.

---

## 2. Per-time-step update order (CRITICAL — must match exactly)

For `ts = 0 .. numberOfTimeSteps-1`:

1. **updateH** (magnetic bulk) — see `assembly/fdtd_kernels.ts::updateH`.
   For every field cell:
   ```
   Hx = Chxh*Hx + Chxey*(Ey[k+1]-Ey[k]) + Chxez*(Ez[j+1]-Ez[j])
   Hy = Chyh*Hy + Chyez*(Ez[i+1]-Ez[i]) + Chyex*(Ex[k+1]-Ex[k])
   Hz = Chzh*Hz + Chzex*(Ex[j+1]-Ex[j]) + Chzey*(Ey[i+1]-Ey[i])
   ```
   Loop bounds (0-based): Hx over i∈[0,nxp1), j∈[0,ny), k∈[0,nz);
   Hy over i∈[0,nx), j∈[0,nyp1), k∈[0,nz);
   Hz over i∈[0,nx), j∈[0,ny), k∈[0,nzp1).
2. **updateMagneticCPML** — `src/cpml.js::updateMagneticCPML` (6 faces).
3. **H captures** (main thread in ref): `captureMagneticFields`,
   `captureSampledCurrents` — `src/sampling.js`.
4. **updateE** (electric bulk) — `assembly/fdtd_kernels.ts::updateE`.
   ```
   Ex = Cexe*Ex + Cexhz*(Hz[j]-Hz[j-1]) + Cexhy*(Hy[k]-Hy[k-1])
   Ey = Ceye*Ey + Ceyhx*(Hx[k]-Hx[k-1]) + Ceyhz*(Hz[i]-Hz[i-1])
   Ez = Ceze*Ez + Cezhy*(Hy[i]-Hy[i-1]) + Cezhx*(Hx[j]-Hx[j-1])
   ```
   Loop bounds: Ex over i∈[0,nx), j∈[1,ny), k∈[1,nz);
   Ey over i∈[1,nxp1), j∈[0,ny), k∈[1,nz);
   Ez over i∈[1,nxp1), j∈[1,nyp1), k∈[0,nz).
   (Interior only; boundary planes stay 0 / handled by CPML.)
5. **updateElectricCPML** — `src/cpml.js::updateElectricCPML` (6 faces).
6. **Source injection**: `injectVoltageSources`, `injectCurrentSources`,
   `updateInductors`, `updateDiodes` — `src/sources.js`. For the default
   problem only one voltage source is active. Injection adds
   `Cexs[n]*voltage_per_e_field[ts]` at the source's `field_indices[n]`.
7. **E captures**: `captureElectricFields`, `captureSampledVoltages`.
8. **Far-field DFT accumulate**: `accumulateFarfieldDFT` — `src/farfield.js`.

Sources, samplers, and far-field accumulators are tiny relative to the bulk.
A GPU backend may keep fields resident on the GPU for the whole loop and only
read back per-step scalars (sampled voltage/current) plus the far-field
accumulators at the end. Source `field_indices` + per-step waveform values can
be uploaded once as small buffers.

---

## 3. The Problem object (what backends receive)

Backends are invoked through the registry with a single `problem` object:

```
problem = {
  grid,        // from buildGrid(): dims, strides, pointers, fields, dt, ...
  coeffs,      // { Cexe, Cexhz, ... } 18 Float64Array
  cpml,        // from initCPML(): 6 faces of b/a/Psi/CPsi arrays
  samplers,    // { sampledEFields, sampledHFields, sampledVoltages, sampledCurrents }
  sources,     // { voltageSources, currentSources, inductors, diodes }
  ff,          // from initFarfield(): DFT accumulator arrays
  wasmBuffer,  // compiled wasm bytes (only used by wasm-cpu backend)
  options,     // { batchSize, ... }
}
```

A backend is an object:
```
{
  name: 'webgpu',
  isAvailable(): Promise<boolean>,   // feature-detect (GPU present, etc.)
  async *run(problem): AsyncGenerator<ProgressSnapshot>,
}
```

`run()` MUST, by the time it returns:
- fill each sampler's `.sampled_value[ts]` for all ts,
- accumulate `ff` DFT arrays identically to `accumulateFarfieldDFT`.

`ProgressSnapshot` (yielded every `batchSize` steps and on the last step):
```
{ step, total, elapsed, percent, voltage, time_ns, backend }
```

---

## 4. Validation protocol

`test/validate_backend.js` (to be added) runs a SMALL grid (short
`numberOfTimeSteps`, e.g. 300) on `wasm-cpu` and on the candidate backend, then
asserts the sampled voltage trace matches within rtol=1e-3 (f32). Far-field and
S11 are compared within ~0.1 dB. Do NOT consider a backend "done" until it
passes this.
