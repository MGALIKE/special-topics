# CUDA backend (native)

A native CUDA implementation of the FDTD time loop, conforming to
[`../CONTRACT.md`](../CONTRACT.md). It runs `updateH`, `updateE`, the 6-face
CPML updates, voltage-source injection, and sampled voltage/current reduction on
an NVIDIA GPU, keeping all field and coefficient arrays resident in device
global memory for the whole run.

It is **optional**: if the toolchain or a GPU is missing, the addon simply does
not build, `isAvailable()` returns `false`, and the app falls back to
`wasm-cpu`. Nothing here is part of the default `npm install` / `npm run build`.

## Files

| File | Purpose |
|------|---------|
| `native/fdtd_cuda.cu` | CUDA kernels: bulk `updateH`/`updateE`, 6-face CPML, source injection, sample reductions. f32. |
| `native/fdtd_cuda.h`  | C ABI between the kernels and the Node-API glue. |
| `native/binding.cpp`  | node-addon-api glue: `init`, `uploadCoeff/Cpml/VSource/SVoltage/SCurrent`, `runBatch`, `readField`, `destroy`. |
| `binding.gyp`         | node-gyp config; compiles `.cu` with `nvcc`, links `cudart`. |
| `cudaBackend.js`      | The JS Backend object the registry loads. Uploads once, loops in batches, fills samplers, yields ProgressSnapshots. |

## Prerequisites

1. **NVIDIA GPU** with a recent driver.
2. **CUDA Toolkit** (11.x or 12.x). `nvcc` must be on `PATH` and `CUDA_PATH`
   (Windows) / `CUDA_HOME` (Linux) set. The Windows installer sets `CUDA_PATH`.
3. **Windows:** Visual Studio Build Tools (MSVC v143, "Desktop development with
   C++"). **Linux:** matching GCC for your CUDA version.
4. **node-gyp** and **node-addon-api** available to the build:
   ```
   npm install --no-save node-gyp node-addon-api
   ```
   (Kept out of the committed `dependencies` so installs without CUDA are not
   affected. Install them only when you intend to build this backend.)

## Build

From the `js_fdtd/` directory:

```
npm install --no-save node-gyp node-addon-api
npm run build:cuda
```

`build:cuda` runs `node-gyp configure build --directory=src/backends/cuda`,
which produces `src/backends/cuda/build/Release/fdtd_cuda.node`.

If your GPU is not in the default gencode list (Turing `sm_75`, Ampere `sm_86`,
Ada `sm_89`), override it:

```
node-gyp configure build --directory=src/backends/cuda --cuda_arch="-gencode=arch=compute_70,code=sm_70"
```

## Opting in at runtime

The registry auto-selects backends in the order `webgpu → cuda → wasm-cpu`. To
force CUDA, pass `preferred: 'cuda'` to `selectBackend()`. If the addon built
and a GPU is present, `isAvailable()` returns true and it is used; otherwise the
app silently falls back.

## Validate

```
npm run build           # ensure build/fdtd_kernels.wasm exists (reference)
node test/validate_cuda.js
```

Runs a 300-step problem on both `wasm-cpu` (golden, f64) and `cuda` (f32) and
asserts the sampled voltage trace matches within `rtol = 1e-3`. If CUDA is
unavailable the script prints `SKIP` and exits 0.

## What is implemented vs TODO

**Fully implemented on GPU (f32):**
- `updateH` / `updateE` bulk (one thread per cell, 3D launch).
- All 6 CPML faces, magnetic and electric, Psi update + field correction.
- Single voltage-source injection.
- Sampled voltage (averaging reduction) and sampled current (loop-integral
  reduction) read back per step/batch.

**On CPU as a documented fallback:**
- **Far-field DFT** (`accumulateFarfieldDFT`). It needs the Huygens-surface E/H
  every step. When `ff.nFreq > 0`, `cudaBackend.js` reads the 6 fields back per
  step and accumulates on the CPU. This is correctness-first and is the main
  cost; **TODO: port the surface DFT to a CUDA kernel** that accumulates
  resident `re/im` buffers and copies them back once at the end. When
  `ff.nFreq === 0`, the readback is skipped and the loop runs fully on the GPU
  in batches.

**Stubbed (not in the default problem):**
- **Current sources, inductors, diodes.** The default problem has none. The
  voltage-source path covers the active case. Porting these is mechanical
  (inductors: a second resident state array `J*`; diodes: per-cell Newton
  iteration) — **TODO** if a problem enables them. `cudaBackend.js` warns if more
  than one voltage source is present (only the first is uploaded).

## Notes

- Precision is f32 throughout the device (CONTRACT §1 allows this; validation is
  against the f64 reference within f32 tolerance).
- The CPML `kappa` division of the FDTD coefficients is done on the CPU by
  `initCPML` (it mutates `coeffs` in place) *before* upload, so the device gets
  the already-divided coefficients — exactly like the reference.
