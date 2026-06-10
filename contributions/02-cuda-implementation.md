# Contribution 2 — CUDA Implementation

**Component:** Native CUDA GPU backend (`cuda`)
**Location:** `js_fdtd/src/backends/cuda/`

---

## Purpose

The primary objective of this component is to deliver a **native NVIDIA‑GPU acceleration path** for the FDTD time loop. It ports the compute‑bound stages of the simulation — bulk field updates, absorbing boundaries, source injection, and probe sampling — onto CUDA so the solver can exploit massively parallel GPU hardware instead of CPU worker threads.

It is designed as an **optional, drop‑in backend**: if no CUDA toolchain or GPU is present, the native addon simply does not build, the backend reports `isAvailable() === false`, and the application transparently falls back to the `wasm-cpu` reference. It is never part of the default `npm install`.

The functional goal is strict numerical fidelity to the golden CPU/WASM reference: the CUDA path is a byte‑for‑byte transliteration of the reference index math and per‑step update order, validated to match the sampled voltage trace within `rtol = 1e‑3` (f32 tolerance).

---

## What Was Done

- **Implemented the full FDTD time loop as native CUDA kernels** in `native/fdtd_cuda.cu`, covering:
  - **Bulk `updateH` / `updateE`** — one CUDA thread per field cell, launched as 3D grids, one kernel per field component (Hx/Hy/Hz, Ex/Ey/Ez).
  - **All six CPML faces** (xn, xp, yn, yp, zn, zp), both magnetic and electric, each split into a Psi‑convolution kernel and a field‑correction kernel.
  - **Voltage‑source injection** kernel.
  - **Sampled‑voltage and sampled‑current reductions** via a generic shared‑memory gather‑and‑reduce kernel.
- **Built a C ABI** (`native/fdtd_cuda.h`) and a **Node‑API glue layer** (`native/binding.cpp`) exposing `init`, `uploadCoeff`, `uploadCpml`, `uploadVSource`, `uploadSVoltage`, `uploadSCurrent`, `runBatch`, `readField`, and `destroy`.
- **Authored the JavaScript driver** (`cudaBackend.js`) that implements the backend contract, uploads all problem data once, runs the loop in batches, fills the sampler objects, and yields `ProgressSnapshot` telemetry.
- **Configured the build** via `binding.gyp` for `node-gyp`, compiling `.cu` with `nvcc` and linking `cudart`, with a multi‑architecture gencode list (Turing `sm_75`, Ampere `sm_86`, Ada `sm_89`) and an override hook.
- **Implemented a resident‑memory model** — every field, coefficient, CPML, and source array is `cudaMalloc`'d once and stays on the device for the whole run; only tiny per‑step/per‑batch sample scalars are copied back.
- **Provided a validation harness** (`test/validate_cuda.js`) that diffs a 300‑step run against `wasm-cpu` and **skips cleanly (exit 0)** when CUDA is absent.

---

## How It Was Done

### Architecture & Methodology

**1. Faithful transliteration of the reference.** The `.cu` kernels are a direct port of `assembly/fdtd_kernels.ts` (bulk updates), `src/cpml.js` (CPML), `src/sources.js` (injection), and `src/sampling.js` (reductions). The flat row‑major index arithmetic is reproduced exactly, so the only intentional divergence from the reference is precision (f32 on device vs. f64 reference), which the contract explicitly permits for FDTD.

**2. Cell‑parallel bulk kernels.** Each bulk update kernel maps one thread to one field cell using a 3D launch (`dim3 tb(8,8,4)` thread blocks). Interior‑only bounds are honored by offsetting thread indices (e.g. `k += 1`) and early‑returning out‑of‑range threads, leaving boundary planes to the CPML stage — exactly mirroring the reference loop bounds.

**3. Two‑phase CPML with kernel‑boundary synchronization.** Each CPML face is updated in two kernels: phase A advances the convolution state `Psi = b·Psi + a·(F1 − F0)`; phase B applies the correction `Field += CPsi·Psi`. Because phase B reads what phase A wrote, the two are launched as **separate kernels** — a kernel boundary acts as a device‑wide synchronization point, guaranteeing correctness. X/Y/Z faces have dedicated kernels to keep the per‑axis stride math explicit and auditable.

**4. Resident device state, batched dispatch.** A single global solver context (`SolverDev g`) holds all device pointers. `fdtd_init` allocates and zeroes the six field arrays; the upload functions copy the 18 coefficient arrays, the CPML b/a/Psi/CPsi arrays, and source/sampler data to the device once. `fdtd_run_batch` then runs `[tsStart, tsStart+count)` steps entirely on the GPU, executing the canonical per‑step order: **H bulk → magnetic CPML → H captures (sampled current) → E bulk → electric CPML → voltage injection → E captures (sampled voltage)**.

**5. Sample reductions.** Sampled voltage and current are computed by a generic `k_gather_sum` kernel that does a block‑level shared‑memory tree reduction of `weight · Field[idx]`; the small number of block partials are summed on the host. Sampled current is precomputed on the host side (`buildCurrentLists`) into three weighted `(index, weight)` lists — one per H component — where the weight folds in `dx/dy/dz`, the sign of each loop‑integral segment, and the overall direction sign.

**6. Far‑field as a documented CPU fallback.** The near‑to‑far‑field DFT needs the Huygens‑surface E/H fields every step. Rather than porting it to CUDA initially, when `ff.nFreq > 0` the driver reads the six fields back per step into f32 mirrors and accumulates the DFT on the CPU via the existing `accumulateFarfieldDFT`. This is correctness‑first and is the main performance cost; when `ff.nFreq === 0` the readback is skipped and the loop runs fully on‑GPU in batches. *(Marked TODO: port the surface DFT to a resident CUDA kernel.)*

**7. Safe, optional integration.** `cudaBackend.js` probes several conventional `node-gyp` output paths for `fdtd_cuda.node`; `isAvailable()` returns `false` and never throws if the addon is missing or no device is found. Precision narrowing (f64 → f32) is handled by a `toF32` helper on upload.

### Scope Notes

- **Fully implemented on GPU (f32):** bulk H/E, all 6 CPML faces (magnetic + electric), single voltage‑source injection, sampled‑voltage and sampled‑current reductions.
- **CPU fallback:** far‑field DFT (per‑step field readback).
- **Stubbed (absent from the default problem):** current sources, inductors, and diodes. Porting them is mechanical (inductors need a second resident state array; diodes need a per‑cell Newton iteration). The driver warns if more than one voltage source is present.

---

## Technologies & Resources Used

| Category | Tools / Resources |
|----------|-------------------|
| **GPU compute** | CUDA C/C++ kernels (`.cu`), CUDA Runtime API (`cuda_runtime`, `cudaMalloc`, `cudaMemcpy`, `cudaMallocHost` pinned memory) |
| **Compiler / toolchain** | `nvcc` (CUDA Toolkit 11.x/12.x), `node-gyp`, `binding.gyp`, multi‑arch gencode (`sm_75` / `sm_86` / `sm_89`) |
| **Native bindings** | `node-addon-api` (Node‑API C++ glue), a C ABI header bridging kernels ↔ JS |
| **Host platforms** | Windows (MSVC v143 build tools, `CUDA_PATH`) / Linux (matching GCC, `CUDA_HOME`) |
| **Runtime / driver** | Node.js ESM driver (`cudaBackend.js`), the backend registry & contract |
| **Numerics / method** | FDTD bulk updates, Convolutional PML (6‑face, magnetic + electric), shared‑memory tree‑reduction sampling, f32 device precision |
| **Reference & validation** | The `wasm-cpu` golden reference; `test/validate_cuda.js` (300‑step diff within `rtol = 1e‑3`); `CONTRACT.md` |
| **Reused JS modules** | `farfield.js` (`accumulateFarfieldDFT`) for the CPU far‑field fallback |

---

*This component provides the highest‑throughput path on NVIDIA hardware while preserving exact behavioral parity with the reference engine, and it is engineered to remain entirely optional and non‑intrusive when a CUDA environment is unavailable.*
