# Contribution 1 — WebAssembly (Wasm) FDTD Engine

**Component:** Multithreaded WebAssembly compute core (`wasm-cpu`)
**Location:** `js_fdtd/assembly/`, `js_fdtd/src/`, `js_fdtd/src/backends/wasmCpu.js`

---

## Purpose

The primary objective of this component is to provide the **high-performance numerical core** of the FDTD (Finite‑Difference Time‑Domain) Inverted‑F Antenna simulator. It rebuilds a legacy single‑threaded MATLAB solver as a maximum‑throughput JavaScript + WebAssembly engine that runs on Node.js.

This component serves two roles:

1. **The production solver** — it integrates Maxwell's equations in the time domain over a discretized 3D Yee grid (default ≈ 39 × 136 × 136 cells), driving the entire simulation pipeline from scenario loading through post‑processing (DFT, S‑parameters, far‑field transform).
2. **The golden reference** — it is the f64 ground‑truth implementation against which the GPU backends (WebGPU and CUDA) are validated within a relative tolerance of `1e‑3`. Its index math and update equations are the canonical contract that every other backend reproduces.

The headline goal was to deliver throughput rivaling natively‑compiled C++ while staying inside the JavaScript/web ecosystem, eliminating the 10‑minute MATLAB runtime.

---

## What Was Done

- **Ported the full FDTD physics pipeline from MATLAB to a modular Node.js engine.** Each stage of the solver is isolated in its own module: grid construction, material mapping, update‑coefficient generation, CPML absorbing boundaries, source excitation, field/voltage/current sampling, DFT, S‑parameter extraction, and the near‑to‑far‑field transform.
- **Implemented the hot inner loop as WebAssembly kernels** (`updateH` / `updateE`) written in AssemblyScript and compiled to `build/fdtd_kernels.wasm`.
- **Built a multithreaded cluster time loop** (`runFDTDCluster`) that partitions the grid across all available CPU cores using Node.js `worker_threads`.
- **Engineered a unified shared‑memory topology** in which all six field arrays (Eₓ, Eᵧ, E_z, Hₓ, Hᵧ, H_z) and the FDTD coefficient arrays live in a single ≈190 MB `SharedArrayBuffer`, eliminating garbage‑collection churn.
- **Designed a custom `Atomics`‑based phase barrier** to synchronize the dependent E↔H update half‑steps across worker threads with microsecond precision.
- **Wrapped the engine behind the pluggable backend contract** as `wasm-cpu`, exposing a uniform `{ name, isAvailable(), async *run(problem) }` interface so it integrates with the backend registry and falls back gracefully when GPU backends are unavailable.
- **Achieved a measured ~22 ms per step (~150 s for a 7000‑step run)**, down from the MATLAB baseline of ~85 ms per step (10+ minutes).

---

## How It Was Done

### Architecture & Methodology

**1. Yee‑grid FDTD formulation.** The solver discretizes 3D space into a staggered Yee grid. Each time step advances the simulation in two dependent half‑steps: the magnetic field **H** is updated from the curl of **E**, then the electric field **E** is updated from the curl of **H** (with CPML correction and source injection). The update coefficients and SI physical constants are matched exactly to the original MATLAB reference to guarantee numerical equivalence.

**2. Bare‑metal WebAssembly kernels.** The innermost triple‑nested loops over the voxel grid are the dominant cost, so they were rewritten in AssemblyScript. The kernels (`assembly/fdtd_kernels.ts`) operate directly on raw WebAssembly linear memory via `usize` byte‑offset pointers and inline `load<f64>` / `store<f64>` helpers — this bypasses JavaScript bounds checking and array‑object indirection, emitting tight native floating‑point opcodes. The kernels are parameterized by per‑worker slice bounds (`p_nx_start` / `p_nx_end`) so each thread updates only its X‑partition of the grid.

**3. Shared‑memory, zero‑GC design.** Standard JS typed arrays would trigger V8 garbage‑collection sweeps that introduce catastrophic micro‑pauses across billions of FLOPs. To avoid this entirely, all field and coefficient data is mapped into one contiguous `SharedArrayBuffer` backed by `WebAssembly.Memory`. Workers address fields by byte offset into this shared buffer, so there is zero data copying between threads and zero GC interference during the time loop.

**4. Multithreaded cluster with futex barriers.** The grid is partitioned along the X axis across N worker threads (typically `os.cpus().length − 1`). The two FDTD half‑steps are data‑dependent (H depends on E and vice versa), so the workers must rendezvous twice per step. Rather than the slow `postMessage` channel, a custom hardware barrier built on `Atomics.wait` / `Atomics.notify` synchronizes the phases. Each worker instantiates the same WASM module over the shared memory and runs its slice of `updateH`, hits the barrier, runs its slice of `updateE`, and hits the barrier again.

**5. Post‑processing.** After the time loop, sampled voltage/current time series are transformed via a running DFT into the frequency domain, S‑parameters (S₁₁ / return loss) are computed, and a near‑to‑far‑field transform yields radiated power, directivity, and the 3D radiation pattern.

**6. Backend integration.** The engine is exposed through `wasmCpu.js`, which implements the backend contract and yields `ProgressSnapshot` objects (step, percent, elapsed, sampled voltage) per batch — feeding the live dashboard over Server‑Sent Events without blocking the event loop.

### Engineering Notes

- **Precision:** f64 throughout — this is what makes it the golden reference.
- **SIMD caveat:** explicit 128‑bit `v128` SIMD was disabled on Windows Node because it triggered access‑violation traps during V8/TurboFan optimization when bound against `SharedArrayBuffer`s concurrently across many threads. The engine therefore runs optimized multithreaded **scalar** math (the build script retains the `--enable simd` flag for platforms where it is safe).
- **Known fragility:** because `runFDTDCluster` runs its barrier on the same thread that serves HTTP, frequent synchronous `GET /status` polling during a run can desync the `Atomics.waitAsync` barrier; normal SSE streaming is unaffected.

---

## Technologies & Resources Used

| Category | Tools / Resources |
|----------|-------------------|
| **Languages** | JavaScript (ESM), AssemblyScript (TypeScript‑derived → Wasm) |
| **Runtime** | Node.js |
| **Parallelism** | Node.js `worker_threads`, `Atomics` (futex‑style `wait` / `notify`), `SharedArrayBuffer` |
| **Compute** | WebAssembly (`WebAssembly.Memory`, shared linear memory, f64 kernels) |
| **Build toolchain** | AssemblyScript compiler (`asc`) with `--optimize --optimizeLevel 3 --enable simd --enable threads --importMemory --sharedMemory`; `@assemblyscript/loader` |
| **Numerics / method** | Finite‑Difference Time‑Domain (FDTD), staggered Yee grid, Convolutional PML (CPML) absorbing boundary, running DFT, S‑parameter & near‑to‑far‑field post‑processing |
| **Reference source** | The original MATLAB FDTD solver (`matlab_codes/`) — source of truth for constants and update equations |
| **Interfaces** | Backend contract (`src/backends/CONTRACT.md`), `ProgressSnapshot` streaming for the SSE dashboard |

---

*This component is the foundation of the simulation stack: it defines the physics, the data layout, and the numerical contract that the CUDA and WebGPU contributions both implement against.*
