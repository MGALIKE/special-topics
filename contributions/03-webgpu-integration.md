# Contribution 3 — WebGPU Integration

**Component:** Cross‑environment WebGPU GPU backend (`webgpu`)
**Location:** `js_fdtd/src/backends/webgpu/`

---

## Purpose

The primary objective of this component is to provide a **portable, vendor‑neutral GPU acceleration path** for the FDTD time loop that runs in **both the browser and Node.js** from a single, shared codebase. Where the CUDA backend targets NVIDIA hardware exclusively, the WebGPU integration runs on any GPU exposed through the WebGPU standard — Chrome/Edge desktop and mobile, Safari 18+, and Node via the Dawn native bindings.

This dual reach is what lets the Next.js dashboard offer in‑browser GPU simulation while the same engine code is validated headlessly on Node/Dawn against the golden `wasm-cpu` reference (precision f32, tolerance `rtol ≈ 1e‑3`).

---

## What Was Done

- **Authored the complete WGSL compute‑shader suite** (`kernels.wgsl.js`): `updateH` and `updateE` (bulk), a single generic parameterized `cpml` kernel serving all 6 faces × both components (magnetic + electric), plus `injectVoltage`, `sampleVoltage`, and `sampleCurrent`.
- **Built an environment‑agnostic driver** (`engine.js`) that takes any `GPUDevice`, uploads the fields, all 18 coefficients, the CPML `b`/`a`/`Psi`/`CPsi` arrays, and the source/sampler buffers, runs the whole loop, and reads back the per‑batch sampled scalars.
- **Provided two thin device‑acquisition adapters** over the same engine:
  - `nodeBackend.js` — acquires a device via the optional `webgpu` (Dawn) npm package.
  - `browserBackend.js` — acquires a device via `navigator.gpu`, importable directly by the Next.js app.
- **Validated the backend on real hardware** (Node/Dawn) against the CPU reference, resolving four concrete hardware/spec issues in the process (detailed below).
- **Ensured graceful optionality** — `isAvailable()` returns `false` (never throws) when the `webgpu` package or a suitable adapter is absent, so the registry falls back to `wasm-cpu`.
- **Wired the backend into the dashboard** so it appears in the Compute Engine selector and can run a simulation client‑side via dynamic import.

---

## How It Was Done

### Architecture & Methodology

**1. One engine, two environments.** The numerically significant code lives entirely in `engine.js` and `kernels.wgsl.js`, both written as **pure ESM with no Node‑only imports** (no `os`/`fs`). This is the key design decision: it means the Node backend and the browser backend share *identical numerics* and differ only in how they obtain a `GPUDevice`. The browser adapter calls `navigator.gpu.requestAdapter()`; the Node adapter dynamically imports the Dawn‑backed `webgpu` package and supports both its `create([])` factory and `navigator.gpu` shapes.

**2. Byte‑for‑byte WGSL kernels.** Every shader reproduces the reference index math and arithmetic exactly: bulk updates from `assembly/fdtd_kernels.ts`, CPML from `src/cpml.js`, injection from `src/sources.js`, and sampling from `src/sampling.js`. All arrays are flat, row‑major, with strides identical to the CPU reference. Grid dimensions ride in a shared `Dims` uniform block (`@group(0) @binding(0)`); per‑dispatch scalars that change between dispatches (CPML face parameters, the per‑step source value, the sample time index) ride in a small secondary `Params` uniform so the large storage bindings never need rebinding.

**3. Generic single CPML kernel.** Rather than 12 separate face/component shaders, a single parameterized `cpml` kernel implements the Psi recurrence `Psi = b·Psi + a·(F1 − F0)` followed by `Field += CPsi·Psi`, with strides and base offsets supplied per dispatch — covering all 6 faces × 2 components. Because each Psi cell maps to a unique field cell, the recurrence and correction are fused into one invocation per (face, component).

**4. Exact per‑step ordering.** The driver enforces the contract's update order: **H bulk → H CPML → H captures → E bulk → E CPML → source inject → E captures → far‑field DFT**. It relies on WebGPU's implicit per‑dispatch barrier (storage‑buffer visibility is guaranteed between successive `dispatchWorkgroups` calls within one compute pass) to sequence bulk → CPML within a single pass.

**5. Sampled voltage as a tree reduction.** `sampleVoltage` performs a single‑workgroup tree reduction of `Csvf · Σ Field[field_indices]` into a per‑step trace buffer, copied back into the sampler at the end (and the latest value per batch into the progress snapshot).

**6. Far‑field handled on CPU (v1).** As with the CUDA backend, the far‑field DFT is computed on the CPU via the shared `accumulateFarfieldDFT` by mapping the six field buffers back each step when `ff.nFreq > 0`. It is numerically identical to the reference (same f64 routine) but the per‑step readback is the known bottleneck. *(Marked TODO: move DFT accumulation on‑GPU.)*

### Real‑Hardware Fixes (Node/Dawn Validation)

Bringing the backend up on actual hardware required four concrete fixes:

1. **Enum globals.** Dawn does not auto‑install the WebGPU enum globals (`GPUBufferUsage`, `GPUMapMode`, `GPUShaderStage`, …) that browsers expose. The Node adapter copies them from the package's `globals` onto `globalThis`, otherwise the shared engine throws `GPUBufferUsage is not defined`.
2. **Raised storage‑buffer limit.** The bulk shaders bind 15 storage buffers in one stage, but the WebGPU default `maxStorageBuffersPerShaderStage` is only 8. `deviceDescriptor()` requests exactly 15 (clamped to what the adapter offers) at device creation, otherwise pipeline creation fails.
3. **Split fused CPML kernel.** The originally fused CPML kernel hit read/read‑write aliasing on real hardware; it was restructured to avoid the conflict.
4. **Reserved keyword.** The reserved WGSL keyword `pass` was renamed to compile cleanly.

Additionally, the Dawn native handles (`module`, `gpu`, `adapter`, `device`) are **pinned in module scope** — if left as locals they are garbage‑collected after `acquireDevice()` returns, freeing native state the running device depends on and producing a hard `SIGSEGV` (exit 139) mid‑run rather than a JS error.

### Scope Notes

- **Fully implemented & validated:** bulk H/E, all 6 CPML faces (both components), voltage‑source injection, sampled voltage, exact per‑step ordering, per‑batch progress snapshots.
- **Implemented but not independently validated here:** far‑field DFT (CPU fallback; the validation script disables far‑field to stay fast), sampled current (a generic weighted H‑sum kernel exists but the term lists are not yet generated in `engine.js`).
- **Stubbed / out of scope:** current sources, inductors, diodes (diodes would need the Newton‑Raphson solver from `src/sources.js`).

---

## Technologies & Resources Used

| Category | Tools / Resources |
|----------|-------------------|
| **GPU API** | WebGPU (`GPUDevice`, `GPUBuffer`, compute pipelines, `dispatchWorkgroups`, uniform & storage buffers, `requiredLimits`) |
| **Shading language** | WGSL (WebGPU Shading Language) compute shaders |
| **Browser runtime** | `navigator.gpu` (Chrome/Edge 113+, Safari 18+, Chrome on Android) |
| **Node runtime** | Optional `webgpu` npm package (Dawn native bindings) |
| **Frontend integration** | Next.js (App Router) + React client component / dynamic import in `visual/` |
| **Numerics / method** | FDTD bulk updates, generic parameterized Convolutional PML (6‑face, magnetic + electric), single‑workgroup tree‑reduction sampling, f32 device precision |
| **Reference & validation** | The `wasm-cpu` golden reference; `test/validate_webgpu.js` (≈300‑step diff within `rtol = 1e‑3`, skips cleanly when WebGPU is absent); `CONTRACT.md` |
| **Reused JS modules** | `farfield.js` (`accumulateFarfieldDFT`) for the CPU far‑field fallback |

---

*This component extends GPU acceleration beyond a single vendor and beyond the server: by sharing one validated engine across Node/Dawn and the browser, it powers both headless validation and live, client‑side simulation in the dashboard.*
