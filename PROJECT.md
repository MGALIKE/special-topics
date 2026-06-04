# FDTD Inverted‑F Antenna Simulation Stack — Full Project Guide

A complete, browser‑driven **3D FDTD (Finite‑Difference Time‑Domain)** electromagnetic
simulator for an **Inverted‑F Antenna (IFA)**, ported from a legacy MATLAB solver into a
high‑performance JavaScript / WebAssembly engine with a real‑time Next.js dashboard.

This document is the single, top‑level explanation of *everything* in the repository:
what the project does, how it is laid out, how the physics engine works, the pluggable
compute backends, the visualization frontend, how to run it, and the current known state.

> If you only want to *run* it, jump to **[Quick Start](#quick-start)**.

---

## 1. What this project is

The goal is to simulate how an Inverted‑F Antenna radiates and how well it is matched
(its return loss / S₁₁) across frequency, by integrating Maxwell's equations directly in
the time domain over a discretized 3D voxel grid (the FDTD method).

The original solver was a single‑threaded MATLAB program (preserved under
[matlab_codes/](matlab_codes/)). This repo rebuilds it as:

- **A multithreaded Node.js physics engine** ([js_fdtd/](js_fdtd/)) that runs the FDTD
  time loop across all CPU cores using `worker_threads` + `Atomics`, with the hot inner
  loop optionally executing as **WebAssembly** kernels over shared memory.
- **A pluggable backend layer** so the *same* physics can run on the CPU (`wasm-cpu`,
  the golden reference), a GPU via **WebGPU** (browser **and** Node/Dawn), or an NVIDIA
  GPU via a **CUDA** native addon.
- **A real‑time React/Next.js dashboard** ([visual/](visual/)) that streams the running
  simulation over Server‑Sent Events (SSE) and plots voltage transients, S‑parameters,
  Smith chart, and a 3D far‑field radiation pattern with Three.js.

### Performance snapshot

| Engine | Model | Per‑step | Total (7000 steps) |
|--------|-------|----------|--------------------|
| MATLAB base | Scalar, single‑threaded, interpreted | ~85 ms | 10+ minutes |
| **WASM scalar, multithreaded** | Shared memory + 16 worker threads + futex barriers | **~22 ms** | **~150 s** |

(See [README.md](README.md) for the original performance write‑up. Note: explicit 128‑bit
SIMD is disabled on Windows Node because it triggered access‑violation traps during V8
optimization; the engine runs optimized multithreaded *scalar* math.)

---

## 2. Repository layout

```
eski_branch/
├── PROJECT.md            ← this file
├── README.md             ← original high-level overview + run instructions
├── package.json          ← root launcher: `npm run dev` starts engine + dashboard together
├── dev.mjs               ← zero-dependency launcher for both processes
├── matlab_codes/         ← the original MATLAB FDTD solver (reference implementation)
│
├── js_fdtd/              ← the physics engine (Node.js)
│   ├── BACKENDS.md       ← authoritative doc for the pluggable compute backends
│   ├── package.json      ← engine scripts (serve / start / build / validate / build:cuda)
│   ├── assembly/         ← AssemblyScript sources for the WASM hot-loop kernels
│   ├── build/            ← compiled fdtd_kernels.wasm
│   ├── scenarios/        ← JSON problem definitions (geometry/materials/sources/steps)
│   ├── src/              ← the solver (see below)
│   └── test/             ← backend validation scripts (validate_webgpu.js, validate_cuda.js)
│
└── visual/               ← the Next.js dashboard (React + Three.js + Recharts)
    ├── app/              ← Next.js app router (page.tsx, layout.tsx, api/)
    ├── components/       ← dashboard widgets (charts, 3D scene, selectors)
    └── lib/              ← client logic (useSimulation hook, backends.ts, fft.ts)
```

---

## 3. The physics engine (`js_fdtd/src/`)

The solver is modular; each file owns one stage of the FDTD pipeline.

| File | Responsibility |
|------|----------------|
| [index.js](js_fdtd/src/index.js) | CLI entry. Loads a scenario, selects a backend, runs the time loop, then post‑processes (DFT → S‑parameters, far‑field). |
| [server.js](js_fdtd/src/server.js) | HTTP + SSE server (port 4000). Endpoints: `POST /simulate`, `GET /stream`, `GET /status`, `GET /results`, `GET /backends`. Zero external deps (Node `http` only). |
| [scenario.js](js_fdtd/src/scenario.js) | Loads a problem definition from `scenarios/*.json` and builds the full simulation problem (grid, coefficients, CPML, sources, samplers, far‑field). |
| [grid.js](js_fdtd/src/grid.js) | Builds the discretized 3D Yee grid (default ~39 × 136 × 136 cells). |
| [constants.js](js_fdtd/src/constants.js) | SI physical constants, matched exactly to the MATLAB values. |
| [materials.js](js_fdtd/src/materials.js) | Material grid, PEC plates, per‑component material averaging. |
| [coefficients.js](js_fdtd/src/coefficients.js) | FDTD update coefficients + lumped‑element (source/resistor) coefficients. |
| [cpml.js](js_fdtd/src/cpml.js) | Convolutional PML absorbing boundary (the open‑space truncation). |
| [sources.js](js_fdtd/src/sources.js) | Excitation waveforms and voltage sources. |
| [sampling.js](js_fdtd/src/sampling.js) | Field/voltage/current samplers (the probes that record the time series). |
| [fdtdSolver.js](js_fdtd/src/fdtdSolver.js) | The multithreaded cluster time loop (`runFDTDCluster`). |
| [fdtdWorker.js](js_fdtd/src/fdtdWorker.js) | Per‑worker thread body; instantiates the WASM module over shared memory. |
| [barrier.js](js_fdtd/src/barrier.js) | Custom `Atomics`‑based phase barrier synchronizing the E↔H update phases. |
| [dft.js](js_fdtd/src/dft.js) | Running DFT of sampled voltages/currents into the frequency domain. |
| [sparameters.js](js_fdtd/src/sparameters.js) | S‑parameter (S₁₁ / return loss) post‑processing. |
| [farfield.js](js_fdtd/src/farfield.js) | Near‑to‑far‑field transform: radiated power, directivity, patterns. |
| [visualization.js](js_fdtd/src/visualization.js) | Results formatting for output / the dashboard. |

### How the engine works (per time step)

1. **Setup** — a scenario JSON is loaded; the grid, material map, update coefficients,
   CPML boundary, sources, and samplers are built.
2. **Threaded time loop** — the grid is partitioned along X across N worker threads
   (`p_nx_start`/`p_nx_end`). Each step alternates two phases:
   - update **H** fields from **E**, hit the barrier;
   - update **E** fields from **H** (plus CPML correction + source injection), hit the
     barrier.
   The custom futex barrier (`Atomics.wait`/`notify`) synchronizes phases with
   microsecond precision instead of slow `postMessage`.
3. **Shared memory** — all six field arrays (Eₓ Eᵧ E_z, Hₓ Hᵧ H_z) and coefficients live
   in a single `SharedArrayBuffer` backed by `WebAssembly.Memory` (~190 MB, Float64), so
   there's zero GC churn and workers address fields by byte offset.
4. **Sampling** — voltage/current probes record a time series each step.
5. **Post‑processing** — after the loop, a DFT turns the time series into the frequency
   domain, S‑parameters are computed, and a near‑to‑far‑field transform yields the
   radiation pattern.

### Scenarios

Problems are data, not code. Each file in [js_fdtd/scenarios/](js_fdtd/scenarios/) fully
describes a run (geometry, materials, sources, boundary, frequency window, step count):

- `ifa-dualband-baseline.json` — default scenario.
- `ifa-finefreq-sweep.json` — finer frequency resolution.
- `ifa-fr4-substrate.json` — IFA on an FR4 dielectric substrate.
- `ifa-matlab-7000.json` — the full 7000‑step run matching the MATLAB reference.

Select one with `--scenario=<name>`, the `FDTD_SCENARIO` env var, or the dashboard's
scenario selector. `FDTD_STEPS` overrides the step count for quick smoke tests.

---

## 4. Compute backends

The time loop runs behind a **pluggable backend interface** so the same physics can run
on different hardware. All backends are validated against the CPU engine (the golden
reference) within `rtol = 1e-3` (f32). Full detail is in
[js_fdtd/BACKENDS.md](js_fdtd/BACKENDS.md).

| Backend | Runs on | Precision | Status |
|---------|---------|-----------|--------|
| `wasm-cpu` | Node, worker_threads + WASM | f64 | ✅ Reference, always available |
| `webgpu` | Browser **and** Node (Dawn) | f32 | ✅ Validated on Node/Dawn (matches wasm-cpu) |
| `cuda` | Node native addon (NVIDIA) | f32 | 🧪 Code complete; needs CUDA toolchain to build |

- `auto` (default) picks the best available and **gracefully falls back** to `wasm-cpu`.
- The registry ([src/backends/registry.js](js_fdtd/src/backends/registry.js)) lazily loads
  each backend and never throws when an optional dependency (the `webgpu` npm package or
  the compiled CUDA `.node`) is missing — it just reports `available: false`.
- The contract every backend reproduces byte‑for‑byte is
  [src/backends/CONTRACT.md](js_fdtd/src/backends/CONTRACT.md).

Backend is chosen entirely from the dashboard's **Compute Engine** selector (POSTed to the
engine per run), so switching engines never requires a restart. The active engine is
streamed back over SSE and shown in the StatsBar "Engine" badge (a `⤳` marks a fallback).

To enable the GPU backends see [BACKENDS.md](js_fdtd/BACKENDS.md) — `npm i webgpu` for
Node/Dawn WebGPU, or `npm run build:cuda` (opt‑in; never part of `npm install`) for CUDA.

---

## 5. The visualization dashboard (`visual/`)

A Next.js (App Router) + React + TypeScript app that consumes the engine's SSE telemetry
and renders it live.

- [app/page.tsx](visual/app/page.tsx) — the dashboard page; [app/api/](visual/app/api/) —
  any client‑side API routes.
- [lib/useSimulation.ts](visual/lib/useSimulation.ts) — the core hook: starts a run
  (`POST :4000/simulate`), subscribes to `:4000/stream` SSE, and exposes live state.
- [lib/backends.ts](visual/lib/backends.ts) — fetches `GET :4000/backends` so the selector
  can grey out unavailable engines.
- [lib/scenarios.ts](visual/lib/scenarios.ts), [lib/fft.ts](visual/lib/fft.ts),
  [lib/demoData.ts](visual/lib/demoData.ts) — scenario list, client‑side FFT, demo data.

Key components in [visual/components/](visual/components/):

- `Dashboard.tsx` — top‑level layout.
- `SimulationControls.tsx`, `BackendSelector.tsx`, `ScenarioSelector.tsx`, `FileUpload.tsx`
  — run controls.
- `StatsBar.tsx` — live step/percent/elapsed + active engine badge.
- `VoltageWaveform.tsx`, `VoltageSpectrum.tsx` — source voltage transient + its spectrum.
- `SParamChart.tsx`, `SmithChart.tsx`, `PolarPattern.tsx` — S₁₁ / impedance / pattern.
- `RadiationPattern3D.tsx`, `AntennaScene.tsx` — Three.js 3D far‑field + antenna geometry.
- `GlassPanel.tsx` — shared UI chrome.

Engine ⇄ dashboard contract:

- `GET :4000/backends` — which engines can actually run (authoritative).
- `POST :4000/simulate` with `{ "backend": "webgpu", "scenario": "..." }` — start a run.
- SSE `status`/`progress` events carry `backend` + `requestedBackend` for the badge.

---

## 6. Quick start

The repo root has a launcher that starts **both** the engine and the dashboard.

```bash
# First time only — install both sub-projects:
npm run install:all

# Start engine (:4000) + dashboard (:3000) together:
npm run dev
```

Then open **http://localhost:3000** and click **Start Engine ⚡**.

### Running the pieces separately

**Engine only:**
```bash
cd js_fdtd
npm install
npm run serve          # HTTP + SSE server on :4000
# or run a single batch job to stdout:
npm start              # node src/index.js (auto backend)
node src/index.js --backend=webgpu --scenario=ifa-fr4-substrate
FDTD_STEPS=60 npm start   # short smoke test
```

**Dashboard only:**
```bash
cd visual
npm install
npm run dev            # Next.js dev server on :3000
```

**Build the WASM kernels** (only if you change the AssemblyScript sources):
```bash
cd js_fdtd
npm run build          # asc assembly/fdtd_kernels.ts -> build/fdtd_kernels.wasm
```

> The launcher `dev.mjs` is zero‑dependency and tears both processes down on Ctrl+C (or if
> either one exits).

---

## 7. Validation & current status

- **CPU reference (`wasm-cpu`)** runs the real IFA geometry (39×136×136) and produces a
  finite, non‑zero voltage trace. ✅
- **WebGPU** is validated on Node/Dawn against the CPU reference (rtol 1e‑3). It required
  four real‑hardware fixes (enum globals, raised storage‑buffer limits, splitting the
  fused CPML kernel to avoid read/read‑write aliasing, renaming the reserved WGSL `pass`
  keyword) plus pinning the Dawn handles so GC can't free native state mid‑run. ✅
- **CUDA** kernels are code‑complete but require an NVIDIA toolchain to build/validate.
  In the environment it was authored in, the build + validation has since been completed
  (see project memory). 🧪
- Far‑field DFT currently falls back to CPU in both GPU backends; current sources /
  inductors / diodes are stubbed since the default problem has none.

Each GPU backend has a `test/validate_*.js` that diffs a 300‑step run against `wasm-cpu`
and **skips cleanly (exit 0)** when the GPU/toolchain is absent.

### Known fragility

`runFDTDCluster` runs its barrier on the same thread that serves HTTP. Polling
`GET /status` ~once a second *during* a run can desync the `Atomics.waitAsync` barrier and
stall the loop — normal SSE (`/stream`) use is unaffected.

---

## 8. Provenance

- The original solver lives in [matlab_codes/](matlab_codes/) and is the source of truth
  for the physics — the JS engine matches its constants and update equations exactly.
- Further reading: [README.md](README.md) (overview/perf), and per‑backend READMEs under
  `js_fdtd/src/backends/webgpu/` and `js_fdtd/src/backends/cuda/`.
