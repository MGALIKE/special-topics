# FDTD Compute Backends

The solver's time loop runs behind a pluggable **backend** interface, so the
same physics can execute on the CPU, a browser/Node GPU (WebGPU), or an NVIDIA
GPU (CUDA). All backends obey one contract and are validated against the CPU
engine as the golden reference.

## Backends

| Name       | Where it runs            | Precision | Status |
|------------|--------------------------|-----------|--------|
| `wasm-cpu` | Node, worker_threads + WASM | f64    | ✅ Reference. Always available. |
| `webgpu`   | Browser **and** Node (Dawn) | f32    | ✅ Validated on Node/Dawn (matches wasm-cpu, rtol 1e-3). |
| `cuda`     | Node native addon (NVIDIA)  | f32    | 🧪 Code complete; needs CUDA toolchain to build. |

`auto` (default) picks the best available, falling back to `wasm-cpu`.

## How selection works

- `src/backends/registry.js` — `selectBackend(pref)`, `listBackends()`. Loads
  each backend lazily and **never throws** if an optional dependency (the
  `webgpu` npm package, or the compiled CUDA `.node`) is missing — that backend
  just reports `available: false` and is skipped.
- `src/backends/CONTRACT.md` — the authoritative spec (memory layout, index
  math, exact per-step order). Every backend reproduces it byte-for-byte.
- `src/backends/wasmCpu.js` — wraps the existing `runFDTDCluster` engine.

## Running

```bash
# Auto-select (prints which backends are available):
npm start

# Force a backend (falls back with a warning if unavailable):
node src/index.js --backend=webgpu
node src/index.js --backend=cuda
node src/index.js --backend=wasm-cpu

# Or via env:
FDTD_BACKEND=webgpu npm start

# Short run for smoke-testing (overrides the 7000-step default):
FDTD_STEPS=60 npm start
```

Example startup line:
```
Available backends: wasm-cpu, webgpu(x), cuda(x)
Running FDTD on backend: wasm-cpu
```
`(x)` = detected-but-unavailable on this machine.

## Enabling the GPU backends

**WebGPU (Node):**
```bash
npm i webgpu           # optional Dawn bindings (installs a prebuilt native module)
node test/validate_webgpu.js   # diffs vs wasm-cpu on a 300-step grid (rtol 1e-3)
```
Verified working on Node/Dawn. Four real-hardware fixes were needed beyond the
package install (all in `src/backends/webgpu/`), each of which also hardens the
in-browser path since they share `engine.js` + `kernels.wgsl.js`:
1. **Enum globals** — Dawn does not expose `GPUBufferUsage`/`GPUMapMode`/… as
   globals the way browsers do; `nodeBackend.js` installs them from the package.
2. **Device limits** — the bulk H/E shaders bind 15 storage buffers, over the
   default cap of 8; the device is now requested with a raised
   `maxStorageBuffersPerShaderStage` (`deviceDescriptor()` in `engine.js`).
3. **CPML aliasing** — the fused CPML kernel bound the same buffer as both
   read-write *and* read in one dispatch (legal on CPU, forbidden by WebGPU). It
   is now two passes (update Psi → correct Field), which also matches the
   reference's two-loop structure exactly.
4. **WGSL `pass` keyword** — `pass` is reserved in current WGSL; the struct field
   was renamed to `passIdx`.
   And: the Dawn GPU/adapter handles are pinned in module scope so GC cannot free
   native state mid-run (that surfaced as a hard SIGSEGV, exit 139).
WebGPU in the browser needs no install — see `visual/` and the in-app backend
selector. The browser entry is `src/backends/webgpu/browserBackend.js`.

**CUDA (Node native addon):**
```bash
# Prereqs: NVIDIA driver, CUDA Toolkit (nvcc on PATH), VS Build Tools (Win) / GCC (Linux)
npm install --no-save node-gyp node-addon-api
npm run build:cuda     # -> build/Release/fdtd_cuda.node
node test/validate_cuda.js
```
The CUDA build is **opt-in** — it is never part of `npm install`, so machines
without CUDA are unaffected.

## Validation protocol

Each GPU backend is considered correct only once its `test/validate_*.js`
passes: it runs a small 300-step problem on both `wasm-cpu` and the candidate
and asserts the sampled-voltage trace matches within `rtol = 1e-3` (f32). The
scripts **SKIP cleanly (exit 0)** when the GPU/toolchain is absent.

## Current honest status (verified in this environment, no GPU present)

- ✅ Backend abstraction, registry, lazy load + graceful fallback — **verified
  running** (`selectBackend('webgpu')` falls back to `wasm-cpu` with a warning).
- ✅ `wasm-cpu` reference runs through the new path on the real antenna geometry
  (39×136×136) and produces a finite, non-zero voltage trace.
- ✅ Fixed a pre-existing bug in `index.js`: `applyLumpedElementCoefficients`
  was called with `(coeffs, grid, dt, …)` instead of `(coeffs, matComps, grid, …)`,
  which had been silently corrupting source-injection coefficients.
- 🧪 `webgpu` / `cuda` kernels are complete and internally consistent but were
  **not executable here** (no GPU/driver/toolkit). They require a validation run
  on real hardware. Each backend's README lists exactly what is implemented vs.
  TODO (notably: far-field DFT currently falls back to CPU in both GPU backends;
  current sources / inductors / diodes are stubbed since the default problem has
  none).

## One-command launch (engine + dashboard)

You no longer start the engine and frontend separately per backend. From the
repo root:

```bash
npm run dev      # starts js_fdtd engine (:4000) AND visual dashboard (:3000)
```

(First time only: `npm run install:all` to install both sub-projects.)

The compute engine — `auto` / `wasm-cpu` / `webgpu` / `cuda` — is chosen purely
in the dashboard's **Compute Engine** selector; the choice is POSTed to the
engine server per run, so switching engines never requires restarting anything.
The launcher (`dev.mjs`) is zero-dependency and tears both processes down on
Ctrl+C (or if either one exits).

## Frontend engine selector (connected)

The `visual/` dashboard's **Compute Engine** selector is wired to the engine
server end-to-end:

- `GET http://localhost:4000/backends` — the server reports which engines it can
  actually run (authoritative; the selector greys out unavailable ones).
- `POST http://localhost:4000/simulate` with body `{ "backend": "webgpu" }` —
  the server resolves it via `selectBackend()` and runs that engine, falling
  back to `wasm-cpu` if unavailable.
- The active engine is streamed back over SSE (`status`/`progress` events carry
  `backend` + `requestedBackend`) and shown in the StatsBar "Engine" badge; a
  `⤳` marks a fallback (e.g. requested `webgpu`, ran `wasm-cpu`).

So picking an engine in the UI changes what actually executes the next run.
(A live *in-browser* WebGPU run — executing the time loop client-side instead of
on the server — is the remaining future piece; the browser entry exists at
`src/backends/webgpu/browserBackend.js`.)

> Known fragility (pre-existing, not selection-related): `runFDTDCluster` runs
> its barrier on the same main thread that serves HTTP. Hammering `GET /status`
> ~once a second *during* a run can desync the `Atomics.waitAsync` barrier and
> stall the loop. Normal use (SSE `/stream`) is unaffected. This is the same
> synchronization scheme flagged in the performance review.

## Per-backend details

See `src/backends/webgpu/README.md` and `src/backends/cuda/README.md`.
