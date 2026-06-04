# WebGPU FDTD backend

A GPU compute backend for the FDTD engine. It runs the entire time loop on the
GPU and is validated against the golden `wasm-cpu` reference (see CONTRACT.md).
Precision is **f32** on the GPU (the reference is f64); per the contract this is
physically fine for FDTD and validated within rtol ≈ 1e-3 on the sampled voltage.

## Files

| File | Purpose |
|------|---------|
| `kernels.wgsl.js` | WGSL compute-shader sources: `updateH`, `updateE` (bulk), a generic parameterized `cpml` kernel (serves all 6 faces × both components, magnetic + electric), `injectVoltage`, `sampleVoltage`, `sampleCurrent`. |
| `engine.js` | Environment-agnostic driver. Takes a `GPUDevice`, uploads fields + 18 coeffs + CPML b/a/Psi/CPsi + source/sampler buffers, runs the loop, reads back per-batch scalars and (for far-field) field snapshots. Pure ESM, no Node-only imports. |
| `nodeBackend.js` | Default-exports the Backend object. `isAvailable()` tries the optional `webgpu` npm package; returns `false` (never throws) if absent. |
| `browserBackend.js` | Same Backend shape, acquires the device from `navigator.gpu`. Importable by the Next.js app in `visual/`. Wraps the same `engine.js`. |

## Installing the optional Node dependency

The Node backend needs the optional [`webgpu`](https://www.npmjs.com/package/webgpu)
package (Dawn native bindings):

```bash
npm i webgpu
```

If it is not installed (or no GPU adapter is present), `isAvailable()` returns
`false` and the registry transparently falls back to `wasm-cpu`.

## Browser support

`browserBackend.js` requires a browser with WebGPU enabled:
- Chrome/Edge 113+ (desktop), recent Chrome on Android.
- Safari 18+ / Firefox: behind flags on some versions.

It uses `navigator.gpu.requestAdapter()` → `requestDevice()`. No `os`/`fs` imports,
so it bundles cleanly under Next.js (client component / dynamic import).

## Validation

```bash
# from js_fdtd/
npm i webgpu            # optional, enables the GPU run
node test/validate_webgpu.js
```

The script builds a small problem (~300 steps, single dielectric brick, one
voltage source, CPML on all faces), runs both `wasm-cpu` and `webgpu`, and asserts
the sampled-voltage traces match within `rtol = 1e-3`. If WebGPU is unavailable it
prints `SKIP` and exits 0.

## What is implemented vs. TODO

### Fully implemented (and reproduces the reference index math byte-for-byte)
- **Bulk H update** (`updateH`) — all three components, exact loop bounds.
- **Bulk E update** (`updateE`) — interior-only bounds, boundary planes left to CPML.
- **CPML, all 6 faces, magnetic + electric** — the per-face Psi recurrence
  `Psi = b·Psi + a·(F1−F0)` followed by `Field += CPsi·Psi`, with strides/bases
  derived directly from `src/cpml.js`. Fused into one invocation per (face,
  component) cell (equivalent because each Psi cell maps to a unique field cell).
- **Voltage source injection** (`injectVoltage`) — `Field[field_indices[n]] += Cs[n]·v`
  with `v = voltage_per_e_field[ts]`. Default problem has exactly one voltage source.
- **Sampled voltage** (`sampleVoltage`) — single-workgroup tree reduction of
  `Csvf · Σ Field[field_indices[n]]`, written into a per-step trace buffer, copied
  back into the sampler object at the end (and the latest value per batch for the
  progress snapshot).
- **Per-step update ORDER** matches CONTRACT.md §2 exactly:
  H bulk → H CPML → (H captures) → E bulk → E CPML → source inject →
  (E captures) → far-field DFT.
- **Progress snapshots** yielded per `batchSize` and on the final step, shape per
  CONTRACT.md.

### Implemented but NOT independently validated here
- **Far-field DFT** — computed on the **CPU** via the existing
  `accumulateFarfieldDFT` by mapping the 6 field buffers back **every step** when
  `ff.nFreq > 0`. Numerically identical to the reference (same f64 routine) but the
  per-step readback is slow. **TODO:** move the DFT accumulation on-GPU to avoid the
  stall. The default problem has 2 frequencies; the validation script disables
  far-field (empty `frequencies`) to keep it fast, so the far-field path itself is
  not exercised by `validate_webgpu.js`.
- **Sampled current** (`sampleCurrent`) — a generic weighted H-sum kernel is
  provided, but the default problem has no current observers wired into the GPU
  loop, and the loop-integral term lists are **not yet generated** in `engine.js`.
  **TODO:** precompute the `(flatIndex, weight)` term lists from
  `captureSampledCurrents` and dispatch per observer. Magnetic-field captures
  (`captureMagneticFields`) are likewise not wired (no H observers by default).

### Stubbed / TODO (clearly out of scope for the default problem)
- **Current sources, inductors, diodes** — NOT implemented on GPU. The default
  problem has none. If present they must be added (diodes need the Newton-Raphson
  solver from `src/sources.js`).

## Assumptions
- WebGPU guarantees memory visibility between successive `dispatchWorkgroups`
  calls within a single compute pass for storage buffers (used to sequence bulk →
  CPML in one pass). This is per the WebGPU spec's implicit per-dispatch barrier.
- `field_indices` (Int32Array, non-negative flat offsets) fit in `u32`.
- The CPML region width `nc` is less than half each dimension, so opposite faces
  (e.g. xn/xp) write disjoint cells and may share a pass without conflict.
- The `webgpu` npm package exposes either a `create([])` factory or a
  `navigator.gpu`; `nodeBackend.js` handles both shapes.
