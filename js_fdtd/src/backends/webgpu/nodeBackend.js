// nodeBackend.js — WebGPU backend for Node.js.
// ---------------------------------------------------------------------------
// Acquires a GPUDevice via the OPTIONAL `webgpu` npm package (Dawn bindings).
// If the package or a suitable adapter is absent, isAvailable() returns false
// (never throws) so the registry falls back to wasm-cpu cleanly.
//
//   npm i webgpu        # optional dependency, installs Dawn native bindings
//
// The registry imports THIS file as the 'webgpu' backend
// (see src/backends/registry.js).
// ---------------------------------------------------------------------------

import { WebGPUEngine, deviceDescriptor } from './engine.js';

let _device = null;       // cached GPUDevice
let _probe = undefined;   // cached availability probe (Promise result)

// IMPORTANT: keep the GPU + adapter (and the `webgpu` module / navigator object)
// alive for the whole process. With the Dawn bindings these wrap native handles;
// if they are only locals they get garbage-collected after acquireDevice()
// returns, freeing native state the still-running GPUDevice depends on — which
// manifests as a hard SIGSEGV partway through a run (exit 139), not a JS error.
// Retaining them in module scope keeps the native objects pinned.
let _mod = null;          // the imported `webgpu` module
let _gpu = null;          // the GPU / navigator.gpu object
let _adapter = null;      // the GPUAdapter

async function acquireDevice() {
  if (_device) return _device;
  // Dynamic import so a missing package does not crash module load.
  try {
    _mod = await import('webgpu');
  } catch {
    return null; // package not installed
  }
  try {
    // The `webgpu` package exposes either a `create([])` factory returning a
    // GPU, or a ready `navigator.gpu`. Support both shapes.
    const entry = _mod.default ?? _mod;

    // Dawn does NOT auto-install the WebGPU enum globals that browsers expose
    // (GPUBufferUsage, GPUMapMode, GPUShaderStage, …). The shared engine.js uses
    // them unqualified, so copy them onto globalThis here. Without this the run
    // throws `GPUBufferUsage is not defined`.
    if (entry.globals && typeof entry.globals === 'object') {
      for (const [k, v] of Object.entries(entry.globals)) {
        if (globalThis[k] === undefined) globalThis[k] = v;
      }
    }

    let gpu = null;
    if (typeof entry.create === 'function') {
      const nav = entry.create([]); // returns an object with .requestAdapter or a navigator
      gpu = nav.gpu ?? nav;
    } else if (entry.gpu) {
      gpu = entry.gpu;
    } else if (globalThis.navigator && globalThis.navigator.gpu) {
      gpu = globalThis.navigator.gpu;
    }
    if (!gpu || typeof gpu.requestAdapter !== 'function') return null;
    _gpu = gpu; // pin (see note above)

    _adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!_adapter) return null;
    const device = await _adapter.requestDevice(deviceDescriptor(_adapter));
    if (!device) return null;
    _device = device;
    return device;
  } catch {
    return null;
  }
}

/** @type {import('../types.js').Backend} */
const backend = {
  name: 'webgpu',

  async isAvailable() {
    if (_probe !== undefined) return _probe;
    const dev = await acquireDevice();
    _probe = !!dev;
    return _probe;
  },

  async *run(problem) {
    const device = await acquireDevice();
    if (!device) throw new Error('webgpu backend: no GPUDevice available');
    const engine = new WebGPUEngine(device);
    await engine.init(problem);
    yield* engine.run(problem);
  },

  meta() {
    return { name: 'webgpu', env: 'node', precision: 'f32', device: !!_device };
  },
};

export default backend;
