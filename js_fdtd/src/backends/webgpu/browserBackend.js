// browserBackend.js — WebGPU backend for the browser (and the Next.js app in
// visual/). Acquires the device from navigator.gpu.requestAdapter().
//
// This file is PURE ESM and imports NO Node-only modules (no os/fs). It wraps the
// SAME engine.js as the Node backend, so the numerics are identical.
// ---------------------------------------------------------------------------

import { WebGPUEngine, deviceDescriptor } from './engine.js';

let _device = null;
let _probe = undefined;

async function acquireDevice() {
  if (_device) return _device;
  const gpu = (typeof navigator !== 'undefined') ? navigator.gpu : undefined;
  if (!gpu || typeof gpu.requestAdapter !== 'function') return null;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const device = await adapter.requestDevice(deviceDescriptor(adapter));
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
    if (!device) throw new Error('webgpu backend: navigator.gpu unavailable');
    const engine = new WebGPUEngine(device);
    await engine.init(problem);
    yield* engine.run(problem);
  },

  meta() {
    return { name: 'webgpu', env: 'browser', precision: 'f32', device: !!_device };
  },
};

export default backend;
