// wasm-cpu backend — the existing multi-threaded WASM + worker engine, wrapped
// behind the backend contract. This is the GOLDEN REFERENCE all other backends
// validate against. It must remain behaviorally identical to calling
// runFDTDCluster() directly.

import os from 'os';
import { runFDTDCluster } from '../fdtdSolver.js';

/** @type {import('./types.js').Backend} */
const backend = {
  name: 'wasm-cpu',

  async isAvailable() {
    // Always available in Node (worker_threads + WebAssembly are built in).
    return typeof WebAssembly !== 'undefined';
  },

  async *run(problem) {
    const { grid, coeffs, cpml, samplers, sources, ff, wasmBuffer, options = {} } = problem;
    const batchSize = options.batchSize ?? 50;
    const gen = runFDTDCluster(grid, coeffs, cpml, samplers, sources, ff, wasmBuffer, batchSize);
    for await (const snap of gen) {
      yield { ...snap, backend: 'wasm-cpu' };
    }
  },

  meta() {
    return { name: 'wasm-cpu', threads: Math.max(1, os.cpus().length - 1), precision: 'f64' };
  },
};

export default backend;
