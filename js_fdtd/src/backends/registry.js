// Backend registry — the single entry point for selecting an FDTD compute engine.
//
// Every backend implements the contract in CONTRACT.md:
//   { name, isAvailable(): Promise<boolean>, async *run(problem) }
//
// Backends are registered lazily (dynamic import) so that a missing optional
// dependency (e.g. the `webgpu` npm package, or a CUDA native addon that did
// not compile) never crashes the process at startup — it just reports
// isAvailable() === false and is skipped by the selector.

/** @typedef {import('./types.js').Problem} Problem */

const LOADERS = {
  'wasm-cpu': () => import('./wasmCpu.js').then(m => m.default),
  'webgpu':   () => import('./webgpu/nodeBackend.js').then(m => m.default),
  'cuda':     () => import('./cuda/cudaBackend.js').then(m => m.default),
};

const _cache = new Map();

/**
 * Load a backend module by name. Returns the backend object, or null if the
 * module failed to load (e.g. optional native dep absent).
 * @param {string} name
 */
export async function loadBackend(name) {
  if (_cache.has(name)) return _cache.get(name);
  const loader = LOADERS[name];
  if (!loader) throw new Error(`Unknown backend "${name}". Known: ${Object.keys(LOADERS).join(', ')}`);
  let backend = null;
  try {
    backend = await loader();
  } catch (err) {
    console.warn(`[backends] backend "${name}" failed to load: ${err.message}`);
    backend = null;
  }
  _cache.set(name, backend);
  return backend;
}

/**
 * Resolve a usable backend, honoring an explicit preference and falling back
 * through a priority list. Returns { name, backend }.
 *
 * @param {string} [preferred] - e.g. 'webgpu' | 'cuda' | 'wasm-cpu' | 'auto'
 */
export async function selectBackend(preferred = 'auto') {
  const order = preferred && preferred !== 'auto'
    ? [preferred, 'webgpu', 'cuda', 'wasm-cpu']
    : ['webgpu', 'cuda', 'wasm-cpu'];

  const seen = new Set();
  for (const name of order) {
    if (seen.has(name)) continue;
    seen.add(name);
    const backend = await loadBackend(name);
    if (!backend) continue;
    let ok = false;
    try { ok = await backend.isAvailable(); } catch { ok = false; }
    if (ok) return { name, backend };
    if (preferred === name && name !== 'wasm-cpu') {
      console.warn(`[backends] requested "${name}" is unavailable; falling back.`);
    }
  }
  throw new Error('No FDTD backend is available (not even wasm-cpu).');
}

/**
 * Probe every backend for availability — used by the UI selector to show which
 * options the current machine supports.
 * @returns {Promise<Array<{name:string, available:boolean, detail?:string}>>}
 */
export async function listBackends() {
  const out = [];
  for (const name of Object.keys(LOADERS)) {
    const backend = await loadBackend(name);
    if (!backend) { out.push({ name, available: false, detail: 'module failed to load' }); continue; }
    let available = false, detail;
    try { available = await backend.isAvailable(); }
    catch (e) { detail = e.message; }
    out.push({ name, available, detail });
  }
  return out;
}
