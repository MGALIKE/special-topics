import { NextResponse } from "next/server";

// ───────────────────────────────────────────────────────────────────────────
// Server-side backend availability probe.
//
// In principle this should call `listBackends()` from the FDTD registry at
// `../../../js_fdtd/src/backends/registry.js`. However that registry eagerly
// (via dynamic import) pulls in `wasmCpu.js`, which imports Node's `os` module
// and the multi-threaded `fdtdSolver.js` (worker_threads + a compiled .wasm).
// Bundling that cross-package, Node-only graph into a Next Route Handler is
// brittle (the bundler tries to trace `worker_threads`, wasm assets, etc.) and
// is exactly the "awkward cross-package import" the task says to avoid.
//
// So we reproduce the SAME availability checks the registry performs, inline:
//   - wasm-cpu : available whenever WebAssembly exists (always true in Node).
//   - cuda     : available only if the native CUDA addon can be resolved.
//                The registry loads `./cuda/cudaBackend.js`; if that module or
//                its native addon is absent, `loadBackend` catches the import
//                error and reports available:false. We mirror that by trying to
//                resolve the addon and degrading gracefully.
//
// Browser-only backends (webgpu) are intentionally NOT probed here — WebGPU
// availability depends on the user's browser/GPU and is detected client-side in
// BackendSelector.tsx. We still list it with location:'browser' for completeness.
// ───────────────────────────────────────────────────────────────────────────

export const dynamic = "force-dynamic";

type BackendInfo = {
  name: string;
  available: boolean;
  detail?: string;
  location: "server" | "browser";
};

function probeCuda(): { available: boolean; detail?: string } {
  // Mirror registry behaviour: cuda is available only if the native addon /
  // backend module resolves. None ships in this repo yet, so this normally
  // reports unavailable — which is the honest answer.
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const req = eval("require") as NodeRequire;
    // Common candidate locations for a compiled CUDA addon.
    const candidates = [
      "fdtd-cuda",
      "../../../js_fdtd/src/backends/cuda/cudaBackend.js",
    ];
    for (const c of candidates) {
      try {
        req.resolve(c);
        return { available: true, detail: "native CUDA addon resolved" };
      } catch {
        // try next candidate
      }
    }
    return { available: false, detail: "native CUDA addon not installed" };
  } catch {
    return { available: false, detail: "cannot probe (no CommonJS require)" };
  }
}

export async function GET() {
  const wasmAvailable = typeof WebAssembly !== "undefined";
  const cuda = probeCuda();

  const backends: BackendInfo[] = [
    {
      name: "wasm-cpu",
      available: wasmAvailable,
      detail: wasmAvailable
        ? "multi-threaded WASM engine (golden reference)"
        : "WebAssembly unavailable in this runtime",
      location: "server",
    },
    {
      name: "cuda",
      available: cuda.available,
      detail: cuda.detail,
      location: "server",
    },
    {
      name: "webgpu",
      available: false, // real availability is decided client-side
      detail: "detected in-browser via navigator.gpu",
      location: "browser",
    },
  ];

  return NextResponse.json({ backends });
}
