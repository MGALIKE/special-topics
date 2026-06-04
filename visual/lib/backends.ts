// Shared types + labels for the compute-backend selector.
//
// A "backend" is the compute engine that runs the FDTD time loop. The four
// names mirror the FDTD registry (see js_fdtd/src/backends/CONTRACT.md):
//   - wasm-cpu : multi-threaded WASM engine (server, golden reference)
//   - webgpu   : in-browser GPU via WebGPU (client)
//   - cuda     : native CUDA addon (server)
//   - auto     : let the registry pick the best available engine
//
// This file is shared by BackendSelector, useSimulation and StatsBar so they
// all agree on the identifiers and display labels.

export type BackendName = "wasm-cpu" | "webgpu" | "cuda";
export type BackendChoice = "auto" | BackendName;

export interface BackendOption {
  name: BackendName;
  /** Whether this engine is usable right now (server probe or client probe). */
  available: boolean;
  /** Human-readable reason / extra info, surfaced as a tooltip. */
  detail?: string;
  /** Where the engine runs. */
  location: "server" | "browser";
}

/** Shape returned by GET /api/backends. */
export interface BackendsResponse {
  backends: BackendOption[];
}

export const BACKEND_LABELS: Record<BackendChoice, string> = {
  auto: "Auto",
  "wasm-cpu": "WASM (CPU)",
  webgpu: "WebGPU",
  cuda: "CUDA",
};

/** Short label used in compact spots like the StatsBar badge. */
export const BACKEND_SHORT_LABELS: Record<BackendChoice, string> = {
  auto: "Auto",
  "wasm-cpu": "WASM",
  webgpu: "WebGPU",
  cuda: "CUDA",
};

/** Rich, marketing-grade copy for the compute-engine card deck. */
export interface BackendMeta {
  /** One-line value proposition. */
  tagline: string;
  /** Two–three sentence description of what the engine actually does. */
  description: string;
  /** Compact technology stack badge. */
  tech: string;
}

export const BACKEND_META: Record<BackendChoice, BackendMeta> = {
  auto: {
    tagline: "Zero-config",
    description:
      "Probes the machine on launch and dispatches the run to the fastest engine that is actually present — CUDA first, then WebGPU, then the WASM reference. You never pick wrong.",
    tech: "Smart routing",
  },
  "wasm-cpu": {
    tagline: "Golden reference",
    description:
      "The full FDTD kernel compiled to WebAssembly with SIMD, fanned out across worker threads on the CPU. This is the numerically-verified baseline every other engine is validated against.",
    tech: "WASM · SIMD · Multi-thread",
  },
  webgpu: {
    tagline: "GPU, no lock-in",
    description:
      "The same Yee-grid update rewritten as WGSL compute shaders. Runs on the server through Node/Dawn today and is portable to any WebGPU GPU — no vendor-specific toolchain required.",
    tech: "WGSL · Compute shaders",
  },
  cuda: {
    tagline: "Maximum throughput",
    description:
      "A native CUDA addon that maps the electric- and magnetic-field updates straight onto NVIDIA cores. Bit-for-bit matched to the reference, with the lowest time-per-step of the four engines.",
    tech: "CUDA · Native addon",
  },
};
