"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  type BackendChoice,
  type BackendName,
  type BackendOption,
  type BackendsResponse,
  BACKEND_LABELS,
  BACKEND_META,
} from "@/lib/backends";

interface BackendSelectorProps {
  value: BackendChoice;
  onChange: (choice: BackendChoice) => void;
}

const ORDER: BackendChoice[] = ["auto", "wasm-cpu", "webgpu", "cuda"];

/**
 * Client-side WebGPU feature detection. We need both the API surface
 * (navigator.gpu) AND a real adapter — a browser can expose navigator.gpu yet
 * fail to hand out an adapter (no compatible GPU / disabled in flags).
 */
async function probeWebGPU(): Promise<BackendOption> {
  const base: BackendOption = {
    name: "webgpu",
    available: false,
    location: "browser",
  };
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    return { ...base, detail: "navigator.gpu not supported in this browser" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      return { ...base, available: true, detail: "WebGPU adapter available" };
    }
    return { ...base, detail: "no WebGPU adapter (GPU unavailable/disabled)" };
  } catch (err) {
    return {
      ...base,
      detail: `WebGPU probe failed: ${(err as Error).message}`,
    };
  }
}

export default function BackendSelector({ value, onChange }: BackendSelectorProps) {
  // Availability per backend name. webgpu is filled by the client probe; the
  // server backends (wasm-cpu, cuda) come from GET /api/backends.
  const [options, setOptions] = useState<Record<BackendName, BackendOption>>({
    "wasm-cpu": { name: "wasm-cpu", available: false, location: "server", detail: "probing…" },
    webgpu: { name: "webgpu", available: false, location: "browser", detail: "probing…" },
    cuda: { name: "cuda", available: false, location: "server", detail: "probing…" },
  });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      // Browser WebGPU capability (independent of the server).
      const clientWebgpu = await probeWebGPU();

      // Authoritative availability = the engine server that actually runs sims
      // (localhost:4000/backends). Fall back to the Next route probe if the
      // engine server is unreachable.
      let serverList: BackendOption[] | null = null;
      try {
        const res = await fetch("http://localhost:4000/backends");
        if (res.ok) serverList = ((await res.json()) as BackendsResponse).backends;
      } catch {
        /* engine server down — try the Next route */
      }
      if (!serverList) {
        try {
          const res = await fetch("/api/backends");
          if (res.ok) serverList = ((await res.json()) as BackendsResponse).backends;
        } catch {
          /* ignore */
        }
      }
      if (cancelled) return;

      const byName: Partial<Record<BackendName, BackendOption>> = {};
      for (const b of serverList ?? []) byName[b.name] = b;
      const serverWebgpu = byName.webgpu;

      setOptions({
        "wasm-cpu":
          byName["wasm-cpu"] ?? {
            name: "wasm-cpu",
            available: false,
            location: "server",
            detail: "engine server unreachable",
          },
        cuda:
          byName.cuda ?? {
            name: "cuda",
            available: false,
            location: "server",
            detail: "engine server unreachable",
          },
        webgpu: {
          name: "webgpu",
          // The run executes on the server today; enable if the runner has
          // Node/Dawn WebGPU OR the browser exposes WebGPU (selecting it falls
          // back gracefully on the server side).
          available: !!serverWebgpu?.available || clientWebgpu.available,
          location: "server",
          detail: serverWebgpu?.available
            ? "server WebGPU (Node/Dawn) ready"
            : clientWebgpu.available
              ? "browser WebGPU detected — server run needs `npm i webgpu` or falls back"
              : serverWebgpu?.detail ?? clientWebgpu.detail,
        },
      });
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const availability = useMemo(() => {
    // "auto" is always selectable; it falls back through the registry's order.
    const map: Record<BackendChoice, BackendOption | null> = {
      auto: null,
      "wasm-cpu": options["wasm-cpu"],
      webgpu: options.webgpu,
      cuda: options.cuda,
    };
    return map;
  }, [options]);

  return (
    <div className="engine-block">
      <div className="engine-head">
        <div className="engine-head-title">
          <span className="engine-head-index">00</span>
          <span>Compute Engine</span>
        </div>
        <span className="engine-head-note">
          One FDTD kernel · three real backends · routed to the fastest hardware
          present
        </span>
      </div>

      <div
        className="engine-grid"
        role="radiogroup"
        aria-label="Compute backend"
      >
        {ORDER.map((choice, i) => {
          const opt = availability[choice];
          const isAuto = choice === "auto";
          const available = isAuto ? true : !!opt?.available;
          const active = value === choice;
          // Auto is always pickable. Concrete engines are disabled when their
          // probe says unavailable, so the user can't pick something that
          // cannot run.
          const disabled = !isAuto && !available;
          const meta = BACKEND_META[choice];
          const statusText = isAuto
            ? "Ready"
            : available
              ? "Available"
              : "Not detected";

          return (
            <motion.button
              key={choice}
              type="button"
              role="radio"
              aria-checked={active}
              aria-label={`${BACKEND_LABELS[choice]} — ${available ? "available" : "unavailable"}`}
              title={opt?.detail ?? meta.description}
              disabled={disabled}
              onClick={() => !disabled && onChange(choice)}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.05 * i }}
              className={`engine-card ${active ? "engine-card--active" : ""} ${
                disabled ? "engine-card--disabled" : ""
              }`}
            >
              <div className="engine-card-top">
                <div>
                  <div className="engine-card-name">
                    {BACKEND_LABELS[choice]}
                  </div>
                  <div className="engine-card-tagline">{meta.tagline}</div>
                </div>
                <span className="engine-card-status">
                  <span
                    className={`engine-card-status-dot ${
                      available ? "engine-card-status-dot--on" : ""
                    }`}
                  />
                  {statusText}
                </span>
              </div>

              <p className="engine-card-desc">{meta.description}</p>

              <div className="engine-card-tech">{meta.tech}</div>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
