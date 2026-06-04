"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { SimulationData } from "./demoData";
import type { BackendChoice } from "./backends";
import type { ScenarioInfo, ScenariosResponse } from "./scenarios";

const BACKEND_STORAGE_KEY = "fdtd.backendChoice";
const SCENARIO_STORAGE_KEY = "fdtd.scenario";
const DEFAULT_SCENARIO = "ifa-dualband-baseline";

export type SimStatus =
  | "disconnected"
  | "idle"
  | "initializing"
  | "running"
  | "postprocessing"
  | "done"
  | "error";

export interface VoltagePoint {
  time_ns: number;
  value: number;
}

export interface SimulationState {
  status: SimStatus;
  step: number;
  total: number;
  elapsed: number;
  percent: string;
  error: string | null;
  voltageHistory: VoltagePoint[];
  results: SimulationData | null;
  // The engine that actually ran (resolved by the server registry — may differ
  // from the requested choice if it fell back, e.g. webgpu -> wasm-cpu).
  activeBackend: string | null;
}

export function useSimulation() {
  const [state, setState] = useState<SimulationState>({
    status: "disconnected",
    step: 0,
    total: 0,
    elapsed: 0,
    percent: "0.0",
    error: null,
    voltageHistory: [],
    results: null,
    activeBackend: null,
  });

  // Chosen compute backend. This is a real, persisted user preference. Today it
  // does NOT change what actually executes (the server runs the WASM engine and
  // the viewer renders its results); it is plumbed through so the UI reflects an
  // honest choice and is ready to drive a live run once a browser backend lands.
  // Default = "auto".
  const [backendChoice, setBackendChoiceState] = useState<BackendChoice>("auto");

  // Hydrate the saved choice on mount (avoids SSR/localStorage mismatch).
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(BACKEND_STORAGE_KEY);
      if (saved === "auto" || saved === "wasm-cpu" || saved === "webgpu" || saved === "cuda") {
        setBackendChoiceState(saved);
      }
    } catch {
      // localStorage unavailable — keep the default.
    }
  }, []);

  const setBackendChoice = useCallback((choice: BackendChoice) => {
    setBackendChoiceState(choice);
    try {
      window.localStorage.setItem(BACKEND_STORAGE_KEY, choice);
    } catch {
      // Non-fatal: persistence is best-effort.
    }
  }, []);

  // Available scenarios (problem definitions) and the chosen one. The list comes
  // from the engine server's GET /scenarios; the selection is persisted and sent
  // to POST /simulate so the engine loads the matching JSON.
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [scenario, setScenarioState] = useState<string>(DEFAULT_SCENARIO);

  const setScenario = useCallback((name: string) => {
    setScenarioState(name);
    try {
      window.localStorage.setItem(SCENARIO_STORAGE_KEY, name);
    } catch {
      // Non-fatal: persistence is best-effort.
    }
  }, []);

  // Load the scenario list (and hydrate the saved selection) on mount.
  useEffect(() => {
    let cancelled = false;
    let saved: string | null = null;
    try {
      saved = window.localStorage.getItem(SCENARIO_STORAGE_KEY);
    } catch {
      // localStorage unavailable — fall back to the server default.
    }

    (async () => {
      try {
        const res = await fetch("http://localhost:4000/scenarios");
        if (!res.ok) return;
        const data = (await res.json()) as ScenariosResponse;
        if (cancelled) return;
        setScenarios(data.scenarios);
        const names = data.scenarios.map((s) => s.name);
        // Prefer the saved pick if it still exists, else the server default.
        if (saved && names.includes(saved)) {
          setScenarioState(saved);
        } else if (data.default && names.includes(data.default)) {
          setScenarioState(data.default);
        } else if (names.length > 0) {
          setScenarioState(names[0]);
        }
      } catch {
        // Engine server unreachable — keep the default; selector stays empty.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const eventSourceRef = useRef<EventSource | null>(null);
  const voltageBufferRef = useRef<VoltagePoint[]>([]);

  // Connect to SSE stream
  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource("http://localhost:4000/stream");
    eventSourceRef.current = es;

    es.addEventListener("status", (e) => {
      const data = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        status: data.status,
        step: data.step ?? prev.step,
        total: data.total ?? prev.total,
        elapsed: data.elapsed ?? prev.elapsed,
        percent: data.percent ?? prev.percent,
        activeBackend: data.backend ?? prev.activeBackend,
      }));
    });

    es.addEventListener("progress", (e) => {
      const data = JSON.parse(e.data);
      // Buffer voltage point
      if (data.voltage !== null && data.voltage !== undefined) {
        voltageBufferRef.current.push({
          time_ns: data.time_ns,
          value: data.voltage,
        });
      }
      setState((prev) => ({
        ...prev,
        status: "running",
        step: data.step,
        total: data.total,
        elapsed: data.elapsed,
        percent: data.percent,
        voltageHistory: [...voltageBufferRef.current],
        activeBackend: data.backend ?? prev.activeBackend,
      }));
    });

    es.addEventListener("voltage_history", (e) => {
      const history: VoltagePoint[] = JSON.parse(e.data);
      voltageBufferRef.current = [...history];
      setState((prev) => ({
        ...prev,
        voltageHistory: [...history],
      }));
    });

    es.addEventListener("complete", () => {
      // The massive JSON payload is no longer sent over SSE directly.
      // We manually fetch from the REST endpoint instead to avoid buffer truncation!
      fetch("/api/fdtd/results")
        .then((res) => {
          if (res.ok) return res.json();
          throw new Error("fetching results failed");
        })
        .then((data: SimulationData) => {
          setState((prev) => ({
            ...prev,
            status: "done",
            results: data,
          }));
        });
    });

    es.addEventListener("error", (e) => {
      try {
        const data = JSON.parse((e as MessageEvent).data);
        setState((prev) => ({
          ...prev,
          status: "error",
          error: data.message,
        }));
      } catch {
        // SSE connection error
        setState((prev) => ({
          ...prev,
          status: "disconnected",
        }));
      }
    });

    es.onerror = () => {
      // Will auto-reconnect
      setState((prev) => ({
        ...prev,
        status: prev.status === "running" ? prev.status : "disconnected",
      }));
    };

    es.onopen = () => {
      // Connection established — status event will set the actual state
    };
  }, []);

  // Start simulation
  const startSimulation = useCallback(async () => {
    voltageBufferRef.current = [];
    setState((prev) => ({
      ...prev,
      voltageHistory: [],
      results: null,
      error: null,
      step: 0,
      percent: "0.0",
    }));

    try {
      const res = await fetch("http://localhost:4000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Tell the engine server which compute backend AND which scenario
        // (problem definition) to use for this run.
        body: JSON.stringify({ backend: backendChoice, scenario }),
      });
      if (!res.ok) {
        const data = await res.json();
        setState((prev) => ({
          ...prev,
          status: "error",
          error: data.error || "Failed to start simulation",
        }));
      }
    } catch (err) {
      setState((prev) => ({
        ...prev,
        status: "error",
        error: "Cannot reach backend server",
      }));
    }
  }, [backendChoice, scenario]);

  // Check backend status
  const checkStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/fdtd/status");
      if (res.ok) {
        const data = await res.json();
        setState((prev) => ({ ...prev, ...data }));
        return data;
      }
    } catch {
      // Backend not available
    }
    return null;
  }, []);

  // Fetch completed results
  const fetchResults = useCallback(async () => {
    try {
      const res = await fetch("http://localhost:4000/results");
      if (res.ok) {
        const data: SimulationData = await res.json();
        setState((prev) => ({ ...prev, results: data, status: "done" }));
        return data;
      }
    } catch {
      // No results
    }
    return null;
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    checkStatus().then((s) => {
      if (s?.hasResults) fetchResults();
    });
    return () => {
      eventSourceRef.current?.close();
    };
  }, [connect, checkStatus, fetchResults]);

  return {
    ...state,
    backendChoice,
    setBackendChoice,
    scenarios,
    scenario,
    setScenario,
    startSimulation,
    connect,
    checkStatus,
    fetchResults,
  };
}
