"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { SimulationData } from "./demoData";

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
  });

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
      const res = await fetch("http://localhost:4000/simulate", { method: "POST" });
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
  }, []);

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
    startSimulation,
    connect,
    checkStatus,
    fetchResults,
  };
}
