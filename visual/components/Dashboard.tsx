"use client";

import { Suspense, useMemo } from "react";
import dynamic from "next/dynamic";
import GlassPanel from "@/components/GlassPanel";
import StatsBar from "@/components/StatsBar";
import FileUpload from "@/components/FileUpload";
import SimulationControls from "@/components/SimulationControls";
import SParamChart from "@/components/SParamChart";
import VoltageWaveform from "@/components/VoltageWaveform";
import { useSimulation } from "@/lib/useSimulation";
import {
  type SimulationData,
  isDataValid,
  generateFullDemoData,
} from "@/lib/demoData";
import { motion } from "framer-motion";

// Dynamic imports for Three.js (no SSR)
const AntennaScene = dynamic(() => import("@/components/AntennaScene"), {
  ssr: false,
  loading: () => (
    <div className="antenna-canvas-container flex items-center justify-center">
      <div className="text-sm text-gray-500 animate-pulse">Loading 3D Scene...</div>
    </div>
  ),
});

const RadiationPattern3D = dynamic(
  () => import("@/components/RadiationPattern3D"),
  {
    ssr: false,
    loading: () => (
      <div className="radiation-container flex items-center justify-center">
        <div className="text-sm text-gray-500 animate-pulse">Loading Pattern...</div>
      </div>
    ),
  }
);

interface DashboardProps {
  initialData: SimulationData | null;
}

export default function Dashboard({ initialData }: DashboardProps) {
  const sim = useSimulation();

  // Determine which data to display:
  // 1. If simulation completed → use results from SSE
  // 2. If initial data passed from server → use that
  // 3. Fallback → demo data
  const displayData: SimulationData = useMemo(() => {
    if (sim.results && isDataValid(sim.results)) {
      return sim.results;
    }
    if (initialData && isDataValid(initialData)) {
      return initialData;
    }
    return generateFullDemoData();
  }, [sim.results, initialData]);

  const isLiveData = sim.results !== null && isDataValid(sim.results);
  const isDemo = !isLiveData && (!initialData || !isDataValid(initialData));

  // Build voltage data for the waveform chart
  // During simulation: use live streamed voltage history
  // After completion: use full results
  const voltageData = useMemo(() => {
    if (sim.status === "running" && sim.voltageHistory.length > 0) {
      return {
        label: "v1 (live)",
        time_ns: sim.voltageHistory.map((v) => v.time_ns),
        values: sim.voltageHistory.map((v) => v.value),
      };
    }
    if (displayData.timeDomain?.voltages?.length > 0) {
      return displayData.timeDomain.voltages[0];
    }
    return null;
  }, [sim.status, sim.voltageHistory, displayData]);

  const handleFileUpload = (json: unknown) => {
    // For file uploads, just set initial data directly
    // This is a fallback — the primary path is real-time simulation
    const data = json as SimulationData;
    if (isDataValid(data)) {
      // Force re-render with the new data
      window.location.reload();
    }
  };

  return (
    <>
      {/* Background grid */}
      <div className="bg-grid" />

      {/* Hero Header */}
      <header className="hero-header">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="hero-badge">FDTD Simulation Engine</div>
          <h1 className="hero-title">Inverted-F Antenna</h1>
          <p className="hero-subtitle">
            Real-time electromagnetic field simulation &amp; analysis dashboard
          </p>
        </motion.div>

        <div className="header-controls">
          <FileUpload onData={handleFileUpload} />
          {isDemo && sim.status !== "running" && (
            <motion.span
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="demo-badge"
            >
              ⚠ Demo Data — Run simulation or upload results
            </motion.span>
          )}
          {isLiveData && (
            <motion.span
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="live-badge"
            >
              ● Live Simulation Data
            </motion.span>
          )}
        </div>
      </header>

      {/* Simulation Controls */}
      <SimulationControls
        status={sim.status}
        step={sim.step}
        total={sim.total}
        elapsed={sim.elapsed}
        percent={sim.percent}
        error={sim.error}
        onStart={sim.startSimulation}
      />

      {/* Stats Bar */}
      <StatsBar meta={displayData.meta} />

      {/* Dashboard Grid */}
      <main className="dashboard-grid">
        {/* 3D Antenna Scene */}
        <GlassPanel title="3D Antenna Model" icon="📡" delay={0.1}>
          <Suspense
            fallback={
              <div className="antenna-canvas-container flex items-center justify-center">
                <div className="text-sm text-gray-500">Loading...</div>
              </div>
            }
          >
            <AntennaScene />
          </Suspense>
        </GlassPanel>

        {/* S-Parameters */}
        <GlassPanel title="S-Parameters (S11)" icon="📈" delay={0.2}>
          {displayData.sparams?.length > 0 ? (
            <SParamChart data={displayData.sparams[0]} />
          ) : (
            <div className="chart-container flex items-center justify-center h-[280px]">
              <span className="text-sm text-gray-500">
                {sim.status === "running"
                  ? "Awaiting simulation completion..."
                  : "No S-parameter data available"}
              </span>
            </div>
          )}
        </GlassPanel>

        {/* Radiation Pattern */}
        <GlassPanel title="Radiation Pattern" icon="🌐" delay={0.3}>
          {displayData.farfield?.length > 0 ? (
            <Suspense
              fallback={
                <div className="radiation-container flex items-center justify-center">
                  <div className="text-sm text-gray-500">Loading...</div>
                </div>
              }
            >
              <RadiationPattern3D data={displayData.farfield} />
            </Suspense>
          ) : (
            <div className="radiation-container flex items-center justify-center">
              <span className="text-sm text-gray-500">
                {sim.status === "running"
                  ? "Awaiting simulation completion..."
                  : "No farfield data available"}
              </span>
            </div>
          )}
        </GlassPanel>

        {/* Voltage Waveform — LIVE during simulation */}
        <GlassPanel
          title={
            sim.status === "running"
              ? "Time-Domain Voltage (LIVE)"
              : "Time-Domain Voltage"
          }
          icon="⚡"
          delay={0.4}
        >
          {voltageData ? (
            <VoltageWaveform
              data={voltageData}
              isLive={sim.status === "running"}
            />
          ) : (
            <div className="chart-container flex items-center justify-center h-[240px]">
              <span className="text-sm text-gray-500">No voltage data</span>
            </div>
          )}
        </GlassPanel>
      </main>
    </>
  );
}
