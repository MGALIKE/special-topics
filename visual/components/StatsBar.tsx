"use client";

import { motion } from "framer-motion";
import { type BackendChoice, BACKEND_SHORT_LABELS } from "@/lib/backends";

interface StatsBarProps {
  meta: {
    numberOfTimeSteps: number;
    dt: number;
    generated: string;
    frequencies_Hz?: number[];
  };
  /** Currently selected compute engine (the active user choice). */
  backendChoice?: BackendChoice;
  /** The engine that actually ran the last/current sim (server-resolved). */
  activeBackend?: string | null;
}

function StatItem({
  label,
  value,
  unit,
  delay,
}: {
  label: string;
  value: string;
  unit?: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="stat-item"
    >
      <span className="stat-label">{label}</span>
      <span className="stat-value" suppressHydrationWarning>
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </span>
    </motion.div>
  );
}

export default function StatsBar({ meta, backendChoice = "auto", activeBackend = null }: StatsBarProps) {
  // Prefer the engine that actually ran; fall back to the user's choice label.
  const engineLabel = activeBackend
    ? (BACKEND_SHORT_LABELS[activeBackend as BackendChoice] ?? activeBackend)
    : BACKEND_SHORT_LABELS[backendChoice];
  // If the resolved engine differs from a concrete request, hint at the fallback.
  const fellBack =
    !!activeBackend && backendChoice !== "auto" && activeBackend !== backendChoice;
  const freqRange = meta.frequencies_Hz
    ? `${(meta.frequencies_Hz[0] / 1e9).toFixed(2)} – ${(
        meta.frequencies_Hz[meta.frequencies_Hz.length - 1] / 1e9
      ).toFixed(1)}`
    : "—";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8, delay: 0.2 }}
      className="stats-bar"
    >
      <StatItem
        label="Time Steps"
        value={meta.numberOfTimeSteps.toLocaleString()}
        delay={0.3}
      />
      <StatItem
        label="Δt"
        value={meta.dt.toExponential(2)}
        unit=" s"
        delay={0.4}
      />
      <StatItem
        label="Freq Range"
        value={freqRange}
        unit=" GHz"
        delay={0.5}
      />
      <StatItem
        label="Generated"
        value={new Date(meta.generated).toLocaleString()}
        delay={0.6}
      />
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.7 }}
        className="stat-item"
      >
        <span className="stat-label">Engine</span>
        <span
          className="engine-badge"
          title={
            fellBack
              ? `Requested ${BACKEND_SHORT_LABELS[backendChoice]}, ran ${engineLabel} (fallback)`
              : `Engine: ${engineLabel}`
          }
        >
          <span className="engine-badge-dot" />
          {engineLabel}
          {fellBack && <span className="engine-badge-fallback"> ⤳</span>}
        </span>
      </motion.div>
    </motion.div>
  );
}
