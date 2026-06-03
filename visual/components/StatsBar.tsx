"use client";

import { motion } from "framer-motion";

interface StatsBarProps {
  meta: {
    numberOfTimeSteps: number;
    dt: number;
    generated: string;
    frequencies_Hz?: number[];
  };
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

export default function StatsBar({ meta }: StatsBarProps) {
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
    </motion.div>
  );
}
