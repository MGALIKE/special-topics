"use client";

import { motion } from "framer-motion";
import type { SimStatus } from "@/lib/useSimulation";
import type { BackendChoice } from "@/lib/backends";
import type { ScenarioInfo } from "@/lib/scenarios";
import BackendSelector from "@/components/BackendSelector";
import ScenarioSelector from "@/components/ScenarioSelector";

interface SimulationControlsProps {
  status: SimStatus;
  step: number;
  total: number;
  elapsed: number;
  percent: string;
  error: string | null;
  onStart: () => void;
  backendChoice: BackendChoice;
  onBackendChange: (choice: BackendChoice) => void;
  scenarios: ScenarioInfo[];
  scenario: string;
  onScenarioChange: (name: string) => void;
}

const STATUS_LABELS: Record<SimStatus, string> = {
  disconnected: "Backend Offline",
  idle: "Ready",
  initializing: "Initializing Grid...",
  running: "Simulating...",
  postprocessing: "Post-Processing...",
  done: "Complete",
  error: "Error",
};

const STATUS_COLORS: Record<SimStatus, string> = {
  disconnected: "#ff3b1f",
  idle: "#6a6a6a",
  initializing: "#f4f4f1",
  running: "#ff3b1f",
  postprocessing: "#f4f4f1",
  done: "#f4f4f1",
  error: "#ff3b1f",
};

export default function SimulationControls({
  status,
  step,
  total,
  elapsed,
  percent,
  error,
  onStart,
  backendChoice,
  onBackendChange,
  scenarios,
  scenario,
  onScenarioChange,
}: SimulationControlsProps) {
  const isRunning =
    status === "running" ||
    status === "initializing" ||
    status === "postprocessing";
  const canStart = status === "idle" || status === "done" || status === "error" || status === "disconnected";

  const eta =
    status === "running" && step > 0
      ? ((elapsed / step) * (total - step)).toFixed(1)
      : null;

  return (
    <div className="sim-controls">
      {/* Scenario (problem definition) selector */}
      <ScenarioSelector
        scenarios={scenarios}
        value={scenario}
        onChange={onScenarioChange}
        disabled={isRunning}
      />

      {/* Compute-engine selector */}
      <BackendSelector value={backendChoice} onChange={onBackendChange} />

      {/* Status indicator */}
      <div className="sim-status-row">
        <div className="sim-status-indicator">
          <motion.span
            className="sim-status-dot"
            style={{ backgroundColor: STATUS_COLORS[status] }}
            animate={
              isRunning
                ? { scale: [1, 1.3, 1], opacity: [1, 0.6, 1] }
                : {}
            }
            transition={isRunning ? { duration: 1.5, repeat: Infinity } : {}}
          />
          <span className="sim-status-text">{STATUS_LABELS[status]}</span>
        </div>

        <motion.button
          whileHover={canStart ? { scale: 1.05 } : {}}
          whileTap={canStart ? { scale: 0.95 } : {}}
          onClick={onStart}
          disabled={!canStart}
          className={`sim-run-btn ${canStart ? "sim-run-btn--active" : "sim-run-btn--disabled"}`}
        >
          {isRunning ? (
            <>
              <span className="sim-spinner" />
              Running...
            </>
          ) : status === "done" ? (
            <>Re-run Simulation</>
          ) : (
            <>Run Simulation</>
          )}
        </motion.button>
      </div>

      {/* Progress bar */}
      {(isRunning || status === "done") && (
        <div className="sim-progress-container">
          <div className="sim-progress-bar">
            <motion.div
              className="sim-progress-fill"
              initial={{ width: 0 }}
              animate={{ width: `${percent}%` }}
              transition={{ duration: 0.3 }}
              style={{
                background: status === "done" ? "#f4f4f1" : "#ff3b1f",
              }}
            />
          </div>
          <div className="sim-progress-info">
            <span>
              Step {step.toLocaleString()} / {total.toLocaleString()}
            </span>
            <span>{percent}%</span>
            <span>{elapsed.toFixed(1)}s elapsed</span>
            {eta && <span>~{eta}s remaining</span>}
          </div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="sim-error">
          ⚠ {error}
        </div>
      )}
    </div>
  );
}
