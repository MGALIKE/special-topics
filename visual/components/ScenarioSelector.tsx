"use client";

import { motion } from "framer-motion";
import type { ScenarioInfo } from "@/lib/scenarios";

interface ScenarioSelectorProps {
  scenarios: ScenarioInfo[];
  value: string;
  onChange: (name: string) => void;
  /** Selection is locked while a run is in flight. */
  disabled?: boolean;
}

/**
 * Picks which problem definition (scenario JSON) the engine loads for the next
 * run. The list comes from the engine server's GET /scenarios; the chosen name
 * is sent to POST /simulate. Mirrors the BackendSelector layout/classes.
 */
export default function ScenarioSelector({
  scenarios,
  value,
  onChange,
  disabled = false,
}: ScenarioSelectorProps) {
  return (
    <div className="engine-block">
      <div className="engine-head">
        <div className="engine-head-title">
          <span className="engine-head-index">0S</span>
          <span>Scenario</span>
        </div>
        <span className="engine-head-note">
          One engine · many problems · pick the antenna / situation to analyze
        </span>
      </div>

      {scenarios.length === 0 ? (
        <div className="scenario-empty">
          No scenarios found — is the engine server running on{" "}
          <code>localhost:4000</code>?
        </div>
      ) : (
        <div className="engine-grid" role="radiogroup" aria-label="Scenario">
          {scenarios.map((s, i) => {
            const active = value === s.name;
            return (
              <motion.button
                key={s.name}
                type="button"
                role="radio"
                aria-checked={active}
                aria-label={s.title}
                title={s.description}
                disabled={disabled}
                onClick={() => !disabled && onChange(s.name)}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.05 * i }}
                className={`engine-card ${active ? "engine-card--active" : ""} ${
                  disabled ? "engine-card--disabled" : ""
                }`}
              >
                <div className="engine-card-top">
                  <div>
                    <div className="engine-card-name scenario-card-name">
                      {s.title}
                    </div>
                    <div className="engine-card-tagline">
                      {s.numberOfTimeSteps != null
                        ? `${s.numberOfTimeSteps.toLocaleString()} steps`
                        : "scenario"}
                    </div>
                  </div>
                </div>

                <p className="engine-card-desc">{s.description}</p>
              </motion.button>
            );
          })}
        </div>
      )}
    </div>
  );
}
