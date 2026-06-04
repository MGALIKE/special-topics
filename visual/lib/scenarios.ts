// Shared types for the scenario picker.
//
// A "scenario" is a problem definition the engine loads from a JSON file under
// js_fdtd/scenarios/. The engine server exposes them via GET /scenarios and runs
// the chosen one when POST /simulate is called with { scenario: <name> }.

export interface ScenarioInfo {
  /** Filename stem, e.g. "ifa-fr4-substrate" — this is what /simulate expects. */
  name: string;
  /** Human-readable title from the scenario's `name` field. */
  title: string;
  /** Long description of what the scenario analyzes. */
  description: string;
  /** Time-step count baked into the scenario (informational). */
  numberOfTimeSteps: number | null;
}

/** Shape returned by GET /scenarios. */
export interface ScenariosResponse {
  scenarios: ScenarioInfo[];
  default: string;
}
