// scenario.js — Load an FDTD problem definition from a JSON scenario file and
// build the full simulation problem from it.
//
// Previously the IFA problem (geometry, materials, sources, boundary, step
// count) was hardcoded inside server.js/index.js. This module externalizes it
// so different "situations" can be described as JSON files under js_fdtd/scenarios/
// and run without touching the engine code. `numberOfTimeSteps` is a per-scenario
// field, so each scenario picks the run length appropriate to it.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { buildGrid } from './grid.js';
import { buildMaterialGrid, computeMaterialComponents, applyPECPlates } from './materials.js';
import { computeGeneralCoefficients, applyLumpedElementCoefficients } from './coefficients.js';
import { initCPML } from './cpml.js';
import { initWaveforms, initVoltageSources } from './sources.js';
import {
  initSampledElectricFields, initSampledMagneticFields,
  initSampledVoltages, initSampledCurrents
} from './sampling.js';
import { initFarfield } from './farfield.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const SCENARIOS_DIR = path.join(__dirname, '../scenarios');

// Fallbacks applied when a scenario omits a field. Mirrors the values that used
// to be hardcoded in buildSimulationProblem().
const DEFAULTS = {
  courantFactor: 0.9,
  numberOfCellsPerWavelength: 20,
  numberOfTimeSteps: 4000,
  cell: { dx: 0.262e-3, dy: 0.4e-3, dz: 0.4e-3 },
  spheres: [],
  dft: { maxFreq: 10e9, step: 20e6 },
  farfield: { frequencies: [], number_of_cells_from_outer_boundary: 13 },
};

/** List available scenario names (filenames without the .json extension). */
export function listScenarios() {
  if (!fs.existsSync(SCENARIOS_DIR)) return [];
  return fs.readdirSync(SCENARIOS_DIR)
    .filter(f => f.endsWith('.json'))
    .map(f => f.replace(/\.json$/, ''))
    .sort();
}

/** Read a scenario by name (looked up in SCENARIOS_DIR) or by explicit path. */
export function loadScenario(nameOrPath) {
  let file = nameOrPath;
  if (!file.endsWith('.json')) {
    file = path.join(SCENARIOS_DIR, `${nameOrPath}.json`);
  } else if (!path.isAbsolute(file)) {
    file = path.resolve(file);
  }
  if (!fs.existsSync(file)) {
    const avail = listScenarios().join(', ') || '(none)';
    throw new Error(`Scenario not found: ${nameOrPath}. Available: ${avail}`);
  }
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

/**
 * Build the full simulation problem from a parsed scenario object.
 * Returns the same shape buildSimulationProblem() used to return, plus a
 * `meta` block (name/description) and the per-scenario `dft` window.
 *
 * FDTD_STEPS env var still overrides the step count for quick experiments.
 */
export function buildProblemFromScenario(scenario) {
  const courantFactor = scenario.courantFactor ?? DEFAULTS.courantFactor;
  const numberOfCellsPerWavelength =
    scenario.numberOfCellsPerWavelength ?? DEFAULTS.numberOfCellsPerWavelength;
  const numberOfTimeSteps =
    Number(process.env.FDTD_STEPS) || scenario.numberOfTimeSteps || DEFAULTS.numberOfTimeSteps;

  const { dx, dy, dz } = { ...DEFAULTS.cell, ...(scenario.cell || {}) };

  const boundary = scenario.boundary;
  const materialTypes = scenario.materialTypes;
  const bricks = scenario.bricks || [];
  const spheres = scenario.spheres || DEFAULTS.spheres;

  const grid = buildGrid({ dx, dy, dz, boundary, bricks, spheres, courantFactor, numberOfTimeSteps });
  console.log(`Grid: ${grid.nx} x ${grid.ny} x ${grid.nz}  (${numberOfTimeSteps} steps)`);

  const matGrid = buildMaterialGrid(grid, bricks, spheres);
  const matComps = computeMaterialComponents(matGrid, materialTypes, grid);
  applyPECPlates(bricks, materialTypes, matComps, grid);

  const dt = grid.dt;
  let coeffs = computeGeneralCoefficients(matComps, grid);
  const cpml = initCPML(boundary, coeffs, grid);

  const waveforms = initWaveforms(
    scenario.waveforms, numberOfTimeSteps, dt, numberOfCellsPerWavelength, [dx, dy, dz]
  );

  const voltageSources = scenario.voltageSources || [];
  initVoltageSources(voltageSources, waveforms, grid);
  // Signature: (coeffs, matComps, grid, voltageSources, currentSources, ...).
  applyLumpedElementCoefficients(coeffs, matComps, grid, voltageSources, [], [], [], [], []);

  const sampledVoltages = scenario.sampledVoltages || [];
  const sampledCurrents = scenario.sampledCurrents || [];
  const ports = scenario.ports || [];
  const sampledEFields = scenario.sampledElectricFields || [];
  const sampledHFields = scenario.sampledMagneticFields || [];
  initSampledVoltages(sampledVoltages, grid);
  initSampledCurrents(sampledCurrents, grid);
  initSampledElectricFields(sampledEFields, grid);
  initSampledMagneticFields(sampledHFields, grid);

  const ffParams = { ...DEFAULTS.farfield, ...(scenario.farfield || {}) };
  const frequencies = ffParams.frequencies || [];
  const farfield = initFarfield(ffParams, grid);

  const dft = { ...DEFAULTS.dft, ...(scenario.dft || {}) };

  return {
    grid, coeffs, cpml,
    samplers: { sampledVoltages, sampledCurrents, sampledEFields, sampledHFields },
    sources: { voltageSources, currentSources: [], inductors: [], diodes: [] },
    farfield, ports, frequencies,
    numberOfTimeSteps, dt, dft,
    meta: { name: scenario.name, description: scenario.description },
  };
}
