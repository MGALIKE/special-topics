// test/validate_webgpu.js
// ---------------------------------------------------------------------------
// Validates the WebGPU backend against the golden wasm-cpu backend on a SMALL
// grid (~300 steps), diffing the sampled voltage trace within rtol=1e-3 (f32).
//
//   node test/validate_webgpu.js
//
// Requires the optional `webgpu` npm package for the WebGPU run:
//   npm i webgpu
// If it is absent (or no adapter), the script prints a clear SKIP and exits 0.
//
// This script ONLY imports the existing public init helpers — it does not modify
// any engine files. It builds the problem TWICE (one fresh grid per backend) so
// each backend starts from zeroed fields.
// ---------------------------------------------------------------------------

import fs from 'fs';
import { buildGrid } from '../src/grid.js';
import {
  buildMaterialGrid, computeMaterialComponents, applyPECPlates,
} from '../src/materials.js';
import {
  computeGeneralCoefficients, applyLumpedElementCoefficients,
} from '../src/coefficients.js';
import { initCPML } from '../src/cpml.js';
import { initWaveforms, initVoltageSources } from '../src/sources.js';
import {
  initSampledElectricFields, initSampledMagneticFields,
  initSampledVoltages, initSampledCurrents,
} from '../src/sampling.js';
import { initFarfield } from '../src/farfield.js';

import wasmCpuBackend from '../src/backends/wasmCpu.js';
import webgpuBackend from '../src/backends/webgpu/nodeBackend.js';

const NUM_STEPS = 300;

// A deliberately small problem: a single dielectric brick + one voltage source,
// CPML on all faces. Mirrors the structure of index.js but tiny.
function buildProblem() {
  const courantFactor = 0.9;
  const numberOfCellsPerWavelength = 20;
  const numberOfTimeSteps = NUM_STEPS;

  const dx = 1e-3, dy = 1e-3, dz = 1e-3;

  const boundary = {
    type_xn: 'cpml', type_xp: 'cpml',
    type_yn: 'cpml', type_yp: 'cpml',
    type_zn: 'cpml', type_zp: 'cpml',
    air_buffer_xn: 3, air_buffer_xp: 3,
    air_buffer_yn: 3, air_buffer_yp: 3,
    air_buffer_zn: 3, air_buffer_zp: 3,
    cpml_cells_xn: 5, cpml_cells_xp: 5,
    cpml_cells_yn: 5, cpml_cells_yp: 5,
    cpml_cells_zn: 5, cpml_cells_zp: 5,
    cpml_order: 3, cpml_sigma_factor: 1.3,
    cpml_kappa_max: 7, cpml_alpha_min: 0, cpml_alpha_max: 0.05,
  };

  const materialTypes = [
    { eps_r: 1, mu_r: 1, sigma_e: 0, sigma_m: 0 },
    { eps_r: 1, mu_r: 1, sigma_e: 1e10, sigma_m: 0 },
    { eps_r: 1, mu_r: 1, sigma_e: 0, sigma_m: 1e10 },
    { eps_r: 2.2, mu_r: 1, sigma_e: 0, sigma_m: 0 },
  ];

  // Small dielectric brick spanning a few cells.
  const bricks = [
    { min_x: 0, min_y: 0, min_z: 0, max_x: 4e-3, max_y: 4e-3, max_z: 4e-3, material_type: 4 },
  ];
  const spheres = [];

  const grid = buildGrid({ dx, dy, dz, boundary, bricks, spheres, courantFactor, numberOfTimeSteps });
  const dt = grid.dt;

  const matGrid = buildMaterialGrid(grid, bricks, spheres);
  const matComps = computeMaterialComponents(matGrid, materialTypes, grid);
  applyPECPlates(bricks, materialTypes, matComps, grid);

  const coeffs = computeGeneralCoefficients(matComps, grid);
  const cpml = initCPML(boundary, coeffs, grid);

  let waveforms = { gaussian: [{ number_of_cells_per_wavelength: 0 }] };
  waveforms = initWaveforms(waveforms, numberOfTimeSteps, dt, numberOfCellsPerWavelength, [dx, dy, dz]);

  const voltageSources = [{
    min_x: 0, min_y: 0, min_z: 0, max_x: 4e-3, max_y: 0, max_z: 2e-3,
    direction: 'xp', resistance: 50, magnitude: 1,
    waveform_type: 'gaussian', waveform_index: 1,
  }];
  initVoltageSources(voltageSources, waveforms, grid);
  applyLumpedElementCoefficients(coeffs, matComps, grid, voltageSources, [], [], [], [], []);

  const sampledVoltages = [{
    min_x: 0, min_y: 0, min_z: 0, max_x: 4e-3, max_y: 0, max_z: 2e-3,
    direction: 'xp', label: 'v1',
  }];
  const sampledCurrents = [];
  const sampledEFields = [];
  const sampledHFields = [];
  initSampledVoltages(sampledVoltages, grid);
  initSampledCurrents(sampledCurrents, grid);
  initSampledElectricFields(sampledEFields, grid);
  initSampledMagneticFields(sampledHFields, grid);
  const samplers = { sampledVoltages, sampledCurrents, sampledEFields, sampledHFields };

  // No far-field for the fast validation (keeps it quick); set [] frequencies.
  const ff = initFarfield({ frequencies: [], number_of_cells_from_outer_boundary: 1 }, grid);

  const wasmBuffer = fs.readFileSync(new URL('../build/fdtd_kernels.wasm', import.meta.url));

  return {
    grid, coeffs, cpml, samplers,
    sources: { voltageSources, currentSources: [], inductors: [], diodes: [] },
    ff, wasmBuffer, options: { batchSize: 50 },
  };
}

async function drain(backend, problem) {
  // eslint-disable-next-line no-unused-vars
  for await (const _snap of backend.run(problem)) { /* consume */ }
  return problem.samplers.sampledVoltages[0].sampled_value;
}

async function main() {
  console.log(`[validate] grid + ${NUM_STEPS} steps`);

  // WebGPU availability check first — skip cleanly if unavailable.
  const available = await webgpuBackend.isAvailable();
  if (!available) {
    console.log('[validate] SKIP: WebGPU device unavailable in this Node process.');
    console.log('           Install the optional dependency:  npm i webgpu');
    process.exit(0);
  }

  const refProblem = buildProblem();
  const refTrace = await drain(wasmCpuBackend, refProblem);

  const gpuProblem = buildProblem();
  const gpuTrace = await drain(webgpuBackend, gpuProblem);

  // Compare with rtol=1e-3 against the peak magnitude of the reference trace.
  let peak = 0;
  for (let i = 0; i < refTrace.length; i++) peak = Math.max(peak, Math.abs(refTrace[i]));
  const tol = 1e-3 * (peak || 1);

  let maxAbsErr = 0, argmax = -1;
  for (let i = 0; i < refTrace.length; i++) {
    const e = Math.abs(refTrace[i] - gpuTrace[i]);
    if (e > maxAbsErr) { maxAbsErr = e; argmax = i; }
  }

  console.log(`[validate] peak |V| = ${peak.toExponential(4)}`);
  console.log(`[validate] max abs err = ${maxAbsErr.toExponential(4)} at step ${argmax} (tol ${tol.toExponential(4)})`);

  if (maxAbsErr <= tol) {
    console.log('[validate] PASS — WebGPU matches wasm-cpu within rtol=1e-3.');
    process.exit(0);
  } else {
    console.error('[validate] FAIL — voltage trace mismatch exceeds tolerance.');
    // Print a few sample rows to aid debugging.
    for (let i = 0; i < refTrace.length; i += Math.ceil(refTrace.length / 10)) {
      console.error(`  step ${i}: ref=${refTrace[i].toExponential(4)}  gpu=${gpuTrace[i].toExponential(4)}`);
    }
    process.exit(1);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
