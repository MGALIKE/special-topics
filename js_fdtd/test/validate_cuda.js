// validate_cuda.js — Validation harness for the CUDA backend.
//
// Builds a SMALL FDTD problem (short numberOfTimeSteps), runs it on BOTH the
// wasm-cpu golden reference and the cuda backend, then diffs the sampled
// voltage trace. Per CONTRACT §4: rtol ~1e-3 (f32), far-field within ~0.1 dB.
//
// Usage:  node test/validate_cuda.js
//
// If the cuda addon is not built / no GPU is present, this script SKIPS the GPU
// run and exits 0 with a clear message (so CI without a GPU does not fail).

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { buildGrid } from '../src/grid.js';
import { buildMaterialGrid, computeMaterialComponents, applyPECPlates } from '../src/materials.js';
import { computeGeneralCoefficients, applyLumpedElementCoefficients } from '../src/coefficients.js';
import { initCPML } from '../src/cpml.js';
import { initWaveforms, initVoltageSources } from '../src/sources.js';
import { initSampledVoltages, initSampledCurrents,
         initSampledElectricFields, initSampledMagneticFields } from '../src/sampling.js';
import { initFarfield } from '../src/farfield.js';
import { loadBackend } from '../src/backends/registry.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Build a compact problem (same geometry as index.js, fewer steps) ─────────
function buildProblem(numberOfTimeSteps) {
  const courantFactor = 0.9;
  const numberOfCellsPerWavelength = 20;
  const dx = 0.262e-3, dy = 0.4e-3, dz = 0.4e-3;

  const boundary = {
    type_xn:'cpml',type_xp:'cpml',type_yn:'cpml',type_yp:'cpml',type_zn:'cpml',type_zp:'cpml',
    air_buffer_xn:10,air_buffer_xp:10,air_buffer_yn:10,air_buffer_yp:10,air_buffer_zn:10,air_buffer_zp:10,
    cpml_cells_xn:8,cpml_cells_xp:8,cpml_cells_yn:8,cpml_cells_yp:8,cpml_cells_zn:8,cpml_cells_zp:8,
    cpml_order:3,cpml_sigma_factor:1.3,cpml_kappa_max:7,cpml_alpha_min:0,cpml_alpha_max:0.05,
  };
  const materialTypes = [
    { eps_r:1, mu_r:1, sigma_e:0, sigma_m:0 },
    { eps_r:1, mu_r:1, sigma_e:1e10, sigma_m:0 },
    { eps_r:1, mu_r:1, sigma_e:0, sigma_m:1e10 },
    { eps_r:2.2, mu_r:1, sigma_e:0, sigma_m:0 },
  ];
  const bricks = [
    { min_x:-0.787e-3, min_y:0, min_z:0, max_x:0, max_y:40e-3, max_z:40e-3, material_type:4 },
    { min_x:0, min_y:0, min_z:24e-3, max_x:0, max_y:28.4e-3, max_z:26.4e-3, material_type:2 },
  ];
  const spheres = [];

  const grid = buildGrid({ dx, dy, dz, boundary, bricks, spheres, courantFactor, numberOfTimeSteps });
  const matGrid = buildMaterialGrid(grid, bricks, spheres);
  const matComps = computeMaterialComponents(matGrid, materialTypes, grid);
  applyPECPlates(bricks, materialTypes, matComps, grid);

  const dt = grid.dt;
  const coeffs = computeGeneralCoefficients(matComps, grid);
  const cpml = initCPML(boundary, coeffs, grid);

  let waveforms = { gaussian: [ { number_of_cells_per_wavelength: 0 } ] };
  waveforms = initWaveforms(waveforms, numberOfTimeSteps, dt, numberOfCellsPerWavelength, [dx,dy,dz]);

  const voltageSources = [{
    min_x:-0.787e-3, min_y:0, min_z:24e-3, max_x:0, max_y:0, max_z:26.4e-3,
    direction:'xp', resistance:50, magnitude:1, waveform_type:'gaussian', waveform_index:1,
  }];
  initVoltageSources(voltageSources, waveforms, grid);
  // Signature: (coeffs, mc, grid, voltageSources, currentSources, resistors, capacitors, inductors, diodes)
  applyLumpedElementCoefficients(coeffs, matComps, grid, voltageSources, [], [], [], [], []);

  const sampledVoltages = [{
    min_x:-0.787e-3, min_y:0, min_z:24.4e-3, max_x:0, max_y:0, max_z:26.4e-3, direction:'xp', label:'v1',
  }];
  const sampledCurrents = [{
    min_x:-0.39e-3, min_y:0, min_z:24e-3, max_x:-0.39e-3, max_y:0, max_z:26.4e-3, direction:'xp', label:'i1',
  }];
  const sampledEFields = [], sampledHFields = [];
  initSampledVoltages(sampledVoltages, grid);
  initSampledCurrents(sampledCurrents, grid);
  initSampledElectricFields(sampledEFields, grid);
  initSampledMagneticFields(sampledHFields, grid);

  const ff = initFarfield({ frequencies: [2.4e9, 5.8e9], number_of_cells_from_outer_boundary: 13 }, grid);

  return {
    grid, coeffs, cpml,
    samplers: { sampledVoltages, sampledCurrents, sampledEFields, sampledHFields },
    sources: { voltageSources, currentSources: [], inductors: [], diodes: [] },
    ff,
    wasmBuffer: fs.readFileSync(path.join(__dirname, '..', 'build', 'fdtd_kernels.wasm')),
    options: { batchSize: 50 },
  };
}

async function runAll(backend, problem) {
  for await (const _snap of backend.run(problem)) { /* drain */ }
  return problem.samplers.sampledVoltages[0].sampled_value.slice();
}

async function main() {
  const N = 300;
  console.log(`Validating CUDA backend on a ${N}-step problem...`);

  const cuda = await loadBackend('cuda');
  if (!cuda || !(await cuda.isAvailable())) {
    console.log('SKIP: cuda backend unavailable (addon not built or no GPU). Falling back is correct.');
    process.exit(0);
  }

  const wasm = await loadBackend('wasm-cpu');
  const refTrace  = await runAll(wasm, buildProblem(N));
  const cudaTrace = await runAll(cuda, buildProblem(N));

  let maxAbs = 0, maxRel = 0, refPeak = 0;
  for (let i = 0; i < N; i++) refPeak = Math.max(refPeak, Math.abs(refTrace[i]));
  for (let i = 0; i < N; i++) {
    const a = Math.abs(cudaTrace[i] - refTrace[i]);
    maxAbs = Math.max(maxAbs, a);
    const denom = Math.max(Math.abs(refTrace[i]), refPeak * 1e-3);
    maxRel = Math.max(maxRel, a / denom);
  }

  console.log(`ref peak |V|        = ${refPeak.toExponential(4)}`);
  console.log(`max abs diff        = ${maxAbs.toExponential(4)}`);
  console.log(`max rel diff        = ${maxRel.toExponential(4)}`);

  const TOL = 1e-3;
  if (maxRel <= TOL || maxAbs <= refPeak * TOL) {
    console.log(`PASS: voltage trace matches within rtol=${TOL}.`);
    process.exit(0);
  } else {
    console.error(`FAIL: voltage trace exceeds rtol=${TOL}.`);
    process.exit(1);
  }
}

main().catch(err => { console.error(err); process.exit(1); });
