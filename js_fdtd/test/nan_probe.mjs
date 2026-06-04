// NaN probe: runs the real antenna problem on wasm-cpu and reports the first
// step at which any field (or the sampled voltage) becomes non-finite, plus
// which field component blows up first and where. Mirrors src/index.js setup.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { buildGrid } from '../src/grid.js';
import { buildMaterialGrid, computeMaterialComponents, applyPECPlates } from '../src/materials.js';
import { computeGeneralCoefficients, applyLumpedElementCoefficients } from '../src/coefficients.js';
import { initCPML } from '../src/cpml.js';
import { initWaveforms, initVoltageSources } from '../src/sources.js';
import { initSampledElectricFields, initSampledMagneticFields, initSampledVoltages, initSampledCurrents } from '../src/sampling.js';
import { initFarfield } from '../src/farfield.js';
import { selectBackend } from '../src/backends/registry.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const courantFactor = 0.9, numberOfCellsPerWavelength = 20;
const numberOfTimeSteps = Number(process.env.FDTD_STEPS) || 200;
const dx = 0.262e-3, dy = 0.4e-3, dz = 0.4e-3;
const boundary = { type_xn:'cpml',type_xp:'cpml',type_yn:'cpml',type_yp:'cpml',type_zn:'cpml',type_zp:'cpml',
  air_buffer_xn:10,air_buffer_xp:10,air_buffer_yn:10,air_buffer_yp:10,air_buffer_zn:10,air_buffer_zp:10,
  cpml_cells_xn:8,cpml_cells_xp:8,cpml_cells_yn:8,cpml_cells_yp:8,cpml_cells_zn:8,cpml_cells_zp:8,
  cpml_order:3,cpml_sigma_factor:1.3,cpml_kappa_max:7,cpml_alpha_min:0,cpml_alpha_max:0.05 };
const materialTypes = [{eps_r:1,mu_r:1,sigma_e:0,sigma_m:0},{eps_r:1,mu_r:1,sigma_e:1e10,sigma_m:0},
  {eps_r:1,mu_r:1,sigma_e:0,sigma_m:1e10},{eps_r:2.2,mu_r:1,sigma_e:0,sigma_m:0}];
const bricks = [
  {min_x:-0.787e-3,min_y:0,min_z:0,max_x:0,max_y:40e-3,max_z:40e-3,material_type:4},
  {min_x:0,min_y:0,min_z:24e-3,max_x:0,max_y:28.4e-3,max_z:26.4e-3,material_type:2},
  {min_x:0,min_y:16e-3,min_z:30e-3,max_x:0,max_y:28.4e-3,max_z:32.4e-3,material_type:2},
  {min_x:0,min_y:26e-3,min_z:8.4e-3,max_x:0,max_y:28.4e-3,max_z:32.4e-3,material_type:2},
  {min_x:0,min_y:20.8e-3,min_z:16e-3,max_x:0,max_y:23.2e-3,max_z:32.4e-3,material_type:2},
  {min_x:-0.787e-3,min_y:16e-3,min_z:30e-3,max_x:0,max_y:16e-3,max_z:32.4e-3,material_type:2},
  {min_x:-0.787e-3,min_y:0,min_z:0,max_x:-0.787e-3,max_y:16e-3,max_z:40e-3,material_type:2}];

const grid = buildGrid({ dx,dy,dz,boundary,bricks,spheres:[],courantFactor,numberOfTimeSteps });
console.log(`Grid ${grid.nx}x${grid.ny}x${grid.nz}  dt=${grid.dt.toExponential(4)}s`);
const matGrid = buildMaterialGrid(grid, bricks, []);
const matComps = computeMaterialComponents(matGrid, materialTypes, grid);
applyPECPlates(bricks, materialTypes, matComps, grid);
let coeffs = computeGeneralCoefficients(matComps, grid);
const cpml = initCPML(boundary, coeffs, grid);
let waveforms = { gaussian: [{number_of_cells_per_wavelength:0},{number_of_cells_per_wavelength:15}] };
waveforms = initWaveforms(waveforms, numberOfTimeSteps, grid.dt, numberOfCellsPerWavelength, [dx,dy,dz]);
const voltageSources = [{min_x:-0.787e-3,min_y:0,min_z:24e-3,max_x:0,max_y:0,max_z:26.4e-3,direction:'xp',resistance:50,magnitude:1,waveform_type:'gaussian',waveform_index:1}];
initVoltageSources(voltageSources, waveforms, grid);
applyLumpedElementCoefficients(coeffs, matComps, grid, voltageSources, [], [], [], [], []);
const sampledVoltages = [{min_x:-0.787e-3,min_y:0,min_z:24.4e-3,max_x:0,max_y:0,max_z:26.4e-3,direction:'xp',label:'v1'}];
const sampledCurrents = [{min_x:-0.39e-3,min_y:0,min_z:24e-3,max_x:-0.39e-3,max_y:0,max_z:26.4e-3,direction:'xp',label:'i1'}];
initSampledVoltages(sampledVoltages, grid);
initSampledCurrents(sampledCurrents, grid);
const samplers = { sampledVoltages, sampledCurrents, sampledEFields:[], sampledHFields:[] };
const farfield = initFarfield({ frequencies:[2.4e9,5.8e9], number_of_cells_from_outer_boundary:13 }, grid);
const wasmBuffer = fs.readFileSync(path.join(__dirname,'../build/fdtd_kernels.wasm'));

const { backend } = await selectBackend('wasm-cpu');

function scanField(name, arr) {
  for (let n = 0; n < arr.length; n++) {
    const v = arr[n];
    if (!Number.isFinite(v)) return { name, index: n, value: v };
  }
  return null;
}
function scanAll(f) {
  for (const k of ['Hx','Hy','Hz','Ex','Ey','Ez']) {
    const hit = scanField(k, f[k]);
    if (hit) return hit;
  }
  return null;
}
function maxAbs(f) {
  let m = 0;
  for (const k of ['Ex','Ey','Ez']) { const a=f[k]; for (let n=0;n<a.length;n++){const v=Math.abs(a[n]); if(v>m)m=v;} }
  return m;
}

const batchSize = Number(process.env.BATCH) || 25;
const problem = { grid, coeffs, cpml, samplers, sources:{voltageSources,currentSources:[],inductors:[],diodes:[]}, ff:farfield, wasmBuffer, options:{batchSize} };

let firstNaN = null;
for await (const snap of backend.run(problem)) {
  const f = grid.fields;
  const hit = scanAll(f);
  const v = sampledVoltages[0].sampled_value[snap.step-1];
  const peakE = maxAbs(f);
  console.log(`step ${String(snap.step).padStart(4)}  V=${(v??NaN).toExponential(3)}  peak|E|=${peakE.toExponential(3)}` + (hit?`  <-- non-finite in ${hit.name}[${hit.index}]=${hit.value}`:''));
  if (hit && !firstNaN) { firstNaN = { step: snap.step, ...hit }; break; }
}
// Final full scan of the voltage trace
const vt = sampledVoltages[0].sampled_value;
let vNaN = 0; for (let n=0;n<vt.length;n++) if(!Number.isFinite(vt[n])) vNaN++;
if (firstNaN) console.log(`\nFIRST NON-FINITE FIELD: step ${firstNaN.step}, ${firstNaN.name}[${firstNaN.index}]=${firstNaN.value}`);
else console.log('\nNo non-finite field values detected during the run.');
console.log(`Voltage trace: ${vNaN} of ${vt.length} samples non-finite.`);
