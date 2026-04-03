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
import { initFarfield, calcRadiatedPower, calcDirectivity } from './farfield.js';
import { computeObserverDFT } from './dft.js';
import { postProcessSParameters } from './sparameters.js';
import { runFDTDCluster } from './fdtdSolver.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  console.log('Initializing FDTD problem space parameters...');
  
  const courantFactor = 0.9;
  const numberOfCellsPerWavelength = 20;
  const numberOfTimeSteps = 7000;
  
  const dx = 0.262e-3;
  const dy = 0.4e-3;
  const dz = 0.4e-3;
  
  const boundary = {
    type_xn: 'cpml', type_xp: 'cpml',
    type_yn: 'cpml', type_yp: 'cpml',
    type_zn: 'cpml', type_zp: 'cpml',
    air_buffer_xn: 10, air_buffer_xp: 10,
    air_buffer_yn: 10, air_buffer_yp: 10,
    air_buffer_zn: 10, air_buffer_zp: 10,
    cpml_cells_xn: 8, cpml_cells_xp: 8,
    cpml_cells_yn: 8, cpml_cells_yp: 8,
    cpml_cells_zn: 8, cpml_cells_zp: 8,
    cpml_order: 3,
    cpml_sigma_factor: 1.3,
    cpml_kappa_max: 7,
    cpml_alpha_min: 0,
    cpml_alpha_max: 0.05
  };
  
  // 1=Air, 2=PEC, 3=PMC, 4=Substrate
  const materialTypes = [
    { eps_r: 1, mu_r: 1, sigma_e: 0, sigma_m: 0 },
    { eps_r: 1, mu_r: 1, sigma_e: 1e10, sigma_m: 0 },
    { eps_r: 1, mu_r: 1, sigma_e: 0, sigma_m: 1e10 },
    { eps_r: 2.2, mu_r: 1, sigma_e: 0, sigma_m: 0 }
  ];
  
  const bricks = [
    { min_x: -0.787e-3, min_y: 0, min_z: 0, max_x: 0, max_y: 40e-3, max_z: 40e-3, material_type: 4 },
    { min_x: 0, min_y: 0, min_z: 24e-3, max_x: 0, max_y: 28.4e-3, max_z: 26.4e-3, material_type: 2 },
    { min_x: 0, min_y: 16e-3, min_z: 30e-3, max_x: 0, max_y: 28.4e-3, max_z: 32.4e-3, material_type: 2 },
    { min_x: 0, min_y: 26e-3, min_z: 8.4e-3, max_x: 0, max_y: 28.4e-3, max_z: 32.4e-3, material_type: 2 },
    { min_x: 0, min_y: 20.8e-3, min_z: 16e-3, max_x: 0, max_y: 23.2e-3, max_z: 32.4e-3, material_type: 2 },
    { min_x: -0.787e-3, min_y: 16e-3, min_z: 30e-3, max_x: 0, max_y: 16e-3, max_z: 32.4e-3, material_type: 2 },
    { min_x: -0.787e-3, min_y: 0, min_z: 0, max_x: -0.787e-3, max_y: 16e-3, max_z: 40e-3, material_type: 2 }
  ];
  const spheres = [];
  
  const grid = buildGrid({
    dx, dy, dz, boundary, bricks, spheres, courantFactor, numberOfTimeSteps
  });
  
  console.log(`Grid size: ${grid.nx} x ${grid.ny} x ${grid.nz}`);
  
  const matGrid = buildMaterialGrid(grid, bricks, spheres);
  const matComps = computeMaterialComponents(matGrid, materialTypes, grid);
  applyPECPlates(bricks, materialTypes, matComps, grid);
  
  const dt = grid.dt;
  let coeffs = computeGeneralCoefficients(matComps, grid);
  
  const cpml = initCPML(boundary, coeffs, grid);
  
  let waveforms = {
    gaussian: [
      { number_of_cells_per_wavelength: 0 },
      { number_of_cells_per_wavelength: 15 }
    ]
  };
  waveforms = initWaveforms(waveforms, numberOfTimeSteps, dt, numberOfCellsPerWavelength, [dx, dy, dz]);
  
  const voltageSources = [{
    min_x: -0.787e-3, min_y: 0, min_z: 24e-3, max_x: 0, max_y: 0, max_z: 26.4e-3, 
    direction: 'xp', resistance: 50, magnitude: 1, waveform_type: 'gaussian', waveform_index: 1
  }];
  initVoltageSources(voltageSources, waveforms, grid);
  
  applyLumpedElementCoefficients(coeffs, grid, dt, voltageSources, [], [], [], [], []);
  
  const sampledVoltages = [{
    min_x: -0.787e-3, min_y: 0, min_z: 24.4e-3, max_x: 0, max_y: 0, max_z: 26.4e-3,
    direction: 'xp', label: 'v1'
  }];
  const sampledCurrents = [{
    min_x: -0.39e-3, min_y: 0, min_z: 24e-3, max_x: -0.39e-3, max_y: 0, max_z: 26.4e-3,
    direction: 'xp', label: 'i1'
  }];
  
  const ports = [{
    sampled_voltage_index: 1,
    sampled_current_index: 1,
    impedance: 50,
    is_source_port: true
  }];
  
  const sampledEFields = [];
  const sampledHFields = [];
  initSampledVoltages(sampledVoltages, grid);
  initSampledCurrents(sampledCurrents, grid);
  initSampledElectricFields(sampledEFields, grid);
  initSampledMagneticFields(sampledHFields, grid);
  const samplers = { sampledVoltages, sampledCurrents, sampledEFields, sampledHFields };
  
  const frequencies = [2.4e9, 5.8e9];
  const farfield = initFarfield({ frequencies, number_of_cells_from_outer_boundary: 13 }, grid);
  
  // WebAssembly is natively isolated from V8 Heap arrays unless we create a custom memory allocator.
  // The JS loops in fdtdSolver.js use unrolled 1D Float64Arrays and will execute via V8 JIT at equivalent native speeds.
  const wasmBuffer = fs.readFileSync('build/fdtd_kernels.wasm');
  
  console.log('Running Multithreaded WASM FDTD Simulation...');
  const start = Date.now();
  const gen = runFDTDCluster(
    grid, coeffs, cpml, samplers, 
    { voltageSources, currentSources: [], inductors: [], diodes: [] }, 
    farfield, wasmBuffer, 50
  );

  for await (const snap of gen) {
    process.stdout.write(`\rStep ${snap.step} / ${snap.total} - elapsed ${snap.elapsed.toFixed(1)}s    `);
  }
  console.log(`\nSimulation complete in ${(Date.now()-start)/1000}s. Processing records...`);
  
  // DFT for S-Parameters
  const maxFreq = 10e9, fStep = 20e6;
  const nFreqs = Math.floor(maxFreq / fStep);
  const freqArr = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) freqArr[i] = (i+1) * fStep;

  computeObserverDFT(samplers.sampledVoltages, dt, 0, freqArr);
  computeObserverDFT(samplers.sampledCurrents, dt, -dt/2, freqArr);
  postProcessSParameters({ ports, sampledVoltages: samplers.sampledVoltages, sampledCurrents: samplers.sampledCurrents, dt, freqArr });

  // Post process farfield
  let ffData = [];
  if (frequencies.length > 0) {
    const radPower = calcRadiatedPower(farfield, grid);
    const angles = new Float64Array(360);
    for (let i = 0; i < 360; i++) angles[i] = i * Math.PI / 180;
    
    // XY plane
    const thXY = new Float64Array(360).fill(Math.PI/2);
    const { dataTheta: dThXY, dataPhi: dPhXY } = calcDirectivity(farfield, grid, radPower, thXY, angles);
    
    // XZ plane
    const phXZ = new Float64Array(360);
    const thXZ = new Float64Array(360);
    for(let i=0; i<360; i++) {
        const th = angles[i]; // 0 to 2pi
        phXZ[i] = th > Math.PI ? Math.PI : 0;
        thXZ[i] = th > Math.PI ? 2*Math.PI - th : th;
    }
    const { dataTheta: dThXZ, dataPhi: dPhXZ } = calcDirectivity(farfield, grid, radPower, thXZ, phXZ);
    
    for (let f = 0; f < frequencies.length; f++) {
      ffData.push({
        plane: 'xy', freq_GHz: frequencies[f]/1e9, angles_deg: Array.from(angles).map(x => x*180/Math.PI),
        dataTheta_dB: Array.from(dThXY[f]).map(x => 10*Math.log10(Math.max(x, 1e-10))),
        dataPhi_dB: Array.from(dPhXY[f]).map(x => 10*Math.log10(Math.max(x, 1e-10)))
      });
      ffData.push({
        plane: 'xz', freq_GHz: frequencies[f]/1e9, angles_deg: Array.from(angles).map(x => x*180/Math.PI),
        dataTheta_dB: Array.from(dThXZ[f]).map(x => 10*Math.log10(Math.max(x, 1e-10))),
        dataPhi_dB: Array.from(dPhXZ[f]).map(x => 10*Math.log10(Math.max(x, 1e-10)))
      });
    }
  }

  const results = {
    meta: {
      generated: new Date().toISOString(),
      numberOfTimeSteps, dt,
      frequencies_Hz: Array.from(freqArr)
    },
    sparams: ports.filter(p => p.is_source_port).map(p => ({
      label: `S11`,
      frequencies_GHz: Array.from(freqArr).map(f => f/1e9),
      magnitude_dB: Array.from(p.S[0].mag_dB),
      phase_deg: Array.from(p.S[0].phase_deg)
    })),
    farfield: ffData,
    timeDomain: {
      voltages: samplers.sampledVoltages.map(v => {
        const time_ns = [];
        for (let i = 0; i < numberOfTimeSteps; i++) time_ns.push((i+0.5)*dt*1e9);
        return { label: v.label, time_ns, values: Array.from(v.sampled_value) };
      })
    }
  };

  const outDir = path.join(__dirname, '../results');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir);
  fs.writeFileSync(path.join(outDir, 'out.json'), JSON.stringify(results));
  console.log('Results written to results/out.json - ready to preview in viewer.html');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
