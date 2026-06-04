import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { calcRadiatedPower, calcDirectivity } from './farfield.js';
import { computeObserverDFT } from './dft.js';
import { postProcessSParameters } from './sparameters.js';
import { selectBackend, listBackends } from './backends/registry.js';
import { loadScenario, listScenarios, buildProblemFromScenario } from './scenario.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  // Scenario selection: --scenario=<name> CLI arg or FDTD_SCENARIO env var.
  // Geometry/materials/sources/step-count all come from the scenario JSON now;
  // see src/scenarios/ and src/scenario.js. FDTD_STEPS still overrides the step count.
  const scenarioName =
    (process.argv.find(a => a.startsWith('--scenario')) || '').split('=')[1] ||
    process.env.FDTD_SCENARIO || 'ifa-dualband-baseline';

  console.log(`Initializing FDTD problem from scenario "${scenarioName}"...`);
  console.log(`Available scenarios: ${listScenarios().join(', ')}`);

  const scenario = loadScenario(scenarioName);
  const sim = buildProblemFromScenario(scenario);

  const {
    grid, coeffs, cpml, samplers, farfield, ports, frequencies,
    numberOfTimeSteps, dt, dft,
  } = sim;

  // WebAssembly is natively isolated from V8 Heap arrays unless we create a custom memory allocator.
  // The JS loops in fdtdSolver.js use unrolled 1D Float64Arrays and will execute via V8 JIT at equivalent native speeds.
  const wasmBuffer = fs.readFileSync('build/fdtd_kernels.wasm');
  
  // ─── Select compute backend ──────────────────────────────────────────────
  // CLI: --backend=webgpu|cuda|wasm-cpu|auto  (default auto -> best available)
  const backendArg = (process.argv.find(a => a.startsWith('--backend')) || '')
    .split('=')[1] || process.env.FDTD_BACKEND || 'auto';

  const probed = await listBackends();
  console.log('Available backends: ' +
    probed.map(b => `${b.name}${b.available ? '' : '(x)'}`).join(', '));

  const { name: backendName, backend } = await selectBackend(backendArg);
  console.log(`Running FDTD on backend: ${backendName}`);

  const problem = {
    grid, coeffs, cpml, samplers,
    sources: sim.sources,
    ff: farfield,
    wasmBuffer,
    options: { batchSize: 50 },
  };

  const start = Date.now();
  for await (const snap of backend.run(problem)) {
    process.stdout.write(`\r[${snap.backend}] Step ${snap.step} / ${snap.total} - elapsed ${snap.elapsed.toFixed(1)}s    `);
  }
  console.log(`\nSimulation complete in ${(Date.now()-start)/1000}s. Processing records...`);
  
  // DFT for S-Parameters (frequency window comes from the scenario's dft block)
  const maxFreq = dft?.maxFreq ?? 10e9, fStep = dft?.step ?? 20e6;
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
      scenario: sim.meta?.name ?? scenarioName,
      description: sim.meta?.description ?? null,
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
