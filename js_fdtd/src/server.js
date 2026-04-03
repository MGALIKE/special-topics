// server.js — Lightweight HTTP + SSE server for real-time FDTD simulation streaming.
// Zero external dependencies — uses only Node built-in `http`.
//
// Endpoints:
//   POST /simulate  — Start a new simulation run
//   GET  /stream    — SSE stream of real-time events
//   GET  /status    — Current simulation state
//   GET  /results   — Final results JSON (after sim completes)

import http from 'http';
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

// ── Simulation State ─────────────────────────────────────────────────────────
let simState = {
  status: 'idle',       // idle | initializing | running | postprocessing | done | error
  step: 0,
  total: 0,
  elapsed: 0,
  percent: '0.0',
  error: null,
  results: null,        // final results JSON (populated after done)
  voltageHistory: [],   // real-time voltage samples [{time_ns, value}]
};

// SSE clients
const sseClients = new Set();

function sendSSE(event, data) {
  const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of sseClients) {
    try { res.write(msg); } catch { /* client disconnected */ }
  }
}

// ── Simulation Setup & Run ───────────────────────────────────────────────────

function buildSimulationProblem() {
  const courantFactor = 0.9;
  const numberOfCellsPerWavelength = 20;
  const numberOfTimeSteps = 7000;

  const dx = 0.262e-3, dy = 0.4e-3, dz = 0.4e-3;

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
    cpml_order: 3, cpml_sigma_factor: 1.3,
    cpml_kappa_max: 7, cpml_alpha_min: 0, cpml_alpha_max: 0.05
  };

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

  const grid = buildGrid({ dx, dy, dz, boundary, bricks, spheres: [], courantFactor, numberOfTimeSteps });
  console.log(`Grid: ${grid.nx} x ${grid.ny} x ${grid.nz}`);

  const matGrid = buildMaterialGrid(grid, bricks, []);
  const matComps = computeMaterialComponents(matGrid, materialTypes, grid);
  applyPECPlates(bricks, materialTypes, matComps, grid);

  const dt = grid.dt;
  let coeffs = computeGeneralCoefficients(matComps, grid);
  const cpml = initCPML(boundary, coeffs, grid);

  let waveforms = { gaussian: [{ number_of_cells_per_wavelength: 0 }, { number_of_cells_per_wavelength: 15 }] };
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
    sampled_voltage_index: 1, sampled_current_index: 1,
    impedance: 50, is_source_port: true
  }];
  const sampledEFields = [];
  const sampledHFields = [];
  initSampledVoltages(sampledVoltages, grid);
  initSampledCurrents(sampledCurrents, grid);
  initSampledElectricFields(sampledEFields, grid);
  initSampledMagneticFields(sampledHFields, grid);

  const frequencies = [2.4e9, 5.8e9];
  const farfield = initFarfield({ frequencies, number_of_cells_from_outer_boundary: 13 }, grid);

  return {
    grid, coeffs, cpml,
    samplers: { sampledVoltages, sampledCurrents, sampledEFields, sampledHFields },
    sources: { voltageSources, currentSources: [], inductors: [], diodes: [] },
    farfield, ports, frequencies,
    numberOfTimeSteps, dt
  };
}

function postProcessResults(sim) {
  const { ports, samplers, farfield, frequencies, dt, grid } = sim;
  const maxFreq = 10e9, fStep = 20e6;
  const nFreqs = Math.floor(maxFreq / fStep);
  const freqArr = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) freqArr[i] = (i + 1) * fStep;

  computeObserverDFT(samplers.sampledVoltages, dt, 0, freqArr);
  computeObserverDFT(samplers.sampledCurrents, dt, -dt / 2, freqArr);
  postProcessSParameters({
    ports, sampledVoltages: samplers.sampledVoltages,
    sampledCurrents: samplers.sampledCurrents, dt, freqArr
  });

  let ffData = [];
  if (frequencies.length > 0) {
    const radPower = calcRadiatedPower(farfield, grid);
    const angles = new Float64Array(360);
    for (let i = 0; i < 360; i++) angles[i] = i * Math.PI / 180;

    const thXY = new Float64Array(360).fill(Math.PI / 2);
    const { dataTheta: dThXY, dataPhi: dPhXY } = calcDirectivity(farfield, grid, radPower, thXY, angles);

    const phXZ = new Float64Array(360);
    const thXZ = new Float64Array(360);
    for (let i = 0; i < 360; i++) {
      const th = angles[i];
      phXZ[i] = th > Math.PI ? Math.PI : 0;
      thXZ[i] = th > Math.PI ? 2 * Math.PI - th : th;
    }
    const { dataTheta: dThXZ, dataPhi: dPhXZ } = calcDirectivity(farfield, grid, radPower, thXZ, phXZ);

    for (let f = 0; f < frequencies.length; f++) {
      ffData.push({
        plane: 'xy', freq_GHz: frequencies[f] / 1e9,
        angles_deg: Array.from(angles).map(x => x * 180 / Math.PI),
        dataTheta_dB: Array.from(dThXY[f]).map(x => 10 * Math.log10(Math.max(x, 1e-10))),
        dataPhi_dB: Array.from(dPhXY[f]).map(x => 10 * Math.log10(Math.max(x, 1e-10)))
      });
      ffData.push({
        plane: 'xz', freq_GHz: frequencies[f] / 1e9,
        angles_deg: Array.from(angles).map(x => x * 180 / Math.PI),
        dataTheta_dB: Array.from(dThXZ[f]).map(x => 10 * Math.log10(Math.max(x, 1e-10))),
        dataPhi_dB: Array.from(dPhXZ[f]).map(x => 10 * Math.log10(Math.max(x, 1e-10)))
      });
    }
  }

  const numberOfTimeSteps = sim.numberOfTimeSteps;
  return {
    meta: {
      generated: new Date().toISOString(),
      numberOfTimeSteps, dt,
      frequencies_Hz: Array.from(freqArr)
    },
    sparams: ports.filter(p => p.is_source_port).map(p => ({
      label: 'S11',
      frequencies_GHz: Array.from(freqArr).map(f => f / 1e9),
      magnitude_dB: Array.from(p.S[0].mag_dB),
      phase_deg: Array.from(p.S[0].phase_deg)
    })),
    farfield: ffData,
    timeDomain: {
      voltages: samplers.sampledVoltages.map(v => {
        const time_ns = [];
        for (let i = 0; i < numberOfTimeSteps; i++) time_ns.push((i + 0.5) * dt * 1e9);
        return { label: v.label, time_ns, values: Array.from(v.sampled_value) };
      })
    }
  };
}

async function runSimulation() {
  if (simState.status === 'running' || simState.status === 'initializing') {
    return;
  }

  // Reset state
  simState = {
    status: 'initializing', step: 0, total: 0,
    elapsed: 0, percent: '0.0', error: null,
    results: null, voltageHistory: [],
  };
  sendSSE('status', { status: 'initializing' });

  let sim;
  try {
    console.log('Building simulation problem...');
    sim = buildSimulationProblem();
    simState.total = sim.numberOfTimeSteps;
    simState.status = 'running';
    sendSSE('status', { status: 'running', total: sim.numberOfTimeSteps });
  } catch (err) {
    simState.status = 'error';
    simState.error = err.message;
    sendSSE('error', { message: err.message });
    console.error('Init error:', err);
    return;
  }

  try {
    console.log('Running Multithreaded WASM SIMD Cluster with SSE streaming...');
    const wasmPath = path.join(__dirname, '../build/fdtd_kernels.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);

    const gen = runFDTDCluster(
      sim.grid, sim.coeffs, sim.cpml, sim.samplers, sim.sources,
      sim.farfield, wasmBuffer, 50
    );

    for await (const snap of gen) {
      simState.step = snap.step;
      simState.elapsed = snap.elapsed;
      simState.percent = snap.percent;

      // Accumulate voltage history for the chart
      if (snap.voltage !== null) {
        simState.voltageHistory.push({
          time_ns: snap.time_ns,
          value: snap.voltage
        });
      }

      sendSSE('progress', {
        step: snap.step,
        total: snap.total,
        elapsed: snap.elapsed,
        percent: snap.percent,
        voltage: snap.voltage,
        time_ns: snap.time_ns,
      });

      if (snap.step % 500 === 0 || snap.step === snap.total) {
        console.log(`Step ${snap.step}/${snap.total} (${snap.percent}%) — ${snap.elapsed.toFixed(1)}s`);
      }
    }

    // Post-process
    simState.status = 'postprocessing';
    sendSSE('status', { status: 'postprocessing' });
    console.log('Post-processing results...');
    simState.results = postProcessResults(sim);
    simState.status = 'done';
    
    // Instead of sending the massive JSON payload through SSE, which can buffer/truncate,
    // we send a small 'complete' flag. The client will then fetch from /results.
    sendSSE('complete', { ok: true });
    sendSSE('status', { status: 'done' });
    console.log('Simulation complete!');
  } catch (err) {
    simState.status = 'error';
    simState.error = err.message;
    sendSSE('error', { message: err.message, stack: err.stack });
    console.error('Simulation error:', err);
  }
}

// ── HTTP Server ──────────────────────────────────────────────────────────────

const PORT = 4000;

const server = http.createServer((req, res) => {
  // CORS headers (for dev — Next.js rewrites handle prod)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;

  // ── POST /simulate ─────────────────────────────────────────────────────
  if (pathname === '/simulate' && req.method === 'POST') {
    if (simState.status === 'running' || simState.status === 'initializing') {
      res.writeHead(409, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Simulation already running' }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, message: 'Simulation started' }));
    // Start simulation asynchronously
    runSimulation();
    return;
  }

  // ── GET /stream ────────────────────────────────────────────────────────
  if (pathname === '/stream' && req.method === 'GET') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',    // disable nginx buffering
    });
    res.write('\n'); // initial flush

    // Send current state immediately
    res.write(`event: status\ndata: ${JSON.stringify({
      status: simState.status,
      step: simState.step,
      total: simState.total,
      elapsed: simState.elapsed,
      percent: simState.percent,
    })}\n\n`);

    // If we already have voltage history, send it as a batch
    if (simState.voltageHistory.length > 0) {
      res.write(`event: voltage_history\ndata: ${JSON.stringify(simState.voltageHistory)}\n\n`);
    }

    // If already done, send results
    if (simState.status === 'done' && simState.results) {
      res.write(`event: complete\ndata: ${JSON.stringify(simState.results)}\n\n`);
    }

    sseClients.add(res);
    req.on('close', () => sseClients.delete(res));
    return;
  }

  // ── GET /status ────────────────────────────────────────────────────────
  if (pathname === '/status' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: simState.status,
      step: simState.step,
      total: simState.total,
      elapsed: simState.elapsed,
      percent: simState.percent,
      error: simState.error,
      hasResults: simState.results !== null,
    }));
    return;
  }

  // ── GET /results ───────────────────────────────────────────────────────
  if (pathname === '/results' && req.method === 'GET') {
    if (!simState.results) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'No results available' }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(simState.results));
    return;
  }

  // ── 404 ────────────────────────────────────────────────────────────────
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

server.listen(PORT, () => {
  console.log(`\n🔬 FDTD Simulation Server running at http://localhost:${PORT}`);
  console.log(`   POST /simulate  — start simulation`);
  console.log(`   GET  /stream    — SSE real-time stream`);
  console.log(`   GET  /status    — current state`);
  console.log(`   GET  /results   — final results\n`);
});
