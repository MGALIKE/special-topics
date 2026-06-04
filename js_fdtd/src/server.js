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

import { calcRadiatedPower, calcDirectivity } from './farfield.js';
import { computeObserverDFT } from './dft.js';
import { postProcessSParameters } from './sparameters.js';
import { selectBackend, listBackends } from './backends/registry.js';
import { loadScenario, listScenarios, buildProblemFromScenario } from './scenario.js';

// Scenario run when /simulate is called without an explicit { scenario } body.
// Override with the FDTD_SCENARIO env var.
const DEFAULT_SCENARIO = process.env.FDTD_SCENARIO || 'ifa-dualband-baseline';

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
  requestedBackend: 'auto', // what the client asked for
  backend: null,        // what actually ran (resolved by the registry)
  scenario: DEFAULT_SCENARIO, // which scenario JSON is loaded
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

// Build the simulation problem from a named scenario JSON (js_fdtd/scenarios/).
// The geometry/materials/sources/step-count all live in the scenario file now;
// see src/scenario.js. FDTD_STEPS still overrides the step count for experiments.
function buildSimulationProblem(scenarioName = DEFAULT_SCENARIO) {
  const scenario = loadScenario(scenarioName);
  console.log(`Scenario: ${scenario.name || scenarioName}`);
  return buildProblemFromScenario(scenario);
}

function postProcessResults(sim) {
  const { ports, samplers, farfield, frequencies, dt, grid } = sim;
  const maxFreq = sim.dft?.maxFreq ?? 10e9;
  const fStep = sim.dft?.step ?? 20e6;
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
      scenario: sim.meta?.name ?? null,
      description: sim.meta?.description ?? null,
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

async function runSimulation(requestedBackend = 'auto', scenarioName = DEFAULT_SCENARIO) {
  if (simState.status === 'running' || simState.status === 'initializing') {
    return;
  }

  // Reset state
  simState = {
    status: 'initializing', step: 0, total: 0,
    elapsed: 0, percent: '0.0', error: null,
    results: null, voltageHistory: [],
    requestedBackend, backend: null, scenario: scenarioName,
  };
  sendSSE('status', { status: 'initializing', requestedBackend, scenario: scenarioName });

  let sim;
  try {
    console.log(`Building simulation problem from scenario "${scenarioName}"...`);
    sim = buildSimulationProblem(scenarioName);
    simState.total = sim.numberOfTimeSteps;
    simState.status = 'running';
    sendSSE('status', { status: 'running', total: sim.numberOfTimeSteps, scenario: scenarioName });
  } catch (err) {
    simState.status = 'error';
    simState.error = err.message;
    sendSSE('error', { message: err.message });
    console.error('Init error:', err);
    return;
  }

  try {
    const wasmPath = path.join(__dirname, '../build/fdtd_kernels.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);

    // Resolve the requested compute backend (falls back to wasm-cpu).
    const { name: backendName, backend } = await selectBackend(requestedBackend);
    simState.backend = backendName;
    console.log(`Running FDTD on backend "${backendName}" (requested "${requestedBackend}") with SSE streaming...`);
    sendSSE('status', { status: 'running', total: sim.numberOfTimeSteps, backend: backendName, requestedBackend });

    const problem = {
      grid: sim.grid, coeffs: sim.coeffs, cpml: sim.cpml,
      samplers: sim.samplers, sources: sim.sources, ff: sim.farfield,
      wasmBuffer, options: { batchSize: 50 },
    };

    const gen = backend.run(problem);

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
        backend: snap.backend ?? simState.backend,
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
    // Read the requested backend from the JSON body: { backend: 'auto'|'webgpu'|... }
    let body = '';
    req.on('data', chunk => { body += chunk; if (body.length > 1e6) req.destroy(); });
    req.on('end', () => {
      let requestedBackend = 'auto';
      let scenario = DEFAULT_SCENARIO;
      try {
        if (body) {
          const parsed = JSON.parse(body);
          requestedBackend = parsed.backend || 'auto';
          scenario = parsed.scenario || DEFAULT_SCENARIO;
        }
      } catch { /* ignore */ }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, message: 'Simulation started', requestedBackend, scenario }));
      // Start simulation asynchronously
      runSimulation(requestedBackend, scenario);
    });
    return;
  }

  // ── GET /scenarios ─────────────────────────────────────────────────────
  // Lists the example scenario JSON files the client can run. Each entry
  // carries its name/description and step count so a picker can show them.
  if (pathname === '/scenarios' && req.method === 'GET') {
    try {
      const scenarios = listScenarios().map(name => {
        const s = loadScenario(name);
        return {
          name,
          title: s.name || name,
          description: s.description || '',
          numberOfTimeSteps: s.numberOfTimeSteps ?? null,
        };
      });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ scenarios, default: DEFAULT_SCENARIO }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // ── GET /backends ──────────────────────────────────────────────────────
  // Real availability on THIS engine process — what will actually run when you
  // hit Run. wasm-cpu is always present; webgpu (Node/Dawn) and cuda depend on
  // optional deps being installed/built here.
  if (pathname === '/backends' && req.method === 'GET') {
    listBackends().then(list => {
      const backends = list.map(b => ({
        name: b.name,
        available: b.available,
        detail: b.detail,
        location: 'server',
      }));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ backends }));
    }).catch(err => {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    });
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
      backend: simState.backend,
      requestedBackend: simState.requestedBackend,
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
      backend: simState.backend,
      requestedBackend: simState.requestedBackend,
      scenario: simState.scenario,
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
  console.log(`   POST /simulate  — start simulation  (body: { backend, scenario })`);
  console.log(`   GET  /scenarios — list example scenarios`);
  console.log(`   GET  /stream    — SSE real-time stream`);
  console.log(`   GET  /status    — current state`);
  console.log(`   GET  /results   — final results`);
  console.log(`   Scenarios: ${listScenarios().join(', ')}`);
  console.log(`   Default scenario: ${DEFAULT_SCENARIO}\n`);
});
