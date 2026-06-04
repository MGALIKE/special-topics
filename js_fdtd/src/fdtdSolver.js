// fdtdSolver.js — Multi-threaded Pure JS FDTD backend orchestrater
import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import path from 'path';
import os from 'os';

import { createBarrierState } from './barrier.js';
import { updateMagneticCPML, updateElectricCPML } from './cpml.js';
import {
  captureMagneticFields, captureSampledCurrents,
  captureElectricFields, captureSampledVoltages,
} from './sampling.js';
import { accumulateFarfieldDFT } from './farfield.js';
import {
  injectVoltageSources, injectCurrentSources,
  updateInductors, updateDiodes,
} from './sources.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Original export logic for legacy support if needed (now stubs throwing errors)
export function runFDTD() { throw new Error('Use runFDTDCluster for multithreading'); }
export async function* runFDTDAsync() { throw new Error('Use runFDTDCluster for multithreading'); }

/**
 * ----------------------------------------------------------------------------
 * Multi-threaded FDTD Async Generator Cluster
 * ----------------------------------------------------------------------------
 * Spawns N Web Workers and yields control back to main thread per batch.
 */
export async function* runFDTDCluster(
  grid, coeffs, cpml, samplers, sources, ff, wasmBuffer, batchSize = 50
) {
  const numWorkers = Math.max(1, os.cpus().length - 1); // Save 1 core for Node
  
  // Create shared atomics barriers
  // index 0: worker arrived count, index 1: generation
  const barrierState = createBarrierState();
  
  // mainSignal: 0=workers working, 1=main thread do H, 2=main thread H done, 3=main do E, 4=main E done
  const mainSignalSab = new SharedArrayBuffer(4);
  const mainSignal = new Int32Array(mainSignalSab);

  // Setup the workers
  const workerArr = [];
  const workerPath = path.join(__dirname, 'fdtdWorker.js');

  const { sampledEFields, sampledHFields, sampledVoltages, sampledCurrents } = samplers;
  const { voltageSources, currentSources, inductors, diodes } = sources;
  const { numberOfTimeSteps } = grid;

  // We must strip typed arrays recursively to avoid massive cloning overhead where unnecessary.
  // Actually, V8 clones Float64Arrays natively in ~10ms.
  // We'll pass them in directly.
  
  for (let i = 0; i < numWorkers; i++) {
    const w = new Worker(workerPath, {
      workerData: {
        workerId: i,
        numWorkers,
        grid: { ...grid, wasmMemory: undefined, fields: undefined },
        coeffs: undefined, // pass heavy arrays later? No coeffs is not used in worker directly now? Wait!
        cpml,
        barrierSab: barrierState.buffer,
        mainSignalSab,
        wasmBuffer
      }
    });
    w.on('error', err => console.error(`Worker ${i} error:`, err));
    w.postMessage({ type: 'INIT', wasmMemory: grid.wasmMemory, fields: grid.fields });
    workerArr.push(w);
  }

  // Promise wrapper to wait for Atomics notifications safely off the main thread.
  // Since Main Thread CANNOT block using Atomics.wait(), we must poll!
  // Node provides no async Atomics.waitAsync natively on all older versions,
  // but it does in modern Node (>16)! We are on Node 22!
  const waitAsync = Atomics.waitAsync;

  const startTime = Date.now();

  try {
    for (let ts = 0; ts < numberOfTimeSteps; ts++) {
      
      // ── 1. Wait for H Bulk Updates & H CPML ──────────────────────────────────
      // Worker 0 will set mainSignal = 1 and notify. Loop on the ACTUAL current
      // value (not a hard-coded expected one) so we re-arm correctly on any
      // early/spurious wakeup and only proceed once the state is really 1.
      for (let v = Atomics.load(mainSignal, 0); v !== 1; v = Atomics.load(mainSignal, 0)) {
        const p = waitAsync(mainSignal, 0, v);
        if (p.async) await p.value;
      }

      // Main Thread Execution Block: H Captures
      if (sampledHFields.length > 0) captureMagneticFields(sampledHFields, grid.fields, grid, ts);
      if (sampledCurrents.length > 0) captureSampledCurrents(sampledCurrents, grid.fields, grid, ts);

      // Release workers to calculate E
      Atomics.store(mainSignal, 0, 2);
      Atomics.notify(mainSignal, 0, numWorkers);

      // ── 2. Wait for E Bulk Updates & E CPML ──────────────────────────────────
      // Worker 0 will set mainSignal = 3 and notify. Same race-free loop as phase 1.
      for (let v = Atomics.load(mainSignal, 0); v !== 3; v = Atomics.load(mainSignal, 0)) {
        const p = waitAsync(mainSignal, 0, v);
        if (p.async) await p.value;
      }

      // Main Thread Execution Block: Sources, E Captures, Farfield
      if (voltageSources.length > 0) injectVoltageSources(voltageSources, grid.fields, ts);
      if (currentSources.length > 0) injectCurrentSources(currentSources, grid.fields, ts);
      if (inductors.length > 0)      updateInductors(inductors, grid.fields);
      if (diodes.length > 0)         updateDiodes(diodes, grid.fields);

      if (sampledEFields.length > 0)   captureElectricFields(sampledEFields, grid.fields, grid, ts);
      if (sampledVoltages.length > 0)  captureSampledVoltages(sampledVoltages, grid.fields, grid, ts);
      if (ff.nFreq > 0) accumulateFarfieldDFT(ff, grid.fields, grid, ts);

      // Yield SSE real-time snapshot
      if ((ts + 1) % batchSize === 0 || ts === numberOfTimeSteps - 1) {
        const voltage = sampledVoltages.length > 0 ? sampledVoltages[0].sampled_value[ts] : null;
        const elapsed = (Date.now() - startTime) / 1000;
        yield {
          step: ts + 1,
          total: numberOfTimeSteps,
          elapsed,
          percent: ((ts + 1) / numberOfTimeSteps * 100).toFixed(1),
          voltage,
          time_ns: (ts + 0.5) * grid.dt * 1e9,
        };
        
        // --- CRITICAL YIELD ---
        // Force the Node Event Loop to process I/O so SSE network buffers flush to clients!
        await new Promise(r => setTimeout(r, 1));
      }

      // Release workers for the NEXT time step
      Atomics.store(mainSignal, 0, 0);
      Atomics.notify(mainSignal, 0, numWorkers);
    }
  } finally {
    // Terminate workers
    for (const w of workerArr) w.terminate();
  }
}
