// fdtdWorker.js — Multithreaded WebAssembly SIMD executor for the FDTD equations

import { workerData, parentPort } from 'worker_threads';
import { updateMagneticCPML, updateElectricCPML } from './cpml.js';
import { syncBarrier } from './barrier.js';

const {
  workerId,
  numWorkers,
  grid,
  cpml,
  barrierSab,
  mainSignalSab,
  wasmBuffer
} = workerData;

const barrierState = new Int32Array(barrierSab);
const mainSignal = new Int32Array(mainSignalSab);

const { nx, ny, nz, nxp1, nyp1, nzp1, numberOfTimeSteps, pointers } = grid;

function getPartition(size, id, total) {
  const chunk = Math.ceil(size / total);
  const start = Math.min(id * chunk, size);
  const end = Math.min(start + chunk, size);
  return { start, end };
}

const p_nx = getPartition(nx, workerId, numWorkers);
const p_nxp1 = getPartition(nxp1, workerId, numWorkers);

// Wait for postMessage carrying the WebAssembly.Memory object
parentPort.once('message', async (msg) => {
  if (msg.type !== 'INIT') return;

  const { wasmMemory, fields } = msg;

  // Instantiate the WebAssembly hardware SIMD logic using the Grid's Shared Memory Space!
  const wasmModule = await WebAssembly.compile(wasmBuffer);
  
  if (!(wasmMemory instanceof WebAssembly.Memory)) {
    console.error(`Worker ${workerId} received invalid wasmMemory:`, wasmMemory);
  }

  const wasmInstance = await WebAssembly.instantiate(wasmModule, {
    env: {
      memory: wasmMemory,
      abort: function(msg, file, line, colm) {
        console.error(`abort: ${msg} at ${file}:${line}:${colm}`);
      }
    }
  });
  
  const wasm = wasmInstance.exports;

  // Enter Physics Calculation Time Loop!
  for (let ts = 0; ts < numberOfTimeSteps; ts++) {

    // ────────────────────────────────────────────────────────────────────────────
    // 1. WebAssembly Hardware SIMD H Bulk Update
    // ────────────────────────────────────────────────────────────────────────────
    wasm.updateH(
      nx, ny, nz, nxp1, nyp1, nzp1,
      p_nxp1.start, p_nxp1.end,
      p_nx.start, p_nx.end,
      pointers.Hx, pointers.Hy, pointers.Hz,
      pointers.Ex, pointers.Ey, pointers.Ez,
      pointers.Chxh, pointers.Chxey, pointers.Chxez,
      pointers.Chyh, pointers.Chyez, pointers.Chyex,
      pointers.Chzh, pointers.Chzex, pointers.Chzey
    );

    syncBarrier(barrierState, numWorkers, 0);

    // ────────────────────────────────────────────────────────────────────────────
    // 1b. Boundary & CPU Synchronization
    // ────────────────────────────────────────────────────────────────────────────
    if (workerId === 0) {
      updateMagneticCPML(cpml, fields, grid);
      Atomics.store(mainSignal, 0, 1);
      Atomics.notify(mainSignal, 0); // Wake main thread to handle sources/capture
    }

    Atomics.wait(mainSignal, 0, 1);
    syncBarrier(barrierState, numWorkers, 1);

    // ────────────────────────────────────────────────────────────────────────────
    // 2. WebAssembly Hardware SIMD E Bulk Update
    // ────────────────────────────────────────────────────────────────────────────
    if (ts === 50) console.log(`Worker ${workerId} entering updateE`);
    wasm.updateE(
      nx, ny, nz, nxp1, nyp1, nzp1,
      p_nx.start, p_nx.end,
      p_nxp1.start, p_nxp1.end,
      pointers.Ex, pointers.Ey, pointers.Ez,
      pointers.Hx, pointers.Hy, pointers.Hz,
      pointers.Cexe, pointers.Cexhz, pointers.Cexhy,
      pointers.Ceye, pointers.Ceyhx, pointers.Ceyhz,
      pointers.Ceze, pointers.Cezhy, pointers.Cezhx
    );
    if (ts === 50) console.log(`Worker ${workerId} finished updateE`);

    syncBarrier(barrierState, numWorkers, 2);

    // ────────────────────────────────────────────────────────────────────────────
    // 2b. Boundary & Main-thread Synchronization
    // ────────────────────────────────────────────────────────────────────────────
    if (workerId === 0) {
      updateElectricCPML(cpml, fields, grid);
      Atomics.store(mainSignal, 0, 3);
      Atomics.notify(mainSignal, 0);
    }

    if (ts === 50) console.log(`Worker ${workerId} waiting for mainSignal 3`);
    Atomics.wait(mainSignal, 0, 3);
    if (ts === 50) console.log(`Worker ${workerId} hit barrier 3`);
    syncBarrier(barrierState, numWorkers, 3);

  }
  
  parentPort.postMessage({ type: 'DONE' });
});
