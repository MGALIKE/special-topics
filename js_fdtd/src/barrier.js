// barrier.js — High-performance synchronization for Node.js worker_threads
// Uses Atomics over a SharedArrayBuffer to create generation-based barriers.

/**
 * Creates a new barrier buffer for synchronization.
 * Requires 2 इंट-32 slots (8 bytes) per barrier.
 * Since we have 4 barrier phases per time step, we allocate 4 * 8 = 32 bytes.
 * @returns {Int32Array} The shared state array for Atomics.
 */
export function createBarrierState() {
  const sab = new SharedArrayBuffer(32); 
  return new Int32Array(sab);
}

/**
 * Wait at a barrier until all `numThreads` have arrived.
 * The last thread to arrive will wake all sleeping threads.
 *
 * @param {Int32Array} state - Shared typed array from createBarrierState()
 * @param {number} numThreads - Total number of worker threads
 * @param {number} phase - The barrier phase ID (0, 1, 2, or 3)
 */
export function syncBarrier(state, numThreads, phase = 0) {
  const offset = phase * 2;

  // Read the generation BEFORE registering our arrival. If we read it after the
  // increment, the last thread to arrive could bump the generation in between —
  // we'd then capture the NEW generation and Atomics.wait() on a value that is
  // already current, blocking forever (no further notify arrives for this round
  // because it already completed). That lost-wakeup is rare with 1–2 threads but
  // near-certain with many, and was the source of the multi-worker deadlock.
  const generation = Atomics.load(state, offset + 1);
  const arrived = Atomics.add(state, offset, 1) + 1;

  if (arrived === numThreads) {
    Atomics.store(state, offset, 0);
    Atomics.add(state, offset + 1, 1);
    Atomics.notify(state, offset + 1, numThreads);
  } else {
    // Loop until the generation actually advances. Atomics.wait returns
    // immediately ("not-equal") if the generation already moved past ours, and
    // re-checking absorbs any spurious wakeup.
    while (Atomics.load(state, offset + 1) === generation) {
      Atomics.wait(state, offset + 1, generation);
    }
  }
}

