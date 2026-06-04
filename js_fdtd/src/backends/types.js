// Shared JSDoc typedefs for backends. No runtime code — import for editor types.

/**
 * @typedef {Object} Problem
 * @property {object} grid       - from buildGrid()
 * @property {object} coeffs     - 18 coefficient Float64Arrays
 * @property {object} cpml       - from initCPML()
 * @property {object} samplers   - { sampledEFields, sampledHFields, sampledVoltages, sampledCurrents }
 * @property {object} sources    - { voltageSources, currentSources, inductors, diodes }
 * @property {object} ff         - from initFarfield()
 * @property {Uint8Array} [wasmBuffer]
 * @property {object} [options]  - { batchSize }
 */

/**
 * @typedef {Object} ProgressSnapshot
 * @property {number} step
 * @property {number} total
 * @property {number} elapsed   - seconds
 * @property {string} percent
 * @property {number|null} voltage
 * @property {number} time_ns
 * @property {string} backend
 */

/**
 * @typedef {Object} Backend
 * @property {string} name
 * @property {() => Promise<boolean>} isAvailable
 * @property {(problem: Problem) => AsyncGenerator<ProgressSnapshot>} run
 * @property {() => object} [meta]
 */

export {};
