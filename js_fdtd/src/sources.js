// Source waveforms, lumped element initialization, and per-step source injection.
// Mirrors MATLAB lines 682-814 (waveform init), 767-813 (source init),
// and 2194-2291 (per-step injection + diode update).
//
// solve_diode_equation: Newton-Raphson for f(E) = A·exp(B·E) + E + C = 0

/**
 * Initialize all waveform arrays.
 * @param {object} waveforms - waveform definitions (sinusoidal, gaussian, etc.)
 * @param {number} numberOfTimeSteps
 * @param {number} dt
 * @param {number} numberOfCellsPerWavelength - default nc for gaussian
 * @param {number[]} cellSizes - [dx, dy, dz]
 * @returns Modified waveforms object with .waveform Float64Array on each entry
 */
export function initWaveforms(waveforms, numberOfTimeSteps, dt, numberOfCellsPerWavelength, cellSizes) {
  const { c } = { c: 2.99792458e8 };
  const maxCellSize = Math.max(...cellSizes);

  // time array: MATLAB time = ([1:N]-0.5)*dt → JS time[n] = (n+0.5)*dt, n=0..N-1
  const time = new Float64Array(numberOfTimeSteps);
  for (let n = 0; n < numberOfTimeSteps; n++) time[n] = (n + 0.5) * dt;

  if (waveforms.sinusoidal) {
    for (const w of waveforms.sinusoidal) {
      w.waveform = new Float64Array(numberOfTimeSteps);
      for (let n = 0; n < numberOfTimeSteps; n++) {
        w.waveform[n] = Math.sin(2 * Math.PI * w.frequency * time[n]);
      }
      w.t_0 = 0;
    }
  }

  if (waveforms.unit_step) {
    for (const w of waveforms.unit_step) {
      w.waveform = new Float64Array(numberOfTimeSteps).fill(1);
      const start = w.start_time_step - 1; // 0-based
      for (let n = 0; n < start; n++) w.waveform[n] = 0;
      w.t_0 = 0;
    }
  }

  if (waveforms.gaussian) {
    for (const w of waveforms.gaussian) {
      const nc = w.number_of_cells_per_wavelength === 0
        ? numberOfCellsPerWavelength
        : w.number_of_cells_per_wavelength;
      w.maximum_frequency = c / (nc * maxCellSize);
      const tau = (nc * maxCellSize) / (2 * c);
      w.tau = tau;
      w.t_0 = 4.5 * tau;
      w.waveform = new Float64Array(numberOfTimeSteps);
      for (let n = 0; n < numberOfTimeSteps; n++) {
        const arg = (time[n] - w.t_0) / tau;
        w.waveform[n] = Math.exp(-arg * arg);
      }
    }
  }

  if (waveforms.derivative_gaussian) {
    for (const w of waveforms.derivative_gaussian) {
      const nc = w.number_of_cells_per_wavelength === 0
        ? numberOfCellsPerWavelength
        : w.number_of_cells_per_wavelength;
      w.maximum_frequency = c / (nc * maxCellSize);
      const tau = (nc * maxCellSize) / (2 * c);
      w.tau = tau;
      w.t_0 = 4.5 * tau;
      const scale = -Math.sqrt(2 * Math.E) / tau;
      w.waveform = new Float64Array(numberOfTimeSteps);
      for (let n = 0; n < numberOfTimeSteps; n++) {
        const dt2 = time[n] - w.t_0;
        const arg = dt2 / tau;
        w.waveform[n] = scale * dt2 * Math.exp(-arg * arg);
      }
    }
  }

  if (waveforms.cosine_modulated_gaussian) {
    for (const w of waveforms.cosine_modulated_gaussian) {
      const tau = 0.966 / w.bandwidth;
      w.tau = tau;
      w.t_0 = 4.5 * tau;
      w.waveform = new Float64Array(numberOfTimeSteps);
      for (let n = 0; n < numberOfTimeSteps; n++) {
        const dt2 = time[n] - w.t_0;
        const arg = dt2 / tau;
        w.waveform[n] = Math.cos(2 * Math.PI * w.modulation_frequency * dt2) * Math.exp(-arg * arg);
      }
    }
  }

  return waveforms;
}

/**
 * Initialize voltage source working arrays.
 * Computes per-element: resistance_per_component, voltage_per_e_field, waveform.
 * Mirrors MATLAB lines 768-813.
 * @param {Array} voltageSources
 * @param {object} waveforms
 * @param {object} grid
 */
export function initVoltageSources(voltageSources, waveforms, grid) {
  const { dx, dy, dz, min_x, min_y, min_z } = grid;
  for (const vs of voltageSources) {
    const is = Math.round((vs.min_x - min_x) / dx) + 1;
    const js = Math.round((vs.min_y - min_y) / dy) + 1;
    const ks = Math.round((vs.min_z - min_z) / dz) + 1;
    const ie = Math.round((vs.max_x - min_x) / dx) + 1;
    const je = Math.round((vs.max_y - min_y) / dy) + 1;
    const ke = Math.round((vs.max_z - min_z) / dz) + 1;
    vs.is = is; vs.js = js; vs.ks = ks;
    vs.ie = ie; vs.je = je; vs.ke = ke;

    const dir = vs.direction[0];
    let n_fields, r_magnitude_factor;
    if (dir === 'x') {
      n_fields = ie - is;
      r_magnitude_factor = (1 + je - js) * (1 + ke - ks) / (ie - is);
    } else if (dir === 'y') {
      n_fields = je - js;
      r_magnitude_factor = (1 + ie - is) * (1 + ke - ks) / (je - js);
    } else {
      n_fields = ke - ks;
      r_magnitude_factor = (1 + ie - is) * (1 + je - js) / (ke - ks);
    }

    const v_sign = vs.direction[1] === 'n' ? -1 : 1;
    const v_magnitude_factor = v_sign * vs.magnitude / n_fields;
    vs.resistance_per_component = r_magnitude_factor * vs.resistance;

    // Get waveform array
    const wfType = vs.waveform_type;
    const wfIdx  = vs.waveform_index - 1; // 0-based
    const baseWaveform = waveforms[wfType][wfIdx].waveform;

    vs.voltage_per_e_field = new Float64Array(baseWaveform.length);
    vs.waveform = new Float64Array(baseWaveform.length);
    for (let n = 0; n < baseWaveform.length; n++) {
      vs.voltage_per_e_field[n] = v_magnitude_factor * baseWaveform[n];
      vs.waveform[n] = v_magnitude_factor * n_fields * baseWaveform[n];
    }
  }
}

/**
 * Initialize current source working arrays.
 * Mirrors MATLAB lines 815-861.
 */
export function initCurrentSources(currentSources, waveforms, grid) {
  const { dx, dy, dz, min_x, min_y, min_z } = grid;
  for (const cs of currentSources) {
    const is = Math.round((cs.min_x - min_x) / dx) + 1;
    const js = Math.round((cs.min_y - min_y) / dy) + 1;
    const ks = Math.round((cs.min_z - min_z) / dz) + 1;
    const ie = Math.round((cs.max_x - min_x) / dx) + 1;
    const je = Math.round((cs.max_y - min_y) / dy) + 1;
    const ke = Math.round((cs.max_z - min_z) / dz) + 1;
    cs.is = is; cs.js = js; cs.ks = ks;
    cs.ie = ie; cs.je = je; cs.ke = ke;

    const dir = cs.direction[0];
    let n_fields, r_magnitude_factor;
    if (dir === 'x') {
      n_fields = (1 + je - js) * (1 + ke - ks);
      r_magnitude_factor = (1 + je - js) * (1 + ke - ks) / (ie - is);
    } else if (dir === 'y') {
      n_fields = (1 + ie - is) * (1 + ke - ks);
      r_magnitude_factor = (1 + ie - is) * (1 + ke - ks) / (je - js);
    } else {
      n_fields = (1 + ie - is) * (1 + je - js);
      r_magnitude_factor = (1 + ie - is) * (1 + je - js) / (ke - ks);
    }

    const i_sign = cs.direction[1] === 'n' ? -1 : 1;
    const i_magnitude_factor = i_sign * cs.magnitude / n_fields;
    cs.resistance_per_component = r_magnitude_factor * cs.resistance;

    const wfType = cs.waveform_type;
    const wfIdx  = cs.waveform_index - 1;
    const baseWaveform = waveforms[wfType][wfIdx].waveform;

    cs.current_per_e_field = new Float64Array(baseWaveform.length);
    cs.waveform = new Float64Array(baseWaveform.length);
    for (let n = 0; n < baseWaveform.length; n++) {
      cs.current_per_e_field[n] = i_magnitude_factor * baseWaveform[n];
      cs.waveform[n] = i_magnitude_factor * n_fields * baseWaveform[n];
    }
  }
}

/**
 * Initialize resistor per-component resistance. Mirrors MATLAB lines 863-887.
 */
export function initResistors(resistors, grid) {
  const { dx, dy, dz, min_x, min_y, min_z } = grid;
  for (const res of resistors) {
    const is = Math.round((res.min_x - min_x) / dx) + 1;
    const js = Math.round((res.min_y - min_y) / dy) + 1;
    const ks = Math.round((res.min_z - min_z) / dz) + 1;
    const ie = Math.round((res.max_x - min_x) / dx) + 1;
    const je = Math.round((res.max_y - min_y) / dy) + 1;
    const ke = Math.round((res.max_z - min_z) / dz) + 1;
    res.is = is; res.js = js; res.ks = ks;
    res.ie = ie; res.je = je; res.ke = ke;
    const dir = res.direction[0];
    let r;
    if (dir === 'x')      r = (1 + je - js) * (1 + ke - ks) / (ie - is);
    else if (dir === 'y') r = (1 + ie - is) * (1 + ke - ks) / (je - js);
    else                   r = (1 + ie - is) * (1 + je - js) / (ke - ks);
    res.resistance_per_component = r * res.resistance;
  }
}

/**
 * Initialize inductor per-component inductance. Mirrors MATLAB lines 889-913.
 */
export function initInductors(inductors, grid) {
  const { dx, dy, dz, min_x, min_y, min_z } = grid;
  for (const ind of inductors) {
    const is = Math.round((ind.min_x - min_x) / dx) + 1;
    const js = Math.round((ind.min_y - min_y) / dy) + 1;
    const ks = Math.round((ind.min_z - min_z) / dz) + 1;
    const ie = Math.round((ind.max_x - min_x) / dx) + 1;
    const je = Math.round((ind.max_y - min_y) / dy) + 1;
    const ke = Math.round((ind.max_z - min_z) / dz) + 1;
    ind.is = is; ind.js = js; ind.ks = ks;
    ind.ie = ie; ind.je = je; ind.ke = ke;
    const dir = ind.direction[0];
    let l;
    if (dir === 'x')      l = (1 + je - js) * (1 + ke - ks) / (ie - is);
    else if (dir === 'y') l = (1 + ie - is) * (1 + ke - ks) / (je - js);
    else                   l = (1 + ie - is) * (1 + je - js) / (ke - ks);
    ind.inductance_per_component = l * ind.inductance;
  }
}

/**
 * Initialize capacitor per-component capacitance. Mirrors MATLAB lines 915-942.
 */
export function initCapacitors(capacitors, grid) {
  const { dx, dy, dz, min_x, min_y, min_z } = grid;
  for (const cap of capacitors) {
    const is = Math.round((cap.min_x - min_x) / dx) + 1;
    const js = Math.round((cap.min_y - min_y) / dy) + 1;
    const ks = Math.round((cap.min_z - min_z) / dz) + 1;
    const ie = Math.round((cap.max_x - min_x) / dx) + 1;
    const je = Math.round((cap.max_y - min_y) / dy) + 1;
    const ke = Math.round((cap.max_z - min_z) / dz) + 1;
    cap.is = is; cap.js = js; cap.ks = ks;
    cap.ie = ie; cap.je = je; cap.ke = ke;
    const dir = cap.direction[0];
    let c;
    if (dir === 'x')      c = (ie - is) / ((1 + je - js) * (1 + ke - ks));
    else if (dir === 'y') c = (je - js) / ((1 + ie - is) * (1 + ke - ks));
    else                   c = (ke - ks) / ((1 + ie - is) * (1 + je - js));
    cap.capacitance_per_component = c * cap.capacitance;
  }
}

// ─── Per-step: inject sources and update inductors/diodes ─────────────────────

/**
 * Inject voltage sources into E-field arrays.
 * Mirrors MATLAB lines 2198-2211.
 * @param {Array} voltageSources
 * @param {object} fields - {Ex, Ey, Ez}
 * @param {number} timeStep - 0-based (MATLAB 1-based, so use timeStep+1 for waveform index)
 */
export function injectVoltageSources(voltageSources, fields, timeStep) {
  const { Ex, Ey, Ez } = fields;
  for (const vs of voltageSources) {
    const { field_indices, direction, Cexs, Ceys, Cezs, voltage_per_e_field } = vs;
    const v = voltage_per_e_field[timeStep]; // MATLAB: voltage_per_e_field(time_step)
    const dir = direction[0];
    if (dir === 'x') {
      for (let n = 0; n < field_indices.length; n++)
        Ex[field_indices[n]] += Cexs[n] * v;
    } else if (dir === 'y') {
      for (let n = 0; n < field_indices.length; n++)
        Ey[field_indices[n]] += Ceys[n] * v;
    } else {
      for (let n = 0; n < field_indices.length; n++)
        Ez[field_indices[n]] += Cezs[n] * v;
    }
  }
}

/**
 * Inject current sources into E-field arrays.
 * Mirrors MATLAB lines 2218-2231.
 */
export function injectCurrentSources(currentSources, fields, timeStep) {
  const { Ex, Ey, Ez } = fields;
  for (const cs of currentSources) {
    const { field_indices, direction, Cexs, Ceys, Cezs, current_per_e_field } = cs;
    const v = current_per_e_field[timeStep];
    const dir = direction[0];
    if (dir === 'x') {
      for (let n = 0; n < field_indices.length; n++)
        Ex[field_indices[n]] += Cexs[n] * v;
    } else if (dir === 'y') {
      for (let n = 0; n < field_indices.length; n++)
        Ey[field_indices[n]] += Ceys[n] * v;
    } else {
      for (let n = 0; n < field_indices.length; n++)
        Ez[field_indices[n]] += Cezs[n] * v;
    }
  }
}

/**
 * Update inductors (implicit current integration).
 * Mirrors MATLAB lines 2239-2258.
 * Update: E += Cexj * Jix; then Jix += Cjex * E_new
 */
export function updateInductors(inductors, fields) {
  const { Ex, Ey, Ez } = fields;
  for (const ind of inductors) {
    const { field_indices, direction } = ind;
    const dir = direction[0];
    if (dir === 'x') {
      for (let n = 0; n < field_indices.length; n++) {
        const idx = field_indices[n];
        Ex[idx] += ind.Cexj[n] * ind.Jix[n];
        ind.Jix[n] += ind.Cjex * Ex[idx];
      }
    } else if (dir === 'y') {
      for (let n = 0; n < field_indices.length; n++) {
        const idx = field_indices[n];
        Ey[idx] += ind.Ceyj[n] * ind.Jiy[n];
        ind.Jiy[n] += ind.Cjey * Ey[idx];
      }
    } else {
      for (let n = 0; n < field_indices.length; n++) {
        const idx = field_indices[n];
        Ez[idx] += ind.Cezj[n] * ind.Jiz[n];
        ind.Jiz[n] += ind.Cjez * Ez[idx];
      }
    }
  }
}

/**
 * Update diode elements using Newton-Raphson solver.
 * Mirrors MATLAB lines 2265-2291.
 * Solves f(E) = A·exp(B·E) + E + C = 0
 */
export function updateDiodes(diodes, fields) {
  const { Ex, Ey, Ez } = fields;
  for (const d of diodes) {
    const { field_indices, direction, B } = d;
    const idx = field_indices[0];
    const dir = direction[0];

    if (dir === 'x') {
      const E_prev = d.Exn;
      const C = -Ex[idx] + d.Cexd;
      const A = -d.Cexd * Math.exp(B * E_prev);
      const E_new = solveDiodeEquation(A, B, C, E_prev);
      Ex[idx] = E_new;
      d.Exn = E_new;
    } else if (dir === 'y') {
      const E_prev = d.Eyn;
      const C = -Ey[idx] + d.Ceyd;
      const A = -d.Ceyd * Math.exp(B * E_prev);
      const E_new = solveDiodeEquation(A, B, C, E_prev);
      Ey[idx] = E_new;
      d.Eyn = E_new;
    } else {
      const E_prev = d.Ezn;
      const C = -Ez[idx] + d.Cezd;
      const A = -d.Cezd * Math.exp(B * E_prev);
      const E_new = solveDiodeEquation(A, B, C, E_prev);
      Ez[idx] = E_new;
      d.Ezn = E_new;
    }
  }
}

/**
 * Newton-Raphson solver for the Shockley diode equation:
 *   f(E) = A·exp(B·E) + E + C = 0
 *   f'(E) = A·B·exp(B·E) + 1
 *
 * @param {number} A
 * @param {number} B
 * @param {number} C
 * @param {number} E0 - initial guess (previous E value)
 * @returns {number} E - solution
 */
function solveDiodeEquation(A, B, C, E0) {
  let E = E0;
  const tol = 1e-10;
  const maxIter = 20;
  for (let iter = 0; iter < maxIter; iter++) {
    const expBE = Math.exp(B * E);
    const f  = A * expBE + E + C;
    const df = A * B * expBE + 1;
    const dE = f / df;
    E -= dE;
    if (Math.abs(dE) < tol) break;
  }
  return E;
}
