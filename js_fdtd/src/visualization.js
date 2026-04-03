// visualization.js: Build Chart.js-compatible JSON output for
// polar radiation patterns and S-parameter frequency-domain plots.
// Mirrors MATLAB polar_plot_constant_phi.m / polar_plot_constant_theta.m
// and the S-parameter figure generation (lines 2618-2644).
//
// This module only produces data structures — no DOM or canvas access.
// The actual rendering happens in viewer.html using Chart.js.

/**
 * Build polar plot data for a constant-theta plane cut (phi varies 0..2pi).
 * Mirrors polar_plot_constant_theta.m:
 *   x = pattern * cos(phi),  y = pattern * sin(phi)
 *
 * @param {Float64Array} phi      - phi angles (radians), length nAngles
 * @param {Float64Array} pat1     - theta-component directivity, length nAngles
 * @param {Float64Array} pat2     - phi-component directivity, length nAngles
 * @param {boolean}      dB       - if true, convert via 10*log10
 * @param {string}       label1   - legend label for pat1
 * @param {string}       label2   - legend label for pat2
 * @returns {object} Chart.js scatter dataset pair
 */
export function buildPolarConstTheta(phi, pat1, pat2, dB, label1, label2) {
  const n = phi.length;
  const p1 = dB ? Array.from(pat1, v => 10 * Math.log10(Math.max(v, 1e-30))) : Array.from(pat1);
  const p2 = dB ? Array.from(pat2, v => 10 * Math.log10(Math.max(v, 1e-30))) : Array.from(pat2);

  const pts1 = [];
  const pts2 = [];
  for (let i = 0; i < n; i++) {
    pts1.push({ x: p1[i] * Math.cos(phi[i]), y: p1[i] * Math.sin(phi[i]) });
    pts2.push({ x: p2[i] * Math.cos(phi[i]), y: p2[i] * Math.sin(phi[i]) });
  }
  // Close the loop
  pts1.push(pts1[0]);
  pts2.push(pts2[0]);

  return {
    type: 'scatter',
    datasets: [
      { label: label1, data: pts1, borderColor: 'blue',  showLine: true, fill: false, pointRadius: 0 },
      { label: label2, data: pts2, borderColor: 'red',   showLine: true, fill: false, pointRadius: 0, borderDash: [5, 5] },
    ],
    maxVal: Math.max(...p1, ...p2),
  };
}

/**
 * Build polar plot data for a constant-phi plane cut (theta varies -pi..pi).
 * Mirrors polar_plot_constant_phi.m:
 *   x = -pattern * cos(theta + pi/2),  y = pattern * sin(theta + pi/2)
 *
 * @param {Float64Array} theta    - theta angles (radians), length nAngles
 * @param {Float64Array} pat1     - theta-component directivity
 * @param {Float64Array} pat2     - phi-component directivity
 * @param {boolean}      dB       - if true, convert via 10*log10
 * @param {string}       label1
 * @param {string}       label2
 * @returns {object} Chart.js scatter dataset pair
 */
export function buildPolarConstPhi(theta, pat1, pat2, dB, label1, label2) {
  const n = theta.length;
  const p1 = dB ? Array.from(pat1, v => 10 * Math.log10(Math.max(v, 1e-30))) : Array.from(pat1);
  const p2 = dB ? Array.from(pat2, v => 10 * Math.log10(Math.max(v, 1e-30))) : Array.from(pat2);

  const pts1 = [];
  const pts2 = [];
  for (let i = 0; i < n; i++) {
    const ang = theta[i] + Math.PI / 2;
    pts1.push({ x: -p1[i] * Math.cos(ang), y: p1[i] * Math.sin(ang) });
    pts2.push({ x: -p2[i] * Math.cos(ang), y: p2[i] * Math.sin(ang) });
  }
  pts1.push(pts1[0]);
  pts2.push(pts2[0]);

  return {
    type: 'scatter',
    datasets: [
      { label: label1, data: pts1, borderColor: 'blue', showLine: true, fill: false, pointRadius: 0 },
      { label: label2, data: pts2, borderColor: 'red',  showLine: true, fill: false, pointRadius: 0, borderDash: [5, 5] },
    ],
    maxVal: Math.max(...p1, ...p2),
  };
}

/**
 * Build S-parameter chart data (magnitude dB + phase) for all source ports.
 *
 * @param {object[]} ports        - ports array with .S matrix filled in
 * @param {Float64Array} freqArr  - frequency array (Hz)
 * @returns {object[]} array of chart data objects, one per S_ij pair
 */
export function buildSParamCharts(ports, freqArr) {
  const charts = [];
  const freqGHz = Array.from(freqArr, f => f * 1e-9);

  for (let ind = 0; ind < ports.length; ind++) {
    const srcPort = ports[ind];
    if (!srcPort.is_source_port) continue;

    for (let oind = 0; oind < ports.length; oind++) {
      const S = srcPort.S[oind];
      charts.push({
        label: `S${oind + 1}${ind + 1}`,
        frequencies_GHz: freqGHz,
        magnitude_dB: Array.from(S.mag_dB),
        phase_deg: Array.from(S.phase_deg),
      });
    }
  }
  return charts;
}

/**
 * Build time-domain sampler charts (for debugging/verification).
 *
 * @param {object[]} observers  - array with .sampled_value, .time, optional .label
 * @param {string}   title
 * @returns {object} chart data
 */
export function buildTimeDomainChart(observers, title) {
  return {
    title,
    series: observers.map((obs, i) => ({
      label: obs.label || `Observer ${i + 1}`,
      time_ns: Array.from(obs.time, t => t * 1e9),
      values: Array.from(obs.sampled_value),
    })),
  };
}

/**
 * Collect all visualization data into one results object for JSON export.
 *
 * @param {object} sim - simulation results object
 * @returns {object} results JSON
 */
export function buildResultsJSON(sim) {
  const { ports, freqArr, farfieldResults, sampledEFields, sampledVoltages } = sim;

  const result = {
    meta: {
      generated: new Date().toISOString(),
      numberOfTimeSteps: sim.numberOfTimeSteps,
      dt: sim.dt,
      frequencies_Hz: Array.from(freqArr),
    },
    sparams: buildSParamCharts(ports, freqArr),
    farfield: [],
    timeDomain: {
      voltages: sampledVoltages.map((v, i) => ({
        label: v.label || `V${i + 1}`,
        time_ns: Array.from(v.time, t => t * 1e9),
        values: Array.from(v.sampled_value),
      })),
      efields: sampledEFields.map((e, i) => ({
        label: e.label || `E${i + 1}`,
        time_ns: Array.from(e.time, t => t * 1e9),
        values: Array.from(e.sampled_value),
      })),
    },
  };

  // Farfield data per plane cut
  if (farfieldResults) {
    for (const cut of farfieldResults) {
      const nFreq = cut.dataTheta.length;
      for (let mi = 0; mi < nFreq; mi++) {
        const freqGHz = cut.frequencies[mi] * 1e-9;
        const pat1 = Array.from(cut.dataTheta[mi], v => isFinite(v) ? v : 0);
        const pat2 = Array.from(cut.dataPhi[mi],   v => isFinite(v) ? v : 0);
        const pat1dB = pat1.map(v => v > 0 ? 10 * Math.log10(v) : -60);
        const pat2dB = pat2.map(v => v > 0 ? 10 * Math.log10(v) : -60);
        result.farfield.push({
          plane: cut.plane,
          freq_GHz: freqGHz,
          angles_deg: Array.from(cut.angles_deg),
          dataTheta_dB: pat1dB,
          dataPhi_dB: pat2dB,
          dataTheta_linear: pat1,
          dataPhi_linear: pat2,
        });
      }
    }
  }

  return result;
}
