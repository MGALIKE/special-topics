// S-parameters: compute incident/reflected power waves and S-matrix.
// Mirrors MATLAB lines 2595-2644.
//
// Port model: each port has a voltage observer (sampled_voltage_index),
// a current observer (sampled_current_index), and a reference impedance Z.
//
// Wave variables (power-normalized):
//   a = 0.5*(V + Z*I) / sqrt(Re(Z))   (incident wave)
//   b = 0.5*(V - conj(Z)*I) / sqrt(Re(Z))  (reflected wave)
//
// S-parameter: S(out,in) = b(out) / a(in)   [in = source port]

import { timeToFreqDomain } from './dft.js';

/**
 * Compute frequency-domain power waves for all ports.
 * Modifies ports in-place, adding .a and .b complex arrays.
 *
 * @param {object[]} ports           - array of port objects
 * @param {object[]} sampledVoltages - voltage observer objects (already have .frequency_domain_value)
 * @param {object[]} sampledCurrents - current observer objects (already have .frequency_domain_value)
 */
export function computePortWaves(ports, sampledVoltages, sampledCurrents) {
  for (const port of ports) {
    const svi = port.sampled_voltage_index - 1;  // 0-based
    const sci = port.sampled_current_index - 1;
    const Z = port.impedance;
    const Zr = typeof Z === 'number' ? Z : Z.re;  // real impedance
    const sqrtRe = Math.sqrt(Zr);

    const V = sampledVoltages[svi].frequency_domain_value;
    const I = sampledCurrents[sci].frequency_domain_value;
    const M = V.re.length;

    const a = { re: new Float64Array(M), im: new Float64Array(M) };
    const b = { re: new Float64Array(M), im: new Float64Array(M) };

    for (let fi = 0; fi < M; fi++) {
      // a = 0.5*(V + Z*I) / sqrt(Re(Z))
      // Z is real, so Z*I = Zr * I
      const aRe = 0.5 * (V.re[fi] + Zr * I.re[fi]) / sqrtRe;
      const aIm = 0.5 * (V.im[fi] + Zr * I.im[fi]) / sqrtRe;
      // b = 0.5*(V - conj(Z)*I) / sqrt(Re(Z))
      // conj(Z) = Zr (real impedance), so same as a formula but with minus
      const bRe = 0.5 * (V.re[fi] - Zr * I.re[fi]) / sqrtRe;
      const bIm = 0.5 * (V.im[fi] - Zr * I.im[fi]) / sqrtRe;

      a.re[fi] = aRe; a.im[fi] = aIm;
      b.re[fi] = bRe; b.im[fi] = bIm;
    }

    port.a = a;
    port.b = b;
    port.frequencies = sampledVoltages[svi].frequencies;
  }
}

/**
 * Compute S-matrix from port power waves.
 * For each source port, compute S(out, in) = b(out) / a(in).
 * Modifies ports in-place, adding .S[oind] = { values: { re, im, mag_dB, phase_deg } }.
 *
 * @param {object[]} ports - ports array (after computePortWaves)
 */
export function computeSMatrix(ports) {
  for (let ind = 0; ind < ports.length; ind++) {
    const srcPort = ports[ind];
    if (!srcPort.is_source_port) continue;

    srcPort.S = [];
    const aRe = srcPort.a.re;
    const aIm = srcPort.a.im;
    const M = aRe.length;

    for (let oind = 0; oind < ports.length; oind++) {
      const bRe = ports[oind].b.re;
      const bIm = ports[oind].b.im;

      const Sre = new Float64Array(M);
      const Sim = new Float64Array(M);
      const Smag_dB = new Float64Array(M);
      const Sphase = new Float64Array(M);

      for (let fi = 0; fi < M; fi++) {
        // S = b/a = (bRe + j*bIm) / (aRe + j*aIm)
        const denom = aRe[fi] * aRe[fi] + aIm[fi] * aIm[fi];
        if (denom < 1e-300) {
          Sre[fi] = 0; Sim[fi] = 0;
        } else {
          Sre[fi] = (bRe[fi] * aRe[fi] + bIm[fi] * aIm[fi]) / denom;
          Sim[fi] = (bIm[fi] * aRe[fi] - bRe[fi] * aIm[fi]) / denom;
        }
        const mag = Math.sqrt(Sre[fi]*Sre[fi] + Sim[fi]*Sim[fi]);
        Smag_dB[fi] = mag > 0 ? 20 * Math.log10(mag) : -Infinity;
        Sphase[fi] = Math.atan2(Sim[fi], Sre[fi]) * 180 / Math.PI;
      }

      srcPort.S[oind] = { re: Sre, im: Sim, mag_dB: Smag_dB, phase_deg: Sphase };
    }
  }
}

/**
 * Run all post-loop S-parameter processing.
 * - Computes DFT for voltages, currents, and source waveforms
 * - Computes port waves and S-matrix
 *
 * @param {object} params - { ports, sampledVoltages, sampledCurrents, voltageSources, currentSources, dt, freqArr }
 */
export function postProcessSParameters(params) {
  const { ports, sampledVoltages, sampledCurrents,
          voltageSources, currentSources, dt, freqArr } = params;

  computePortWaves(ports, sampledVoltages, sampledCurrents);
  computeSMatrix(ports);
}
