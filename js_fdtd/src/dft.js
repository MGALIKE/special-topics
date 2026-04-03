// DFT: time↔frequency domain transforms.
// Mirrors MATLAB time_to_frequency_domain.m and frequency_to_time_domain.m
//
// time_to_frequency_domain: X(f) = dt * Σ_{n=1}^{N} x(n) * exp(-j*w*(n*dt + time_shift))
// frequency_to_time_domain: x(t) = df * [ X(0) + 2*Re( Σ_{m=2}^{M} X(m)*exp(j*w*(m-1)*t) ) ]

/**
 * Time → frequency domain DFT.
 * Equivalent to MATLAB time_to_frequency_domain.m
 *
 * @param {Float64Array} x        - time-domain samples, length N (1-indexed in MATLAB: x(1)..x(N))
 * @param {number}       dt       - FDTD time step (seconds)
 * @param {Float64Array} freqArr  - frequency array (Hz), length M
 * @param {number}       timeShift - 0 for E-fields; -dt/2 for H-fields and currents
 * @returns {{ re: Float64Array, im: Float64Array }} - complex spectrum, length M
 */
export function timeToFreqDomain(x, dt, freqArr, timeShift) {
  const N = x.length;
  const M = freqArr.length;
  const Xre = new Float64Array(M);
  const Xim = new Float64Array(M);

  for (let mi = 0; mi < M; mi++) {
    const w = 2 * Math.PI * freqArr[mi];
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      // MATLAB: t = (n+1)*dt + time_shift  (1-based n in MATLAB → n+1 here)
      const t = (n + 1) * dt + timeShift;
      const phase = w * t;
      // exp(-j*phase) = cos(phase) - j*sin(phase)
      re += x[n] * Math.cos(phase);
      im -= x[n] * Math.sin(phase);
    }
    Xre[mi] = re * dt;
    Xim[mi] = im * dt;
  }
  return { re: Xre, im: Xim };
}

/**
 * Frequency → time domain inverse DFT.
 * Assumes real-valued time signal with conjugate-symmetric spectrum.
 * Equivalent to MATLAB frequency_to_time_domain.m
 *
 * @param {{ re: Float64Array, im: Float64Array }} X - complex spectrum (one-sided)
 * @param {number} df         - frequency step (Hz)
 * @param {Float64Array} timeArr - time points to evaluate (seconds)
 * @returns {Float64Array} real-valued time-domain signal
 */
export function freqToTimeDomain(X, df, timeArr) {
  const M = X.re.length;
  const T = timeArr.length;
  const x = new Float64Array(T);
  const dw = 2 * Math.PI * df;

  // DC component: X(0) (real for real-valued time signal)
  for (let ti = 0; ti < T; ti++) {
    x[ti] = X.re[0];
  }

  // Positive frequencies: add X(m)*exp(j*w*t) + conj(X(m))*exp(-j*w*t) = 2*Re(X(m)*exp(j*w*t))
  for (let m = 1; m < M; m++) {
    const w = m * dw;  // MATLAB: w = (m-1)*dw with 1-based m, same as m*dw with 0-based m
    const Xr = X.re[m];
    const Xi = X.im[m];
    for (let ti = 0; ti < T; ti++) {
      const phase = w * timeArr[ti];
      const cosP = Math.cos(phase);
      const sinP = Math.sin(phase);
      // X(m)*exp(j*phase) + conj(X(m))*exp(-j*phase) = 2*(Xr*cosP - Xi*sinP)
      x[ti] += 2 * (Xr * cosP - Xi * sinP);
    }
  }

  for (let ti = 0; ti < T; ti++) {
    x[ti] *= df;
  }
  return x;
}

/**
 * Compute DFT for all field/voltage/current observers after the time loop.
 * Modifies each observer in-place, adding frequency_domain_value = { re, im }.
 *
 * @param {object[]} observers      - array of observer objects with .sampled_value (Float64Array)
 * @param {number}   dt             - time step
 * @param {number}   timeShift      - 0 for E-field/voltage; -dt/2 for H-field/current
 * @param {Float64Array} freqArr    - frequency array
 */
export function computeObserverDFT(observers, dt, timeShift, freqArr) {
  for (const obs of observers) {
    obs.frequency_domain_value = timeToFreqDomain(obs.sampled_value, dt, freqArr, timeShift);
    obs.frequencies = freqArr;
  }
}
