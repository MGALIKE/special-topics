// lib/fft.ts
// Minimal radix-2 Cooley–Tukey FFT for the voltage-spectrum panel.
// The input is real-valued time-domain voltage; we zero-pad to the next
// power of two and return the single-sided magnitude spectrum.

/**
 * In-place iterative radix-2 FFT. `re`/`im` must have a power-of-two length.
 */
function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Danielson–Lanczos.
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const bIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = aRe + bRe;
        im[i + k] = aIm + bIm;
        re[i + k + len / 2] = aRe - bRe;
        im[i + k + len / 2] = aIm - bIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

export interface Spectrum {
  freq_GHz: number[];
  magnitude_dB: number[];
}

/**
 * Compute the single-sided amplitude spectrum of a real signal.
 *
 * @param values  real samples
 * @param dt_s    sample spacing in seconds
 * @returns frequency axis (GHz) and magnitude normalized to 0 dB peak
 */
export function computeSpectrum(values: number[], dt_s: number): Spectrum {
  const n0 = values.length;
  if (n0 < 2 || !isFinite(dt_s) || dt_s <= 0) {
    return { freq_GHz: [], magnitude_dB: [] };
  }

  // Next power of two >= n0.
  let n = 1;
  while (n < n0) n <<= 1;

  const re = new Float64Array(n);
  const im = new Float64Array(n);
  // Remove DC bias and apply a Hann window to suppress spectral leakage.
  let mean = 0;
  for (let i = 0; i < n0; i++) mean += values[i];
  mean /= n0;
  for (let i = 0; i < n0; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n0 - 1)));
    re[i] = (values[i] - mean) * w;
  }

  fftInPlace(re, im);

  const half = n >> 1;
  const fs = 1 / dt_s; // sample rate (Hz)
  const freq_GHz: number[] = new Array(half);
  const mag: number[] = new Array(half);
  let peak = 1e-30;
  for (let k = 0; k < half; k++) {
    const m = Math.hypot(re[k], im[k]);
    mag[k] = m;
    if (m > peak) peak = m;
    freq_GHz[k] = (k * fs) / n / 1e9;
  }

  const magnitude_dB = mag.map((m) => 20 * Math.log10(Math.max(m / peak, 1e-6)));
  return { freq_GHz, magnitude_dB };
}
