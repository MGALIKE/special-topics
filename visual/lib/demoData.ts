// lib/demoData.ts
// Generate synthetic FDTD simulation data for demonstration
// Produces realistic-looking IFA response at 2.4 GHz and 5.8 GHz

export interface SimulationData {
  meta: {
    generated: string;
    numberOfTimeSteps: number;
    dt: number;
    frequencies_Hz: number[];
  };
  sparams: SParamData[];
  farfield: FarfieldData[];
  timeDomain: {
    voltages: VoltageData[];
  };
}

export interface SParamData {
  label: string;
  frequencies_GHz: number[];
  magnitude_dB: (number | null)[];
  phase_deg: (number | null)[];
}

export interface FarfieldData {
  plane: string;
  freq_GHz: number;
  angles_deg: number[];
  dataTheta_dB: (number | null)[];
  dataPhi_dB: (number | null)[];
}

export interface VoltageData {
  label: string;
  time_ns: number[];
  values: number[];
}

function hasValidData(arr: (number | null)[] | undefined): boolean {
  if (!arr || arr.length === 0) return false;
  return arr.some((v) => v !== null && isFinite(v as number));
}

export function isDataValid(data: SimulationData): boolean {
  const sparamValid =
    data.sparams?.length > 0 && hasValidData(data.sparams[0].magnitude_dB);
  const ffValid =
    data.farfield?.length > 0 && hasValidData(data.farfield[0].dataTheta_dB);
  return sparamValid || ffValid;
}

// Generate realistic S11 for an IFA with resonances at 2.4 and 5.8 GHz
export function generateDemoSParams(): SParamData {
  const nFreqs = 500;
  const maxFreq = 10; // GHz
  const freqs: number[] = [];
  const magdB: number[] = [];
  const phaseDeg: number[] = [];

  for (let i = 0; i < nFreqs; i++) {
    const f = ((i + 1) / nFreqs) * maxFreq;
    freqs.push(f);

    // Two resonances with Lorentzian dips
    const r1 = -18 * Math.exp(-Math.pow((f - 2.42) / 0.12, 2)); // 2.4 GHz resonance
    const r2 = -14 * Math.exp(-Math.pow((f - 5.78) / 0.18, 2)); // 5.8 GHz resonance
    const r3 = -6 * Math.exp(-Math.pow((f - 8.2) / 0.3, 2)); // spurious

    // Base return loss curve (gentle slope)
    const base = -0.5 - 0.02 * f;
    const ripple = 0.3 * Math.sin(f * 4.5) * Math.exp(-f * 0.1);
    const mag = Math.min(base + r1 + r2 + r3 + ripple, -0.1);
    magdB.push(mag);

    // Phase: smooth transition through resonances
    const p1 = -120 * Math.atan((f - 2.42) / 0.08);
    const p2 = -80 * Math.atan((f - 5.78) / 0.12);
    const phaseBase = 170 - f * 8;
    phaseDeg.push(((phaseBase + p1 + p2 + 360 * 10) % 360) - 180);
  }

  return {
    label: "S11",
    frequencies_GHz: freqs,
    magnitude_dB: magdB,
    phase_deg: phaseDeg,
  };
}

// Generate radiation pattern for a given plane and frequency
export function generateDemoFarfield(
  plane: string,
  freqGHz: number
): FarfieldData {
  const nAngles = 360;
  const angles: number[] = [];
  const dataTheta: number[] = [];
  const dataPhi: number[] = [];

  const isLowFreq = freqGHz < 4;

  for (let i = 0; i < nAngles; i++) {
    const deg = i;
    const rad = (deg * Math.PI) / 180;
    angles.push(deg);

    if (plane === "xy") {
      // XY plane: nearly omnidirectional with slight asymmetry
      const base = isLowFreq ? 2.5 : 4.2;
      const dTheta =
        base +
        1.2 * Math.cos(rad) +
        0.4 * Math.cos(2 * rad) +
        (isLowFreq ? 0 : 0.8 * Math.cos(3 * rad));
      const dPhi =
        base -
        0.8 +
        0.9 * Math.sin(rad) +
        0.3 * Math.sin(2 * rad) +
        (isLowFreq ? 0 : 0.5 * Math.sin(3 * rad));
      dataTheta.push(dTheta);
      dataPhi.push(dPhi);
    } else {
      // XZ plane: figure-eight like with ground plane effect
      const base = isLowFreq ? 2.0 : 3.5;
      const dTheta =
        base +
        2.5 * Math.abs(Math.sin(rad)) +
        0.6 * Math.cos(2 * rad) -
        1.5 * Math.max(0, -Math.cos(rad)); // ground plane suppression
      const dPhi =
        base -
        1.0 +
        1.8 * Math.abs(Math.sin(rad)) +
        0.4 * Math.cos(2 * rad) -
        1.2 * Math.max(0, -Math.cos(rad));
      dataTheta.push(Math.max(dTheta, -15));
      dataPhi.push(Math.max(dPhi, -15));
    }
  }

  return {
    plane,
    freq_GHz: freqGHz,
    angles_deg: angles,
    dataTheta_dB: dataTheta,
    dataPhi_dB: dataPhi,
  };
}

// Generate time-domain voltage (Gaussian pulse + ringing)
export function generateDemoVoltage(): VoltageData {
  const nSteps = 7000;
  const dt = 5.77e-13;
  const times: number[] = [];
  const values: number[] = [];

  const t0 = 0.15; // ns, pulse center
  const sigma = 0.04; // ns, pulse width

  for (let i = 0; i < nSteps; i++) {
    const t = (i + 0.5) * dt * 1e9; // in ns
    times.push(t);

    // Gaussian pulse with ringing
    const gauss = Math.exp(-Math.pow((t - t0) / sigma, 2));
    const ring1 =
      0.15 * Math.exp(-Math.pow((t - t0 - 0.3) / 0.6, 2)) * Math.sin(2 * Math.PI * 2.4 * t);
    const ring2 =
      0.08 * Math.exp(-Math.pow((t - t0 - 0.5) / 0.8, 2)) * Math.sin(2 * Math.PI * 5.8 * t);
    const decay =
      0.03 *
      Math.exp(-Math.pow((t - 0.5) / 1.5, 2)) *
      Math.sin(2 * Math.PI * 2.4 * t + 0.5);

    values.push(gauss + ring1 + ring2 + decay);
  }

  return {
    label: "v1",
    time_ns: times,
    values,
  };
}

export function generateFullDemoData(): SimulationData {
  const sparams = generateDemoSParams();
  const farfield = [
    generateDemoFarfield("xy", 2.4),
    generateDemoFarfield("xz", 2.4),
    generateDemoFarfield("xy", 5.8),
    generateDemoFarfield("xz", 5.8),
  ];
  const voltage = generateDemoVoltage();

  return {
    meta: {
      generated: new Date().toISOString(),
      numberOfTimeSteps: 7000,
      dt: 5.77e-13,
      frequencies_Hz: sparams.frequencies_GHz.map((f) => f * 1e9),
    },
    sparams: [sparams],
    farfield,
    timeDomain: {
      voltages: [voltage],
    },
  };
}
