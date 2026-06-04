"use client";

import { useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { VoltageData } from "@/lib/demoData";
import { computeSpectrum } from "@/lib/fft";

interface VoltageSpectrumProps {
  data: VoltageData;
  /** Highlight markers, e.g. design frequencies in GHz. */
  markers?: number[];
  /** Upper bound of the frequency axis (GHz). */
  maxFreqGHz?: number;
}

export default function VoltageSpectrum({
  data,
  markers = [2.4, 5.8],
  maxFreqGHz = 10,
}: VoltageSpectrumProps) {
  const chartData = useMemo(() => {
    if (!data?.time_ns || data.time_ns.length < 4) return [];
    // Derive dt from the (uniform) time axis; time_ns is in nanoseconds.
    const dt_s = ((data.time_ns[1] - data.time_ns[0]) * 1e-9) || 0;
    const { freq_GHz, magnitude_dB } = computeSpectrum(data.values, dt_s);

    const out: { freq: number; mag: number }[] = [];
    // Downsample to keep the path light.
    const maxPoints = 400;
    const inBand = freq_GHz.filter((f) => f <= maxFreqGHz).length || freq_GHz.length;
    const step = Math.max(1, Math.floor(inBand / maxPoints));
    for (let i = 0; i < freq_GHz.length; i += step) {
      if (freq_GHz[i] > maxFreqGHz) break;
      out.push({ freq: freq_GHz[i], mag: magnitude_dB[i] });
    }
    return out;
  }, [data, maxFreqGHz]);

  // Find the dominant spectral peak for an annotation.
  const peak = useMemo(() => {
    let best = { freq: 0, mag: -Infinity };
    for (const d of chartData) {
      if (d.freq > 0.1 && d.mag > best.mag) best = d;
    }
    return best.mag > -Infinity ? best : null;
  }, [chartData]);

  if (chartData.length === 0) {
    return (
      <div className="chart-container flex items-center justify-center h-[240px]">
        <span className="text-sm text-gray-500">No spectrum available</span>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <div className="waveform-controls">
        <span className="waveform-time">
          {peak ? `Peak: ${peak.freq.toFixed(2)} GHz` : "Spectrum"} · Hann-windowed FFT
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="spectrumGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f4f4f1" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#f4f4f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis
            dataKey="freq"
            type="number"
            domain={[0, maxFreqGHz]}
            stroke="#f4f4f1"
            tick={{ fill: "#9a9a9a", fontSize: 11 }}
            label={{
              value: "Frequency (GHz)",
              position: "insideBottom",
              offset: -2,
              fill: "#9a9a9a",
              fontSize: 11,
            }}
          />
          <YAxis
            stroke="#f4f4f1"
            tick={{ fill: "#f4f4f1", fontSize: 11 }}
            domain={[-60, 0]}
            label={{
              value: "Magnitude (dB)",
              angle: -90,
              position: "insideLeft",
              fill: "#f4f4f1",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#151515",
              border: "1.5px solid #f4f4f1",
              borderRadius: "0px",
              color: "#f4f4f1",
              fontSize: 12,
            }}
            formatter={(value) => [`${Number(value ?? 0).toFixed(1)} dB`, "Magnitude"]}
            labelFormatter={(label) => `${Number(label ?? 0).toFixed(3)} GHz`}
          />
          {markers.map((m) => (
            <ReferenceLine
              key={m}
              x={m}
              stroke="#f4f4f1"
              strokeDasharray="3 3"
              strokeOpacity={0.4}
              label={{
                value: `${m} GHz`,
                fill: "#f4f4f1",
                fontSize: 9,
                position: "top",
              }}
            />
          ))}
          <Area
            type="monotone"
            dataKey="mag"
            stroke="#f4f4f1"
            strokeWidth={1.5}
            fill="url(#spectrumGradient)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
