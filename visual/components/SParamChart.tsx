"use client";

import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import type { SParamData } from "@/lib/demoData";

interface SParamChartProps {
  data: SParamData;
}

export default function SParamChart({ data }: SParamChartProps) {
  const chartData = useMemo(() => {
    const step = Math.max(1, Math.floor(data.frequencies_GHz.length / 300));
    const result = [];
    for (let i = 0; i < data.frequencies_GHz.length; i += step) {
      result.push({
        freq: data.frequencies_GHz[i],
        mag: data.magnitude_dB[i],
        phase: data.phase_deg[i],
      });
    }
    return result;
  }, [data]);

  const minMag = useMemo(() => {
    const validMags = chartData
      .map((d) => d.mag)
      .filter((v) => v !== null && isFinite(v as number)) as number[];
    return validMags.length > 0 ? Math.min(...validMags) : -25;
  }, [chartData]);

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="magGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f4f4f1" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#f4f4f1" stopOpacity={0} />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis
            dataKey="freq"
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
            yAxisId="left"
            stroke="#f4f4f1"
            tick={{ fill: "#f4f4f1", fontSize: 11 }}
            domain={[Math.floor(minMag / 5) * 5, 0]}
            label={{
              value: "|S11| (dB)",
              angle: -90,
              position: "insideLeft",
              fill: "#f4f4f1",
              fontSize: 11,
            }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#f4f4f1"
            tick={{ fill: "#ff3b1f", fontSize: 11 }}
            domain={[-180, 180]}
            label={{
              value: "Phase (°)",
              angle: 90,
              position: "insideRight",
              fill: "#ff3b1f",
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
            formatter={(value: number, name: string) => [
              value?.toFixed(2),
              name === "mag" ? "|S11| (dB)" : "Phase (°)",
            ]}
            labelFormatter={(label: number) => `${label.toFixed(2)} GHz`}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#9a9a9a" }}
            formatter={(value: string) =>
              value === "mag" ? "|S11| Magnitude" : "∠S11 Phase"
            }
          />
          <ReferenceLine
            yAxisId="left"
            y={-10}
            stroke="#ff3b1f"
            strokeDasharray="5 5"
            strokeOpacity={0.5}
            label={{
              value: "-10 dB",
              fill: "#ff3b1f",
              fontSize: 10,
              position: "right",
            }}
          />
          <ReferenceLine
            x={2.4}
            stroke="#f4f4f1"
            strokeDasharray="3 3"
            strokeOpacity={0.4}
            label={{
              value: "2.4 GHz",
              fill: "#f4f4f1",
              fontSize: 9,
              position: "top",
            }}
          />
          <ReferenceLine
            x={5.8}
            stroke="#f4f4f1"
            strokeDasharray="3 3"
            strokeOpacity={0.4}
            label={{
              value: "5.8 GHz",
              fill: "#f4f4f1",
              fontSize: 9,
              position: "top",
            }}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="mag"
            stroke="#f4f4f1"
            strokeWidth={2}
            dot={false}
            filter="url(#glow)"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="phase"
            stroke="#ff3b1f"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
            opacity={0.7}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
