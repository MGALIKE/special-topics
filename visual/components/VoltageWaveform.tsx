"use client";

import { useMemo, useState, useEffect, useCallback } from "react";
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

interface VoltageWaveformProps {
  data: VoltageData;
  isLive?: boolean;
}

export default function VoltageWaveform({
  data,
  isLive = false,
}: VoltageWaveformProps) {
  const [cursor, setCursor] = useState(0);
  const [isPlaying, setIsPlaying] = useState(!isLive);

  const chartData = useMemo(() => {
    // During live streaming, show all available points (less downsampling)
    const maxPoints = isLive ? 600 : 400;
    const step = Math.max(1, Math.floor(data.time_ns.length / maxPoints));
    const result = [];
    for (let i = 0; i < data.time_ns.length; i += step) {
      result.push({
        t: data.time_ns[i],
        v: data.values[i],
      });
    }
    return result;
  }, [data, isLive]);

  const maxTime = chartData[chartData.length - 1]?.t ?? 1;

  // Animated cursor for playback mode (non-live)
  useEffect(() => {
    if (isLive || !isPlaying) return;
    const interval = setInterval(() => {
      setCursor((prev) => {
        const next = prev + maxTime * 0.003;
        return next > maxTime ? 0 : next;
      });
    }, 30);
    return () => clearInterval(interval);
  }, [isPlaying, maxTime, isLive]);

  // In live mode, cursor follows the latest data point
  useEffect(() => {
    if (isLive && chartData.length > 0) {
      setCursor(chartData[chartData.length - 1].t);
    }
  }, [isLive, chartData]);

  const togglePlay = useCallback(() => setIsPlaying((p) => !p), []);

  return (
    <div className="chart-container">
      <div className="waveform-controls">
        {isLive ? (
          <span className="live-indicator">
            <span className="live-indicator-dot" />
            LIVE
          </span>
        ) : (
          <button onClick={togglePlay} className="play-btn">
            {isPlaying ? (
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <polygon points="5,3 19,12 5,21" />
              </svg>
            )}
          </button>
        )}
        <span className="waveform-time">
          {isLive
            ? `${chartData.length} samples — ${maxTime.toFixed(3)} ns`
            : `${cursor.toFixed(2)} / ${maxTime.toFixed(2)} ns`}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <AreaChart
          data={chartData}
          margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
        >
          <defs>
            <linearGradient id="voltGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor={isLive ? "#00d4ff" : "#34d399"}
                stopOpacity={0.3}
              />
              <stop
                offset="95%"
                stopColor={isLive ? "#00d4ff" : "#34d399"}
                stopOpacity={0}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a2744" />
          <XAxis
            dataKey="t"
            stroke="#4a5568"
            tick={{ fill: "#718096", fontSize: 11 }}
            label={{
              value: "Time (ns)",
              position: "insideBottom",
              offset: -2,
              fill: "#718096",
              fontSize: 11,
            }}
          />
          <YAxis
            stroke="#4a5568"
            tick={{ fill: isLive ? "#00d4ff" : "#34d399", fontSize: 11 }}
            label={{
              value: "Voltage (V)",
              angle: -90,
              position: "insideLeft",
              fill: isLive ? "#00d4ff" : "#34d399",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(10, 22, 40, 0.95)",
              border: "1px solid #1a2744",
              borderRadius: "8px",
              color: "#e2e8f0",
              fontSize: 12,
            }}
            formatter={(value: number) => [
              (value ?? 0).toFixed(6) + " V",
              "Voltage",
            ]}
            labelFormatter={(label: number) => `t = ${(label ?? 0).toFixed(3)} ns`}
          />
          {!isLive && (
            <ReferenceLine
              x={cursor}
              stroke="#ff6b35"
              strokeWidth={2}
              strokeOpacity={0.8}
            />
          )}
          <Area
            type="monotone"
            dataKey="v"
            stroke={isLive ? "#00d4ff" : "#34d399"}
            strokeWidth={1.5}
            fill="url(#voltGradient)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
