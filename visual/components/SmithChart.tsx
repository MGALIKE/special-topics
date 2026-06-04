"use client";

import { useMemo } from "react";
import type { SParamData } from "@/lib/demoData";

interface SmithChartProps {
  data: SParamData;
  /** Design frequencies (GHz) to annotate with markers. */
  markers?: number[];
}

const SIZE = 320;
const CX = SIZE / 2;
const CY = SIZE / 2;
const R = 140;

// Round to a fixed precision so server- and client-rendered SVG coordinate
// strings match exactly (avoids React hydration mismatches).
const r3 = (n: number) => Math.round(n * 1000) / 1000;

// Map a reflection coefficient Γ=(gr,gi) (unit disk) to SVG pixel space.
function toPx(gr: number, gi: number): [number, number] {
  return [r3(CX + gr * R), r3(CY - gi * R)];
}

// Constant-resistance circle in Γ-plane: center (r/(1+r), 0), radius 1/(1+r).
function resCircle(r: number) {
  const cx = r / (1 + r);
  const rad = 1 / (1 + r);
  const [px, py] = toPx(cx, 0);
  return { cx: px, cy: py, r: r3(rad * R) };
}

// Constant-reactance arc in Γ-plane: center (1, 1/x), radius 1/x.
function reactCircle(x: number) {
  const [px, py] = toPx(1, 1 / x);
  return { cx: px, cy: py, r: r3(Math.abs(1 / x) * R) };
}

export default function SmithChart({ data, markers = [2.4, 5.8] }: SmithChartProps) {
  const { path, dots } = useMemo(() => {
    const freqs = data.frequencies_GHz;
    const pts: string[] = [];
    const step = Math.max(1, Math.floor(freqs.length / 300));
    for (let i = 0; i < freqs.length; i += step) {
      const dB = data.magnitude_dB[i];
      const ph = data.phase_deg[i];
      if (dB == null || ph == null || !isFinite(dB) || !isFinite(ph)) continue;
      const mag = Math.min(Math.pow(10, dB / 20), 1); // clamp into unit disk
      const rad = (ph * Math.PI) / 180;
      const [px, py] = toPx(mag * Math.cos(rad), mag * Math.sin(rad));
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
    }

    // Markers nearest each design frequency.
    const dots = markers
      .map((mf) => {
        let bi = -1;
        let bd = Infinity;
        for (let i = 0; i < freqs.length; i++) {
          const d = Math.abs(freqs[i] - mf);
          if (d < bd) {
            bd = d;
            bi = i;
          }
        }
        if (bi < 0) return null;
        const dB = data.magnitude_dB[bi];
        const ph = data.phase_deg[bi];
        if (dB == null || ph == null || !isFinite(dB) || !isFinite(ph)) return null;
        const mag = Math.min(Math.pow(10, dB / 20), 1);
        const rad = (ph * Math.PI) / 180;
        const [px, py] = toPx(mag * Math.cos(rad), mag * Math.sin(rad));
        const vswr = mag < 1 ? (1 + mag) / (1 - mag) : Infinity;
        return { px, py, label: `${mf} GHz`, vswr };
      })
      .filter((d): d is NonNullable<typeof d> => d !== null);

    return { path: pts.join(" "), dots };
  }, [data, markers]);

  const resVals = [0.2, 0.5, 1, 2, 5];
  const reactVals = [0.2, 0.5, 1, 2, 5];

  return (
    <div className="chart-container flex flex-col items-center">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        width="100%"
        height={280}
        style={{ maxWidth: SIZE }}
      >
        <defs>
          <clipPath id="smithClip">
            <circle cx={CX} cy={CY} r={R} />
          </clipPath>
          <filter id="smithGlow">
            <feGaussianBlur stdDeviation="2" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <g clipPath="url(#smithClip)" stroke="#2a2a2a" fill="none" strokeWidth={1}>
          {resVals.map((r) => {
            const c = resCircle(r);
            return <circle key={`r${r}`} cx={c.cx} cy={c.cy} r={c.r} />;
          })}
          {reactVals.map((x) => {
            const cp = reactCircle(x);
            const cn = reactCircle(-x);
            return (
              <g key={`x${x}`}>
                <circle cx={cp.cx} cy={cp.cy} r={cp.r} />
                <circle cx={cn.cx} cy={cn.cy} r={cn.r} />
              </g>
            );
          })}
          {/* Real axis */}
          <line x1={CX - R} y1={CY} x2={CX + R} y2={CY} stroke="#c8c8c8" />
        </g>

        {/* Outer boundary */}
        <circle cx={CX} cy={CY} r={R} fill="none" stroke="#f4f4f1" strokeWidth={1.5} />

        {/* S11 locus */}
        <polyline
          points={path}
          fill="none"
          stroke="#f4f4f1"
          strokeWidth={2}
          filter="url(#smithGlow)"
        />

        {/* Markers */}
        {dots.map((d) => (
          <g key={d.label}>
            <circle cx={d.px} cy={d.py} r={4} fill="#ff3b1f" stroke="#151515" strokeWidth={1.5} />
            <text
              x={d.px + 7}
              y={d.py - 5}
              fill="#ff3b1f"
              fontSize={10}
              fontFamily="monospace"
            >
              {d.label}
            </text>
          </g>
        ))}
      </svg>

      <div className="flex gap-4 mt-1 text-[11px] text-gray-400 font-mono">
        {dots.map((d) => (
          <span key={d.label}>
            <span style={{ color: "#ff3b1f" }}>{d.label}</span>{" "}
            VSWR {isFinite(d.vswr) ? d.vswr.toFixed(2) : "∞"}
          </span>
        ))}
      </div>
    </div>
  );
}
