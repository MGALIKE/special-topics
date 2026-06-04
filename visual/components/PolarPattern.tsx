"use client";

import { useMemo, useState } from "react";
import type { FarfieldData } from "@/lib/demoData";

interface PolarPatternProps {
  data: FarfieldData[];
}

const SIZE = 320;
const CX = SIZE / 2;
const CY = SIZE / 2;
const R = 130;
const DYN_RANGE = 40; // dB shown from outer ring (peak) to center

// Round to a fixed precision so the SVG coordinate strings React emits are
// byte-identical on the server and the client (avoids hydration mismatches
// from floating-point formatting differences).
const r3 = (n: number) => Math.round(n * 1000) / 1000;

// Polar: 0° at top, increasing clockwise.
function polarToPx(deg: number, rNorm: number): [number, number] {
  const a = ((deg - 90) * Math.PI) / 180;
  return [r3(CX + rNorm * R * Math.cos(a)), r3(CY + rNorm * R * Math.sin(a))];
}

function buildLocus(angles: number[], vals: (number | null)[], peak: number): string {
  const floor = peak - DYN_RANGE;
  const pts: string[] = [];
  for (let i = 0; i < angles.length; i++) {
    const v = vals[i];
    if (v == null || !isFinite(v)) continue;
    const rNorm = Math.max(0, Math.min(1, (v - floor) / DYN_RANGE));
    const [px, py] = polarToPx(angles[i], rNorm);
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  if (pts.length) pts.push(pts[0]); // close loop
  return pts.join(" ");
}

export default function PolarPattern({ data }: PolarPatternProps) {
  const [sel, setSel] = useState(0);
  const cut = data[Math.min(sel, data.length - 1)];

  const { thetaLocus, phiLocus, peak } = useMemo(() => {
    if (!cut) return { thetaLocus: "", phiLocus: "", peak: 0 };
    let pk = -Infinity;
    for (const arr of [cut.dataTheta_dB, cut.dataPhi_dB]) {
      for (const v of arr) if (v != null && isFinite(v) && v > pk) pk = v;
    }
    if (!isFinite(pk)) pk = 0;
    const peak = Math.ceil(pk / 5) * 5; // round outer ring up to a clean 5 dB
    return {
      thetaLocus: buildLocus(cut.angles_deg, cut.dataTheta_dB, peak),
      phiLocus: buildLocus(cut.angles_deg, cut.dataPhi_dB, peak),
      peak,
    };
  }, [cut]);

  if (!cut) {
    return (
      <div className="chart-container flex items-center justify-center h-[280px]">
        <span className="text-sm text-gray-500">No farfield data</span>
      </div>
    );
  }

  const rings = [0, 0.25, 0.5, 0.75, 1]; // fraction of radius
  const spokes = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330];

  return (
    <div className="chart-container flex flex-col items-center">
      {/* Cut selector */}
      <div className="flex flex-wrap gap-1 mb-1 justify-center">
        {data.map((d, i) => (
          <button
            key={`${d.plane}-${d.freq_GHz}-${i}`}
            onClick={() => setSel(i)}
            className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide transition-colors"
            style={{
              background: i === sel ? "#f4f4f1" : "#151515",
              color: i === sel ? "#151515" : "#9a9a9a",
              border: `1px solid ${i === sel ? "#f4f4f1" : "#2a2a2a"}`,
            }}
          >
            {d.plane.toUpperCase()} · {d.freq_GHz} GHz
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width="100%" height={250} style={{ maxWidth: SIZE }}>
        {/* Radial grid + dB labels */}
        <g stroke="#2a2a2a" fill="none">
          {rings.map((f) => (
            <circle key={f} cx={CX} cy={CY} r={f * R} />
          ))}
        </g>
        <g stroke="#2a2a2a">
          {spokes.map((deg) => {
            const [x, y] = polarToPx(deg, 1);
            return <line key={deg} x1={CX} y1={CY} x2={x} y2={y} />;
          })}
        </g>
        {/* dB ring labels */}
        <g fill="#6a6a6a" fontSize={8} fontFamily="monospace">
          {rings.slice(1).map((f) => (
            <text key={f} x={CX + 2} y={CY - f * R + 9}>
              {(peak - (1 - f) * DYN_RANGE).toFixed(0)}
            </text>
          ))}
        </g>
        {/* Angle labels */}
        <g fill="#9a9a9a" fontSize={9} fontFamily="monospace" textAnchor="middle">
          {[0, 90, 180, 270].map((deg) => {
            const [x, y] = polarToPx(deg, 1.12);
            return (
              <text key={deg} x={x} y={y + 3}>
                {deg}°
              </text>
            );
          })}
        </g>

        <polyline points={thetaLocus} fill="none" stroke="#f4f4f1" strokeWidth={2} />
        <polyline
          points={phiLocus}
          fill="none"
          stroke="#ff3b1f"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      </svg>

      <div className="flex gap-4 mt-1 text-[11px] font-mono">
        <span style={{ color: "#f4f4f1" }}>— E_θ</span>
        <span style={{ color: "#ff3b1f" }}>-- E_φ</span>
        <span className="text-gray-500">peak {peak} dB · {DYN_RANGE} dB range</span>
      </div>
    </div>
  );
}
