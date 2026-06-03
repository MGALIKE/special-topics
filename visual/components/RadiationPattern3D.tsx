"use client";

import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import type { FarfieldData } from "@/lib/demoData";

interface RadPatternMeshProps {
  data: FarfieldData;
  color: string;
  opacity?: number;
}

function RadPatternMesh({ data, color, opacity = 0.75 }: RadPatternMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = clock.getElapsedTime() * 0.15;
    }
  });

  const geometry = useMemo(() => {
    const pts = data.angles_deg.length;
    const radii = data.dataTheta_dB as number[];
    const minR = Math.min(...radii.filter((v) => v !== null && isFinite(v)));
    const maxR = Math.max(...radii.filter((v) => v !== null && isFinite(v)));
    const range = maxR - minR || 1;

    // Normalize to 0..2 range for visualization
    const norm = radii.map((r) =>
      r !== null && isFinite(r) ? ((r - minR) / range) * 2 + 0.3 : 0.3
    );

    const shape = new THREE.Shape();
    for (let i = 0; i <= pts; i++) {
      const idx = i % pts;
      const angle = (data.angles_deg[idx] * Math.PI) / 180;
      const r = norm[idx];
      const x = r * Math.cos(angle);
      const y = r * Math.sin(angle);
      if (i === 0) shape.moveTo(x, y);
      else shape.lineTo(x, y);
    }

    const geo = new THREE.ShapeGeometry(shape, 64);

    // Color by radius
    const colors = new Float32Array(geo.attributes.position.count * 3);
    const colorObj = new THREE.Color(color);
    const accentColor = new THREE.Color("#ff6b35");
    const pos = geo.attributes.position;

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const dist = Math.sqrt(x * x + y * y);
      const t = Math.min(dist / 2.3, 1);
      const c = colorObj.clone().lerp(accentColor, t * 0.5);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [data, color]);

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshBasicMaterial
        vertexColors
        transparent
        opacity={opacity}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function PolarGrid() {
  const circles = [0.5, 1.0, 1.5, 2.0];
  return (
    <group>
      {circles.map((r) => (
        <mesh key={r}>
          <ringGeometry args={[r - 0.005, r + 0.005, 64]} />
          <meshBasicMaterial
            color="#1a2744"
            transparent
            opacity={0.4}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
      {/* Axis lines */}
      {[0, 45, 90, 135].map((deg) => {
        const rad = (deg * Math.PI) / 180;
        const pts = [
          new THREE.Vector3(-2.5 * Math.cos(rad), -2.5 * Math.sin(rad), 0),
          new THREE.Vector3(2.5 * Math.cos(rad), 2.5 * Math.sin(rad), 0),
        ];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        return (
          <lineSegments key={deg} geometry={geo}>
            <lineBasicMaterial color="#1a2744" transparent opacity={0.3} />
          </lineSegments>
        );
      })}
      {/* Labels */}
      {[0, 90, 180, 270].map((deg) => {
        const rad = (deg * Math.PI) / 180;
        return (
          <Text
            key={deg}
            position={[2.7 * Math.cos(rad), 2.7 * Math.sin(rad), 0]}
            fontSize={0.18}
            color="#4a5568"
          >
            {deg}°
          </Text>
        );
      })}
    </group>
  );
}

// Outline of the pattern
function RadPatternOutline({ data, color }: { data: FarfieldData; color: string }) {
  const geometry = useMemo(() => {
    const pts = data.angles_deg.length;
    const radii = data.dataTheta_dB as number[];
    const minR = Math.min(...radii.filter((v) => v !== null && isFinite(v)));
    const maxR = Math.max(...radii.filter((v) => v !== null && isFinite(v)));
    const range = maxR - minR || 1;

    const norm = radii.map((r) =>
      r !== null && isFinite(r) ? ((r - minR) / range) * 2 + 0.3 : 0.3
    );

    const points: THREE.Vector3[] = [];
    for (let i = 0; i <= pts; i++) {
      const idx = i % pts;
      const angle = (data.angles_deg[idx] * Math.PI) / 180;
      const r = norm[idx];
      points.push(new THREE.Vector3(r * Math.cos(angle), r * Math.sin(angle), 0.01));
    }

    return new THREE.BufferGeometry().setFromPoints(points);
  }, [data]);

  const lineRef = useRef<THREE.Line>(null!);

  useFrame(({ clock }) => {
    if (lineRef.current) {
      lineRef.current.rotation.z = clock.getElapsedTime() * 0.15;
    }
  });

  return (
    <line ref={lineRef as React.RefObject<THREE.Line>} geometry={geometry}>
      <lineBasicMaterial color={color} linewidth={2} transparent opacity={0.9} />
    </line>
  );
}

interface RadiationPattern3DProps {
  data: FarfieldData[];
}

export default function RadiationPattern3D({ data }: RadiationPattern3DProps) {
  const [selectedFreq, setSelectedFreq] = useState(0);
  const freqOptions = useMemo(() => {
    const seen = new Set<number>();
    return data
      .filter((d) => {
        if (seen.has(d.freq_GHz)) return false;
        seen.add(d.freq_GHz);
        return true;
      })
      .map((d) => d.freq_GHz);
  }, [data]);

  const currentData = useMemo(
    () => data.filter((d) => d.freq_GHz === freqOptions[selectedFreq]),
    [data, freqOptions, selectedFreq]
  );

  return (
    <div className="radiation-container">
      {/* Freq selector */}
      <div className="freq-selector">
        {freqOptions.map((f, i) => (
          <button
            key={f}
            onClick={() => setSelectedFreq(i)}
            className={`freq-btn ${i === selectedFreq ? "freq-btn--active" : ""}`}
          >
            {f.toFixed(1)} GHz
          </button>
        ))}
      </div>

      <Canvas
        camera={{ position: [0, 0, 5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <color attach="background" args={["#060d1a"]} />

        <PolarGrid />

        {currentData.map((d, i) => (
          <group key={`${d.plane}-${d.freq_GHz}`}>
            <RadPatternMesh
              data={d}
              color={d.plane === "xy" ? "#00d4ff" : "#a78bfa"}
              opacity={0.4}
            />
            <RadPatternOutline
              data={d}
              color={d.plane === "xy" ? "#00d4ff" : "#a78bfa"}
            />
          </group>
        ))}

        {/* Legend */}
        {currentData.length > 1 && (
          <>
            <Text position={[-2.2, -2.7, 0]} fontSize={0.15} color="#00d4ff">
              ● XY Plane
            </Text>
            <Text position={[0.5, -2.7, 0]} fontSize={0.15} color="#a78bfa">
              ● XZ Plane
            </Text>
          </>
        )}

        <OrbitControls enablePan={false} enableZoom={true} />
      </Canvas>
    </div>
  );
}
