"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, Float } from "@react-three/drei";
import * as THREE from "three";

// Palette — kept in sync with the Swiss dark theme.
const COPPER = "#c87a45";
const FR4 = "#0e3b34";
const RED = "#ff3b1f";
const BONE = "#f4f4f1";

// The feed/excitation point, in the antenna's local space. Everything that
// "radiates" (wave rings, particles) emanates from here.
const FEED = new THREE.Vector3(-0.8, 0.22, 0.12);

/* ── Animated EM energy streaming off the feed point ───────────────────── */
function EMParticles({ count = 160 }: { count?: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const particles = useMemo(() => {
    const arr = [];
    for (let i = 0; i < count; i++) {
      arr.push({
        theta: Math.random() * Math.PI * 2,
        phi: Math.acos(2 * Math.random() - 1),
        speed: 0.4 + Math.random() * 0.9,
        offset: Math.random() * Math.PI * 2,
        maxRadius: 2.2 + Math.random() * 2.4,
      });
    }
    return arr;
  }, [count]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    particles.forEach((p, i) => {
      const r = (t * p.speed + p.offset) % p.maxRadius;
      dummy.position.set(
        FEED.x + r * Math.sin(p.phi) * Math.cos(p.theta),
        FEED.y + r * Math.cos(p.phi),
        FEED.z + r * Math.sin(p.phi) * Math.sin(p.theta)
      );
      dummy.scale.setScalar(Math.max(0, 1 - r / p.maxRadius) * 0.045);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshBasicMaterial color={RED} transparent opacity={0.85} />
    </instancedMesh>
  );
}

/* ── Expanding wavefront rings (radiation) ─────────────────────────────── */
function WaveRings() {
  const ringsRef = useRef<THREE.Group>(null!);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    ringsRef.current?.children.forEach((child, i) => {
      const ring = child as THREE.Mesh;
      const phase = (t * 0.55 + i * 0.5) % 3;
      ring.scale.setScalar(phase * 1.7);
      (ring.material as THREE.MeshBasicMaterial).opacity = Math.max(
        0,
        0.5 * (1 - phase / 3)
      );
    });
  });

  return (
    <group ref={ringsRef} position={FEED.toArray()} rotation={[Math.PI / 2, 0, 0]}>
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <mesh key={i}>
          <ringGeometry args={[0.97, 1, 80]} />
          <meshBasicMaterial
            color={RED}
            transparent
            opacity={0.35}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

/* ── A single copper trace segment ─────────────────────────────────────── */
function Trace({
  size,
  position,
  radiating = false,
}: {
  size: [number, number, number];
  position: [number, number, number];
  radiating?: boolean;
}) {
  return (
    <mesh position={position} castShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial
        color={COPPER}
        metalness={1}
        roughness={0.32}
        emissive={radiating ? RED : "#000000"}
        emissiveIntensity={radiating ? 0.35 : 0}
      />
    </mesh>
  );
}

/* ── The Inverted-F Antenna on its PCB ─────────────────────────────────── */
function AntennaModel() {
  return (
    <Float speed={1.2} rotationIntensity={0.12} floatIntensity={0.25}>
      <group position={[0, -0.3, 0]}>
        {/* Dielectric substrate (FR-4 board) */}
        <mesh receiveShadow position={[0, 0, 0]}>
          <boxGeometry args={[5, 0.16, 7]} />
          <meshStandardMaterial color={FR4} metalness={0.1} roughness={0.6} />
        </mesh>

        {/* Copper ground plane — covers the rear portion of the board */}
        <mesh position={[0, 0.105, 1.75]}>
          <boxGeometry args={[4.6, 0.05, 2.9]} />
          <meshStandardMaterial color={COPPER} metalness={1} roughness={0.3} />
        </mesh>

        {/* Inverted-F trace: top radiating arm + feed line + shorting stub */}
        <Trace size={[3.6, 0.06, 0.34]} position={[0.2, 0.14, -2.8]} radiating />
        <Trace size={[0.34, 0.06, 2.8]} position={[-0.8, 0.14, -1.4]} />
        <Trace size={[0.34, 0.06, 3.3]} position={[-1.6, 0.14, -1.15]} />

        {/* Feed port — the excitation gap between feed line and ground */}
        <mesh position={FEED.toArray()}>
          <sphereGeometry args={[0.11, 20, 20]} />
          <meshStandardMaterial
            color={BONE}
            emissive={RED}
            emissiveIntensity={2.4}
          />
        </mesh>
      </group>
    </Float>
  );
}

/* ── Faint reference grid beneath the board ────────────────────────────── */
function GridFloor() {
  return (
    <gridHelper
      args={[28, 28, "#3a3a3a", "#1c1c1c"]}
      position={[0, -1.4, 0]}
    />
  );
}

export default function AntennaScene() {
  return (
    <div className="antenna-canvas-container">
      <Canvas
        shadows
        camera={{ position: [5, 3.4, 5.5], fov: 42 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <color attach="background" args={["#000000"]} />
        <fog attach="fog" args={["#000000", 11, 26]} />

        {/* Lighting — white key + red rim for the Swiss accent */}
        <ambientLight intensity={0.35} />
        <directionalLight position={[6, 9, 5]} intensity={1.6} color={BONE} castShadow />
        <pointLight position={[-5, 3, -4]} intensity={0.8} color={RED} />
        <spotLight position={[0, 10, 2]} angle={0.4} penumbra={1} intensity={0.6} color={BONE} />

        <AntennaModel />
        <WaveRings />
        <EMParticles count={160} />
        <GridFloor />

        <OrbitControls
          enablePan={false}
          enableZoom
          autoRotate
          autoRotateSpeed={0.7}
          minDistance={4}
          maxDistance={16}
          target={[0, 0, 0]}
        />
        <Environment preset="city" />
      </Canvas>

      <div className="scene-label">
        <span className="scene-badge">LIVE</span>
        Inverted-F Antenna — PCB Model
      </div>
    </div>
  );
}
