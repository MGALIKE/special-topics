"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  Environment,
  Float,
  MeshTransmissionMaterial,
} from "@react-three/drei";
import * as THREE from "three";

// Animated EM particles emanating from antenna feed point
function EMParticles({ count = 200 }: { count?: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const particles = useMemo(() => {
    const arr = [];
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const speed = 0.3 + Math.random() * 0.7;
      arr.push({
        theta,
        phi,
        speed,
        offset: Math.random() * Math.PI * 2,
        radius: 0,
        maxRadius: 2.5 + Math.random() * 2,
      });
    }
    return arr;
  }, [count]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    particles.forEach((p, i) => {
      p.radius = ((t * p.speed + p.offset) % p.maxRadius);
      const r = p.radius;
      const x = r * Math.sin(p.phi) * Math.cos(p.theta);
      const y = r * Math.sin(p.phi) * Math.sin(p.theta);
      const z = r * Math.cos(p.phi);

      dummy.position.set(x, y + 0.5, z);
      const scale = Math.max(0, 1 - r / p.maxRadius) * 0.04;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial color="#00d4ff" transparent opacity={0.7} />
    </instancedMesh>
  );
}

// Wave ring animation
function WaveRings() {
  const ringsRef = useRef<THREE.Group>(null!);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (ringsRef.current) {
      ringsRef.current.children.forEach((child, i) => {
        const ring = child as THREE.Mesh;
        const phase = (t * 0.5 + i * 0.4) % 3;
        const scale = phase * 1.5;
        ring.scale.setScalar(scale);
        (ring.material as THREE.MeshBasicMaterial).opacity = Math.max(
          0,
          0.4 * (1 - phase / 3)
        );
      });
    }
  });

  return (
    <group ref={ringsRef} position={[0, 0.5, 0]}>
      {[0, 1, 2, 3, 4].map((i) => (
        <mesh key={i} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.95, 1, 64]} />
          <meshBasicMaterial
            color="#00d4ff"
            transparent
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

// IFA Antenna Geometry
function AntennaModel() {
  // Scale: convert mm to scene units (1 unit = 10mm)
  const s = 1 / 10;

  return (
    <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.3}>
      <group rotation={[0, 0, 0]} position={[0, -0.5, 0]}>
        {/* Ground plane (bottom PEC) */}
        <mesh position={[-0.787 * s / 2, 0.8 * s, 2 * s]}>
          <boxGeometry
            args={[0.787 * s, 16 * s, 40 * s]}
          />
          <meshStandardMaterial
            color="#1a2744"
            metalness={0.9}
            roughness={0.15}
          />
        </mesh>

        {/* Substrate (dielectric) */}
        <mesh position={[-0.787 * s / 2, 2 * s, 2 * s]}>
          <boxGeometry args={[0.787 * s, 40 * s, 40 * s]} />
          <MeshTransmissionMaterial
            backside
            samples={4}
            thickness={0.2}
            chromaticAberration={0.05}
            anisotropy={0.1}
            distortion={0}
            distortionScale={0}
            temporalDistortion={0}
            color="#2d5a27"
            transmission={0.6}
            roughness={0.3}
          />
        </mesh>

        {/* Feed patch (small PEC on top) */}
        <mesh position={[0, 1.4 * s, 2.5 * s]}>
          <boxGeometry args={[0.02, 2.84 * s, 2.4 * s]} />
          <meshStandardMaterial
            color="#ff6b35"
            metalness={0.95}
            roughness={0.1}
            emissive="#ff6b35"
            emissiveIntensity={0.3}
          />
        </mesh>

        {/* Top radiating arm */}
        <mesh position={[0, 2.72 * s, 2.04 * s]}>
          <boxGeometry args={[0.02, 2.4 * s, 24 * s * 0.5]} />
          <meshStandardMaterial
            color="#e8a838"
            metalness={0.95}
            roughness={0.1}
            emissive="#e8a838"
            emissiveIntensity={0.2}
          />
        </mesh>

        {/* Vertical shorting arm */}
        <mesh position={[-0.39 * s, 1.84 * s, 3.12 * s]}>
          <boxGeometry args={[0.787 * s, 1.2 * s, 0.24 * s]} />
          <meshStandardMaterial
            color="#e8a838"
            metalness={0.95}
            roughness={0.1}
            emissive="#e8a838"
            emissiveIntensity={0.2}
          />
        </mesh>

        {/* Feed point indicator (glowing sphere) */}
        <mesh position={[0, 0, 2.5 * s]}>
          <sphereGeometry args={[0.08, 16, 16]} />
          <meshStandardMaterial
            color="#00d4ff"
            emissive="#00d4ff"
            emissiveIntensity={2}
          />
        </mesh>
      </group>
    </Float>
  );
}

// Grid floor
function GridFloor() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
      <planeGeometry args={[20, 20, 20, 20]} />
      <meshBasicMaterial
        color="#0a1628"
        wireframe
        transparent
        opacity={0.15}
      />
    </mesh>
  );
}

export default function AntennaScene() {
  return (
    <div className="antenna-canvas-container">
      <Canvas
        camera={{ position: [3, 2.5, 4], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <color attach="background" args={["#060d1a"]} />
        <fog attach="fog" args={["#060d1a", 8, 18]} />

        {/* Lighting */}
        <ambientLight intensity={0.15} />
        <pointLight position={[5, 5, 5]} intensity={1} color="#4fc3f7" />
        <pointLight position={[-3, 3, -3]} intensity={0.5} color="#ff6b35" />
        <spotLight
          position={[0, 8, 0]}
          angle={0.3}
          penumbra={1}
          intensity={0.4}
          color="#ffffff"
        />

        {/* Scene */}
        <AntennaModel />
        <EMParticles count={250} />
        <WaveRings />
        <GridFloor />

        {/* Controls */}
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          autoRotate
          autoRotateSpeed={0.8}
          minDistance={3}
          maxDistance={12}
        />
        <Environment preset="night" />
      </Canvas>

      {/* Overlay label */}
      <div className="scene-label">
        <span className="scene-badge">LIVE</span>
        Inverted-F Antenna — 3D Model
      </div>
    </div>
  );
}
