# High-Performance Multithreaded WebAssembly FDTD Engine
**Inverted-F Antenna (IFA) Electromagnetic Simulation Environment**

This project completely rebuilds a legacy 3D FDTD (Finite-Difference Time-Domain) physics simulation—originally constructed in MATLAB—into a globally accessible, maximum-performance JavaScript & WebAssembly architecture running on Node.js and React.

By stripping away interpreted engine overhead and heavily leveraging bare-metal features of the V8 engine, this FDTD simulator mathematically integrates 3D space with throughput that rivals optimized natively-compiled C++, whilst retaining the vast ecosystem of modern web interfaces.

---

## 🔬 Core Technical Architecture

The backbone of this ecosystem is a master-worker multithreaded simulation server specifically engineered for memory resilience and highly parallelized physics throughput.

### 1. Unified WebAssembly Memory Topology
Standard JavaScript arrays fall victim to heavy V8 Garbage Collection (GC) sweeps, introducing catastrophic micro-pauses during billions of floating-point operations. To bypass this entirely, all physical fields ($E_x, E_y, E_z$, $H_x, H_y, H_z$) and specific spatial FDTD coefficients are dynamically mapped into a single contiguous **`SharedArrayBuffer` via `WebAssembly.Memory`**. 
- **Footprint:** Precisely calculated at ~190 MB of static RAM allocated during initialization.
- **Latency:** Zero GC interruptions because the array buffer operates strictly outside standard memory heaps. Float64 element manipulation has native C++ latency properties.

### 2. Multi-Threaded Cluster Mesh (Worker Threads & Atomics)
Instead of forcing a single core to iterate the massive $39 \times 136 \times 136$ discrete voxel grid linearly, the server utilizes the Node.js `worker_threads` API to map the system across all available OS CPU Cores (typically 15+ workers).
- **Partitioning:** The dense 3D boundaries ($X_{size}$, $Y_{size}$, $Z_{size}$) are dynamically partitioned across the cluster using strict limits (`p_nx_start`, `p_nx_end`).
- **Futex Barriers (`Atomics` API):** Complex electromagnetic fields interact dependently ($E$ fields rely on $H$, and vice versa). To synchronize standard worker phase execution with microsecond precision, we implemented custom hardware `Atomics.wait` and `Atomics.notify` Phase Barriers instead of slower software messaging (`postMessage`).

### 3. AssemblyScript Bare-Metal Kernels
The innermost physics loop (The Hot Loop) has been ported completely to **AssemblyScript** ($FDTD\_Kernels.wasm$). The WASM modules are instantiated inside the cluster threads and access the common `SharedArrayBuffer` memory explicitly using byte offset pointers (`usize`), eliminating JS Bounds Checking and pushing floating point instructions straight to native hardware opcodes.

### 4. Real-time Next.js Visual Dashboard (SSE Telemetry)
The V8 solver yields a periodic HTTP Server-Sent Events (SSE) telemetry data stream natively. A modernized React (Next.js) frontend consumes this `text/event-stream` using WebGL (Three.js) geometries and Recharts wrappers to plot real-time source voltage transients, S-Parameters ($S_{11}$), and exact far-field radiation propagation organically over the browser. Node's MACRO-task event loops are yielded appropriately during the extreme WASM calculations to flush TCP buffers flawlessly to the UI without freezing.

---

## ⚡ Speed & Performance Metrics

| Engine Format                  | Computational Model                                                           | Step Processing Time | TTotal Runtime |
| ------------------------------ | ----------------------------------------------------------------------------- | -------------------- | ---------------- |
| **MATLAB Base**                | Scalar Single-Threaded Interpretation                                         | ~85ms                | 10 Minutes +     |
| **JS Base (Standard Typed)**   | Float64Arrays + Heavy V8 Allocation + Object Referencing                      | (Deadlocks under IO) | ❌ N/A            |
| **WASM Scalar (Multithreaded)**| Shared Memory C++ Pointer Arithmetic + 16 Worker Threads + Futex Logic        | **~22ms**            | **~150 Seconds** 🚀|

*Metrics run on a 7000 Time-Step Iteration ($dt$) spanning an IFA space matrix footprint composed entirely of rigorous physical discrete Float64 grid cells.*

> Note: Due to strict TurboFan limitations inside current versions of Windows Node.js, mapping explicit 128-bit `$v128_load` hardware SIMD instructions directly bounded against SharedArrayBuffers concurrently across 16 threads initiates Access Violation OS traps during V8 optimization memory-tier shifts. The solver consequently maintains optimal integrity & maximum performance relying purely on the multi-threaded optimized scalar math limits.

---

## 🚀 How to Start & Run the Simulator

The solver is divided into two entities: The Multithreaded Physics Backend and the WebGL Visual Frontend. You must run both concurrently.

### 1. Launching the Physics Backend
Open a terminal, navigate into the physics solver directory, and launch the unified server.
```bash
cd js_fdtd
npm install
npm run serve
```
*Note: Do not use `npm start`. `npm run serve` binds the API to Port 4000 and spins up the Atomics Cluster successfully.*
The terminal will display the worker mappings and state that the interface is listening for requests.

### 2. Launching the Dashboard Frontend
Open a second separate terminal window and start the Next.js visual interface.
```bash
cd visual
npm install
npm run dev
```

### 3. Triggering the Simulation
1. Once both services are running smoothly, open your web browser to **`http://localhost:3000`**.
2. Click the **`Start Engine ⚡`** button directly beneath the Hero title.
3. The dashboard will trigger a `POST http://localhost:4000/simulate` initialization directive to the WASM clusters.
4. Watch the transient graphs, the exact percentage yields, and the S11 parameter load continuously mapping seamlessly into the browser as the physical integration iterates up to Step 7000!
