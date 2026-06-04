# package / build notes — CUDA backend

This backend is intentionally **decoupled from the default install/build** so
that machines without an NVIDIA GPU or CUDA Toolkit are never broken.

## What was (and was not) changed in package.json

- **Added** one script line:
  ```json
  "build:cuda": "node-gyp configure build --directory=src/backends/cuda"
  ```
- **Not added** to `dependencies` or `devDependencies`: `node-gyp` and
  `node-addon-api`. They are only needed to build this optional addon. Install
  them on demand:
  ```
  npm install --no-save node-gyp node-addon-api
  ```
  If you want them committed for a CUDA-targeting deployment, add them to
  `devDependencies` yourself — but do **not** wire the native build into the
  default `build`/`postinstall`, or installs on GPU-less machines will fail.

## Build commands

```
# from js_fdtd/
npm install --no-save node-gyp node-addon-api   # build-time only deps
npm run build:cuda                              # configure + build the .node
```

Output: `src/backends/cuda/build/Release/fdtd_cuda.node`.
`cudaBackend.js` searches `build/Release`, `build/Debug`, and
`native/build/Release` for it.

## Toolchain prerequisites

| Platform | Needs |
|----------|-------|
| Windows  | CUDA Toolkit (sets `CUDA_PATH`); Visual Studio Build Tools (MSVC v143, C++); `nvcc.exe` on PATH |
| Linux    | CUDA Toolkit (`CUDA_HOME` or `/usr/local/cuda`); matching GCC; `nvcc` on PATH |
| macOS    | Not supported (no modern CUDA). Target is a no-op; backend reports unavailable. |

## Override GPU architecture

`binding.gyp` defaults to `sm_75 sm_86 sm_89`. Override:

```
node-gyp configure build --directory=src/backends/cuda \
  --cuda_arch="-gencode=arch=compute_70,code=sm_70"
```

## Validate

```
node test/validate_cuda.js
```

Skips cleanly (exit 0) when the addon is absent or no GPU is present.

## Validated vs TODO

- Validated by design: voltage trace vs the wasm-cpu golden reference (f32 rtol
  1e-3) via `test/validate_cuda.js`. **Note:** this could not be executed in the
  authoring environment (no CUDA Toolkit / GPU), so the build and numeric match
  are *unverified here*; the kernels are a line-by-line transliteration of the
  reference.
- TODO: GPU far-field DFT (currently CPU fallback), current sources, inductors,
  diodes. See README.md.
