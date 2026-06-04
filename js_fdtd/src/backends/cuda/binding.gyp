{
  # node-gyp build config for the CUDA FDTD native addon.
  #
  # Two translation units:
  #   native/binding.cpp   -> host C++ (MSVC on Windows, g++ on Linux), node-addon-api
  #   native/fdtd_cuda.cu  -> compiled by nvcc into an object, then linked in
  #
  # node-gyp does not know about .cu files, so we compile fdtd_cuda.cu with an
  # explicit nvcc action and feed the resulting object to the linker.
  #
  # Prerequisites: CUDA Toolkit installed and `nvcc` on PATH; CUDA_PATH env var
  # set (the Windows installer sets it). See README.md / package-notes.md.

  "variables": {
    # Allow override: `CUDA_PATH=/usr/local/cuda node-gyp configure`
    "cuda_root%": "<!(node -e \"process.stdout.write(process.env.CUDA_PATH || process.env.CUDA_HOME || (process.platform==='win32' ? '' : '/usr/local/cuda'))\")",
    # nvcc gencode — Turing/Ampere/Ada are common; adjust for your GPU.
    "cuda_arch%": "-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89"
  },

  "targets": [
    {
      "target_name": "fdtd_cuda",
      "sources": [ "native/binding.cpp" ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<(cuda_root)/include",
        "native"
      ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS=0", "NAPI_VERSION=8" ],
      "cflags_cc!": [ "-fno-exceptions" ],

      # The nvcc-compiled object is produced by the action below and linked here.
      "conditions": [
        ["OS=='win'", {
          "libraries": [
            "<(cuda_root)/lib/x64/cudart.lib",
            "<(INTERMEDIATE_DIR)/fdtd_cuda_kernels.obj"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": { "ExceptionHandling": 1, "AdditionalOptions": [ "/std:c++17" ] }
          },
          "actions": [
            {
              "action_name": "nvcc_compile",
              "inputs": [ "native/fdtd_cuda.cu" ],
              "outputs": [ "<(INTERMEDIATE_DIR)/fdtd_cuda_kernels.obj" ],
              "message": "Compiling fdtd_cuda.cu with nvcc (Windows)",
              "action": [
                "<(cuda_root)/bin/nvcc.exe",
                "-c", "native/fdtd_cuda.cu",
                "-o", "<(INTERMEDIATE_DIR)/fdtd_cuda_kernels.obj",
                "-Xcompiler", "/MT,/EHsc",
                "-std=c++17",
                "<@(cuda_arch)",
                "-O3"
              ]
            }
          ]
        }],
        ["OS=='linux'", {
          "libraries": [
            "-L<(cuda_root)/lib64", "-lcudart",
            "<(INTERMEDIATE_DIR)/fdtd_cuda_kernels.o"
          ],
          "cflags_cc": [ "-std=c++17", "-fexceptions" ],
          "actions": [
            {
              "action_name": "nvcc_compile",
              "inputs": [ "native/fdtd_cuda.cu" ],
              "outputs": [ "<(INTERMEDIATE_DIR)/fdtd_cuda_kernels.o" ],
              "message": "Compiling fdtd_cuda.cu with nvcc (Linux)",
              "action": [
                "<(cuda_root)/bin/nvcc",
                "-c", "native/fdtd_cuda.cu",
                "-o", "<(INTERMEDIATE_DIR)/fdtd_cuda_kernels.o",
                "-Xcompiler", "-fPIC",
                "-std=c++17",
                "<@(cuda_arch)",
                "-O3"
              ]
            }
          ]
        }],
        ["OS=='mac'", {
          # CUDA is not supported on modern macOS; this target will not build.
          # isAvailable() in cudaBackend.js returns false and the app falls back.
        }]
      ]
    }
  ]
}
