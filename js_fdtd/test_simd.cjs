const fs = require('fs');
const { performance } = require('perf_hooks');

const buf = fs.readFileSync('build/fdtd_kernels.wasm');
WebAssembly.compile(buf).then(async m => {
  const mem = new WebAssembly.Memory({initial: 3000, maximum: 65536, shared: true});
  const inst = await WebAssembly.instantiate(m, {
    env: {
      memory: mem,
      abort: () => console.log('abort')
    }
  });

  const w = inst.exports;

  // nx=39, ny=136, nz=136
  const nx=39, ny=136, nz=136, nxp1=40, nyp1=137, nzp1=137;
  
  // mock pointers
  const p = {
    Hx: 1000000, Hy: 2000000, Hz: 3000000,
    Ex: 4000000, Ey: 5000000, Ez: 6000000,
    Chxh: 7000000, Chxey: 8000000, Chxez: 9000000,
    Chyh: 10000000, Chyez: 11000000, Chyex: 12000000,
    Chzh: 13000000, Chzex: 14000000, Chzey: 15000000,
    Cexe: 16000000, Cexhz: 17000000, Cexhy: 18000000,
    Ceye: 19000000, Ceyhx: 20000000, Ceyhz: 21000000,
    Ceze: 22000000, Cezhy: 23000000, Cezhx: 24000000
  };

  const t0 = performance.now();
  
  w.updateE(
    nx, ny, nz, nxp1, nyp1, nzp1,
    0, nx, 0, nxp1,
    p.Ex, p.Ey, p.Ez,
    p.Hx, p.Hy, p.Hz,
    p.Cexe, p.Cexhz, p.Cexhy,
    p.Ceye, p.Ceyhx, p.Ceyhz,
    p.Ceze, p.Cezhy, p.Cezhx
  );

  console.log('Finished updateE in', performance.now() - t0, 'ms');
}).catch(console.error);
