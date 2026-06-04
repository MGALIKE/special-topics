// kernels.wgsl.js
// ---------------------------------------------------------------------------
// WGSL compute-shader sources for the WebGPU FDTD backend.
//
// Every shader reproduces, *byte-for-byte*, the index math and arithmetic of
// the golden CPU/WASM reference:
//   - bulk H/E updates  -> assembly/fdtd_kernels.ts  (updateH / updateE)
//   - CPML updates       -> src/cpml.js              (updateMagneticCPML / updateElectricCPML)
//   - source injection   -> src/sources.js           (injectVoltageSources)
//   - voltage sampling    -> src/sampling.js          (captureSampledVoltages)
//   - current sampling    -> src/sampling.js          (captureSampledCurrents)
//
// PRECISION NOTE: WGSL `f32` is used everywhere. The reference is f64. Per the
// CONTRACT this is acceptable (validate within ~1e-3 rtol / ~0.1 dB). Fields and
// all 18 coefficients are cast Float64Array -> Float32Array on upload by engine.js.
//
// LAYOUT NOTE: All arrays are flat, row-major, identical strides to the CPU
// reference. See CONTRACT.md §1 for the index helpers reproduced below.
//
// A single uniform block `Dims` carries the grid dimensions. It is bound at
// @group(0) @binding(0) in every shader. Per-dispatch scalars that change
// between dispatches (CPML face params, the per-step source value, the sample
// time index) ride in a second small uniform block `Params` at binding 1 of the
// relevant shaders, so the big storage bindings never need rebinding.
// ---------------------------------------------------------------------------

// Shared uniform struct + index helpers, prepended to every shader.
const COMMON = /* wgsl */ `
struct Dims {
  nx   : u32,
  ny   : u32,
  nz   : u32,
  nxp1 : u32,
  nyp1 : u32,
  nzp1 : u32,
  // sizes (element counts) of each field component, handy for bounds.
  nhx  : u32,
  nhy  : u32,
  nhz  : u32,
  nex  : u32,
  ney  : u32,
  nez  : u32,
};

// 0-based row-major index helpers — identical to src/grid.js.
fn idxHx(i:u32,j:u32,k:u32,ny:u32,nz:u32)     -> u32 { return i*ny*nz     + j*nz   + k; }
fn idxHy(i:u32,j:u32,k:u32,nyp1:u32,nz:u32)   -> u32 { return i*nyp1*nz   + j*nz   + k; }
fn idxHz(i:u32,j:u32,k:u32,ny:u32,nzp1:u32)   -> u32 { return i*ny*nzp1   + j*nzp1 + k; }
fn idxEx(i:u32,j:u32,k:u32,nyp1:u32,nzp1:u32) -> u32 { return i*nyp1*nzp1 + j*nzp1 + k; }
fn idxEy(i:u32,j:u32,k:u32,ny:u32,nzp1:u32)   -> u32 { return i*ny*nzp1   + j*nzp1 + k; }
fn idxEz(i:u32,j:u32,k:u32,nyp1:u32,nz:u32)   -> u32 { return i*nyp1*nz   + j*nz   + k; }
`;

// ---------------------------------------------------------------------------
// updateH — magnetic bulk update. One invocation per (linearized) cell, three
// passes (Hx, Hy, Hz) selected by a `pass` value in Params, because the three
// components have different sizes/loop bounds. engine.js dispatches it 3 times.
//
//   Hx = Chxh*Hx + Chxey*(Ey[k+1]-Ey[k]) + Chxez*(Ez[j+1]-Ez[j])
//   Hy = Chyh*Hy + Chyez*(Ez[i+1]-Ez[i]) + Chyex*(Ex[k+1]-Ex[k])
//   Hz = Chzh*Hz + Chzex*(Ex[j+1]-Ex[j]) + Chzey*(Ey[i+1]-Ey[i])
//
// Bounds: Hx i∈[0,nxp1) j∈[0,ny) k∈[0,nz)
//         Hy i∈[0,nx)   j∈[0,nyp1) k∈[0,nz)
//         Hz i∈[0,nx)   j∈[0,ny)   k∈[0,nzp1)
// ---------------------------------------------------------------------------
export const updateH_wgsl = COMMON + /* wgsl */ `
struct HParams { passIdx : u32, _p0:u32, _p1:u32, _p2:u32, };
@group(0) @binding(0) var<uniform> d : Dims;
@group(0) @binding(1) var<uniform> pr : HParams;

@group(0) @binding(2)  var<storage, read_write> Hx : array<f32>;
@group(0) @binding(3)  var<storage, read_write> Hy : array<f32>;
@group(0) @binding(4)  var<storage, read_write> Hz : array<f32>;
@group(0) @binding(5)  var<storage, read>       Ex : array<f32>;
@group(0) @binding(6)  var<storage, read>       Ey : array<f32>;
@group(0) @binding(7)  var<storage, read>       Ez : array<f32>;
@group(0) @binding(8)  var<storage, read>       Chxh  : array<f32>;
@group(0) @binding(9)  var<storage, read>       Chxey : array<f32>;
@group(0) @binding(10) var<storage, read>       Chxez : array<f32>;
@group(0) @binding(11) var<storage, read>       Chyh  : array<f32>;
@group(0) @binding(12) var<storage, read>       Chyez : array<f32>;
@group(0) @binding(13) var<storage, read>       Chyex : array<f32>;
@group(0) @binding(14) var<storage, read>       Chzh  : array<f32>;
@group(0) @binding(15) var<storage, read>       Chzex : array<f32>;
@group(0) @binding(16) var<storage, read>       Chzey : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = gid.x;
  if (pr.passIdx == 0u) {
    // Hx over (nxp1, ny, nz)
    if (g >= d.nhx) { return; }
    let i = g / (d.ny * d.nz);
    let r = g % (d.ny * d.nz);
    let j = r / d.nz;
    let k = r % d.nz;
    let eey1 = Ey[idxEy(i, j, k+1u, d.ny, d.nzp1)];
    let eey0 = Ey[idxEy(i, j, k,    d.ny, d.nzp1)];
    let eez1 = Ez[idxEz(i, j+1u, k, d.nyp1, d.nz)];
    let eez0 = Ez[idxEz(i, j,    k, d.nyp1, d.nz)];
    Hx[g] = Chxh[g]*Hx[g] + Chxey[g]*(eey1 - eey0) + Chxez[g]*(eez1 - eez0);
  } else if (pr.passIdx == 1u) {
    // Hy over (nx, nyp1, nz)
    if (g >= d.nhy) { return; }
    let i = g / (d.nyp1 * d.nz);
    let r = g % (d.nyp1 * d.nz);
    let j = r / d.nz;
    let k = r % d.nz;
    let eez1 = Ez[idxEz(i+1u, j, k, d.nyp1, d.nz)];
    let eez0 = Ez[idxEz(i,    j, k, d.nyp1, d.nz)];
    let eex1 = Ex[idxEx(i, j, k+1u, d.nyp1, d.nzp1)];
    let eex0 = Ex[idxEx(i, j, k,    d.nyp1, d.nzp1)];
    Hy[g] = Chyh[g]*Hy[g] + Chyez[g]*(eez1 - eez0) + Chyex[g]*(eex1 - eex0);
  } else {
    // Hz over (nx, ny, nzp1)
    if (g >= d.nhz) { return; }
    let i = g / (d.ny * d.nzp1);
    let r = g % (d.ny * d.nzp1);
    let j = r / d.nzp1;
    let k = r % d.nzp1;
    let eex1 = Ex[idxEx(i, j+1u, k, d.nyp1, d.nzp1)];
    let eex0 = Ex[idxEx(i, j,    k, d.nyp1, d.nzp1)];
    let eey1 = Ey[idxEy(i+1u, j, k, d.ny, d.nzp1)];
    let eey0 = Ey[idxEy(i,    j, k, d.ny, d.nzp1)];
    Hz[g] = Chzh[g]*Hz[g] + Chzex[g]*(eex1 - eex0) + Chzey[g]*(eey1 - eey0);
  }
}
`;

// ---------------------------------------------------------------------------
// updateE — electric bulk update. Interior only; boundary planes stay 0 and are
// handled by CPML. Three passes selected by Params.pass.
//
//   Ex = Cexe*Ex + Cexhz*(Hz[j]-Hz[j-1]) + Cexhy*(Hy[k]-Hy[k-1])
//   Ey = Ceye*Ey + Ceyhx*(Hx[k]-Hx[k-1]) + Ceyhz*(Hz[i]-Hz[i-1])
//   Ez = Ceze*Ez + Cezhy*(Hy[i]-Hy[i-1]) + Cezhx*(Hx[j]-Hx[j-1])
//
// Bounds: Ex i∈[0,nx)   j∈[1,ny)   k∈[1,nz)
//         Ey i∈[1,nxp1) j∈[0,ny)   k∈[1,nz)
//         Ez i∈[1,nxp1) j∈[1,nyp1) k∈[0,nz)
// We launch one invocation per element and skip those outside the interior box.
// ---------------------------------------------------------------------------
export const updateE_wgsl = COMMON + /* wgsl */ `
struct EParams { passIdx : u32, _p0:u32, _p1:u32, _p2:u32, };
@group(0) @binding(0) var<uniform> d : Dims;
@group(0) @binding(1) var<uniform> pr : EParams;

@group(0) @binding(2)  var<storage, read_write> Ex : array<f32>;
@group(0) @binding(3)  var<storage, read_write> Ey : array<f32>;
@group(0) @binding(4)  var<storage, read_write> Ez : array<f32>;
@group(0) @binding(5)  var<storage, read>       Hx : array<f32>;
@group(0) @binding(6)  var<storage, read>       Hy : array<f32>;
@group(0) @binding(7)  var<storage, read>       Hz : array<f32>;
@group(0) @binding(8)  var<storage, read>       Cexe  : array<f32>;
@group(0) @binding(9)  var<storage, read>       Cexhz : array<f32>;
@group(0) @binding(10) var<storage, read>       Cexhy : array<f32>;
@group(0) @binding(11) var<storage, read>       Ceye  : array<f32>;
@group(0) @binding(12) var<storage, read>       Ceyhx : array<f32>;
@group(0) @binding(13) var<storage, read>       Ceyhz : array<f32>;
@group(0) @binding(14) var<storage, read>       Ceze  : array<f32>;
@group(0) @binding(15) var<storage, read>       Cezhy : array<f32>;
@group(0) @binding(16) var<storage, read>       Cezhx : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = gid.x;
  if (pr.passIdx == 0u) {
    // Ex over (nx, nyp1, nzp1); interior j∈[1,ny) k∈[1,nz)
    if (g >= d.nex) { return; }
    let i = g / (d.nyp1 * d.nzp1);
    let r = g % (d.nyp1 * d.nzp1);
    let j = r / d.nzp1;
    let k = r % d.nzp1;
    if (i >= d.nx || j < 1u || j >= d.ny || k < 1u || k >= d.nz) { return; }
    let hhz1 = Hz[idxHz(i, j,    k, d.ny, d.nzp1)];
    let hhz0 = Hz[idxHz(i, j-1u, k, d.ny, d.nzp1)];
    let hhy1 = Hy[idxHy(i, j, k,    d.nyp1, d.nz)];
    let hhy0 = Hy[idxHy(i, j, k-1u, d.nyp1, d.nz)];
    Ex[g] = Cexe[g]*Ex[g] + Cexhz[g]*(hhz1 - hhz0) + Cexhy[g]*(hhy1 - hhy0);
  } else if (pr.passIdx == 1u) {
    // Ey over (nxp1, ny, nzp1); interior i∈[1,nxp1) k∈[1,nz)
    if (g >= d.ney) { return; }
    let i = g / (d.ny * d.nzp1);
    let r = g % (d.ny * d.nzp1);
    let j = r / d.nzp1;
    let k = r % d.nzp1;
    if (i < 1u || i >= d.nxp1 || j >= d.ny || k < 1u || k >= d.nz) { return; }
    let hhx1 = Hx[idxHx(i, j, k,    d.ny, d.nz)];
    let hhx0 = Hx[idxHx(i, j, k-1u, d.ny, d.nz)];
    let hhz1 = Hz[idxHz(i,    j, k, d.ny, d.nzp1)];
    let hhz0 = Hz[idxHz(i-1u, j, k, d.ny, d.nzp1)];
    Ey[g] = Ceye[g]*Ey[g] + Ceyhx[g]*(hhx1 - hhx0) + Ceyhz[g]*(hhz1 - hhz0);
  } else {
    // Ez over (nxp1, nyp1, nz); interior i∈[1,nxp1) j∈[1,nyp1)
    if (g >= d.nez) { return; }
    let i = g / (d.nyp1 * d.nz);
    let r = g % (d.nyp1 * d.nz);
    let j = r / d.nz;
    let k = r % d.nz;
    if (i < 1u || i >= d.nxp1 || j < 1u || j >= d.nyp1 || k >= d.nz) { return; }
    let hhy1 = Hy[idxHy(i,    j, k, d.nyp1, d.nz)];
    let hhy0 = Hy[idxHy(i-1u, j, k, d.nyp1, d.nz)];
    let hhx1 = Hx[idxHx(i, j,    k, d.ny, d.nz)];
    let hhx0 = Hx[idxHx(i, j-1u, k, d.ny, d.nz)];
    Ez[g] = Ceze[g]*Ez[g] + Cezhy[g]*(hhy1 - hhy0) + Cezhx[g]*(hhx1 - hhx0);
  }
}
`;

// ---------------------------------------------------------------------------
// CPML — magnetic. One shader, parameterized per-face by CPmlParams. engine.js
// dispatches it once per active face (xn,xp,yn,yp,zn,zp). The shader encodes all
// six faces' Psi-update + field-correction math from src/cpml.js, selected by
// `face` (0..5) and `phase` (0 = update Psi for component A, etc.).
//
// To keep one dispatch self-contained we fuse "update Psi" and "correct field"
// into a single invocation grid per (face, component). Because each face touches
// two field components (e.g. xn touches Hy and Hz) with different array shapes,
// we run the shader twice per face — once per component — selected by `comp`.
//
// The Psi recurrence is:   Psi = b*Psi + a*(F1 - F0)
// then the field correction:  Field[fieldIdx] += CPsi[pi]*Psi[pi]
// Both use the *new* Psi, exactly as the reference does (it updates all Psi for
// the face, then adds CPsi*Psi). Since each (pi) maps to a unique field cell on
// a face, doing both in one invocation is equivalent to the reference's two
// separate loops.
//
// Buffers bound:
//   b, a    : 1D arrays length nc (the b_m/a_m for this face)
//   Psi     : the convolution field for (face,comp)
//   CPsi    : the coefficient slice for (face,comp)
//   Field   : the H component being corrected (Hx/Hy/Hz) — read_write
//   Src0/1  : the E (or H) component differenced into Psi — read
//
// Params carry: nc, and the per-component geometry needed to map a linear
// invocation id -> (ci, a, b) loop indices and to compute pi, source idx, and
// destination field idx. See engine.js for how the per-face/per-comp Params are
// filled; the mapping mirrors src/cpml.js exactly.
// ---------------------------------------------------------------------------
//
// Rather than encode all six faces' bespoke index algebra inside one giant WGSL
// switch (which is error-prone), we expose a *generic strided* CPML kernel. The
// reference's per-face loops all have the same shape:
//
//   for ci in [0,nc): for u in [0,nU): for v in [0,nV):
//       pi      = ci*sPi_ci + u*sPi_u + v*sPi_v
//       srcIdx0 = base0 + ci*src_ci + u*src_u + v*src_v
//       srcIdx1 = srcIdx0 + srcDelta        // the (+1) neighbour
//       Psi[pi] = b[ci]*Psi[pi] + a[ci]*(Src[srcIdx1]-Src[srcIdx0])
//       fldIdx  = fbase + ci*fld_ci + u*fld_u + v*fld_v
//       Field[fldIdx] += CPsi[pi]*Psi[pi]
//
// engine.js computes all the strides/bases for each (face,comp) from src/cpml.js
// and uploads them in CPmlParams. This single kernel then serves H and E CPML
// for every face & component. `bUsesElectric` is irrelevant here — engine.js
// simply binds the correct b/a (magnetic vs electric) and Src/Field buffers.

// CPmlParams + the shared (ci,u,v) index decomposition, prepended to both CPML
// kernels below.
const CPML_COMMON = /* wgsl */ `
struct CPmlParams {
  nc      : u32,
  nU      : u32,
  nV      : u32,
  sPi_ci  : u32,
  sPi_u   : u32,
  sPi_v   : u32,
  base0   : u32,   // base flat index into Src for (ci=0,u=0,v=0)
  src_ci  : u32,
  src_u   : u32,
  src_v   : u32,
  srcDelta: u32,   // offset from Src0 to Src1 (the +1 neighbour)
  fbase   : u32,   // base flat index into Field
  fld_ci  : u32,
  fld_u   : u32,
  fld_v   : u32,
  _pad    : u32,
};
`;

// TWO-PASS CPML.
//
// The CPML update is naturally   Psi = b*Psi + a*(Src1-Src0)   then
// Field += CPsi*Psi. Fusing both into one dispatch forced the SAME buffer to be
// bound as read_write (Field) AND read (Src) whenever Field and Src are the same
// component (the YN/YP magnetic faces difference Hx into Hx, etc.). WebGPU
// forbids aliasing a writable and another binding of one buffer in a single
// dispatch ("includes writable usage and another usage in the same
// synchronization scope") — the CPU reference has no such rule, which is why this
// only surfaced on real hardware.
//
// Splitting into two kernels gives each dispatch a single writable buffer:
//   updatePsi : reads Src (+b/a), writes Psi.            (no Field binding)
//   correctField : reads Psi (+CPsi), writes Field.      (no Src binding)
// Compute-pass usage scopes are per-dispatch, so Src(read) in pass A and
// Field(read_write) in pass B never collide even when they alias. Dispatch order
// within the pass (all updatePsi, then all correctField) also reproduces the
// reference exactly: every Psi is updated from the pre-correction field before
// any field correction is applied.
//
// Bindings are NOT contiguous (they keep the original numbering minus the unused
// Dims slot) so the host can build both bind groups from one shared layout map.

export const cpmlUpdatePsi_wgsl = CPML_COMMON + /* wgsl */ `
@group(0) @binding(1) var<uniform> p  : CPmlParams;
@group(0) @binding(2) var<storage, read>        bcoef : array<f32>;  // b_m or b_e (len nc)
@group(0) @binding(3) var<storage, read>        acoef : array<f32>;  // a_m or a_e (len nc)
@group(0) @binding(4) var<storage, read_write>  Psi   : array<f32>;
@group(0) @binding(7) var<storage, read>        Src   : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = p.nc * p.nU * p.nV;
  let g = gid.x;
  if (g >= total) { return; }
  let ci = g / (p.nU * p.nV);
  let r  = g % (p.nU * p.nV);
  let u  = r / p.nV;
  let v  = r % p.nV;

  let pi = ci*p.sPi_ci + u*p.sPi_u + v*p.sPi_v;
  let s0 = p.base0 + ci*p.src_ci + u*p.src_u + v*p.src_v;
  let s1 = s0 + p.srcDelta;
  Psi[pi] = bcoef[ci]*Psi[pi] + acoef[ci]*(Src[s1] - Src[s0]);
}
`;

export const cpmlCorrectField_wgsl = CPML_COMMON + /* wgsl */ `
@group(0) @binding(1) var<uniform> p  : CPmlParams;
@group(0) @binding(4) var<storage, read>        Psi   : array<f32>;
@group(0) @binding(5) var<storage, read>        CPsi  : array<f32>;
@group(0) @binding(6) var<storage, read_write>  Field : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = p.nc * p.nU * p.nV;
  let g = gid.x;
  if (g >= total) { return; }
  let ci = g / (p.nU * p.nV);
  let r  = g % (p.nU * p.nV);
  let u  = r / p.nV;
  let v  = r % p.nV;

  let pi = ci*p.sPi_ci + u*p.sPi_u + v*p.sPi_v;
  let f  = p.fbase + ci*p.fld_ci + u*p.fld_u + v*p.fld_v;
  Field[f] = Field[f] + CPsi[pi]*Psi[pi];
}
`;

// ---------------------------------------------------------------------------
// injectVoltageSource — adds Cexs[n]*v at field_indices[n] for the (single)
// voltage source. `v` (voltage_per_e_field[ts]) and the destination field are
// chosen on the host; one invocation per source field point.
//   Field[ field_indices[n] ] += Cs[n] * v
// ---------------------------------------------------------------------------
export const injectVoltage_wgsl = /* wgsl */ `
struct SrcParams { count : u32, v : f32, _p0:u32, _p1:u32, };
@group(0) @binding(0) var<uniform> p : SrcParams;
@group(0) @binding(1) var<storage, read>        fieldIndices : array<u32>;
@group(0) @binding(2) var<storage, read>        Cs           : array<f32>;
@group(0) @binding(3) var<storage, read_write>  Field        : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = gid.x;
  if (n >= p.count) { return; }
  let fi = fieldIndices[n];
  Field[fi] = Field[fi] + Cs[n] * p.v;
}
`;

// ---------------------------------------------------------------------------
// sampleVoltage — reduces sum_{n} Field[ field_indices[n] ], multiplies by Csvf,
// and writes the scalar into out[ts]. A single-workgroup tree reduction; the
// default problem has a small source patch so one workgroup (<= 256 wide,
// strided) is plenty. out is a per-step f32 buffer of length numberOfTimeSteps.
// ---------------------------------------------------------------------------
export const sampleVoltage_wgsl = /* wgsl */ `
struct SampParams { count : u32, csvf : f32, tsIndex : u32, _p0:u32, };
@group(0) @binding(0) var<uniform> p : SampParams;
@group(0) @binding(1) var<storage, read>        fieldIndices : array<u32>;
@group(0) @binding(2) var<storage, read>        Field        : array<f32>;
@group(0) @binding(3) var<storage, read_write>  outBuf       : array<f32>;

const WG : u32 = 256u;
var<workgroup> scratch : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  var acc : f32 = 0.0;
  var n : u32 = t;
  loop {
    if (n >= p.count) { break; }
    acc = acc + Field[ fieldIndices[n] ];
    n = n + WG;
  }
  scratch[t] = acc;
  workgroupBarrier();
  // tree reduction
  var stride : u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (t < stride) { scratch[t] = scratch[t] + scratch[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (t == 0u) {
    outBuf[p.tsIndex] = p.csvf * scratch[0];
  }
}
`;

// ---------------------------------------------------------------------------
// sampleCurrent — Ampère's-law loop integral around the current observer (see
// src/sampling.js::captureSampledCurrents). The default problem has NO current
// sources/observers, so this path is provided but exercised only if present.
//
// The loop integral is a sum of a handful of H samples with ±dx/dy/dz weights.
// engine.js precomputes, per current observer, a list of (flatIndex, weight)
// pairs into the relevant H component(s) so this kernel is a generic weighted
// reduction over one observer at a time. The H components are concatenated into
// a single readable view per kernel call by binding the right buffer; for a
// loop that spans two H components engine.js issues two accumulating calls.
//
// NOTE: This is a straightforward weighted sum but its correctness against the
// f64 reference has NOT been validated here (no current observers in the default
// problem). Marked accordingly in README. Layout: terms[m] = (idx, weight).
// ---------------------------------------------------------------------------
export const sampleCurrent_wgsl = /* wgsl */ `
struct CurParams { count : u32, tsIndex : u32, accumulate : u32, sign : f32, };
@group(0) @binding(0) var<uniform> p : CurParams;
@group(0) @binding(1) var<storage, read>        termIdx : array<u32>;
@group(0) @binding(2) var<storage, read>        termW   : array<f32>;
@group(0) @binding(3) var<storage, read>        H       : array<f32>;
@group(0) @binding(4) var<storage, read_write>  outBuf  : array<f32>;

const WG : u32 = 256u;
var<workgroup> scratch : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  var acc : f32 = 0.0;
  var n : u32 = t;
  loop {
    if (n >= p.count) { break; }
    acc = acc + termW[n] * H[ termIdx[n] ];
    n = n + WG;
  }
  scratch[t] = acc;
  workgroupBarrier();
  var stride : u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (t < stride) { scratch[t] = scratch[t] + scratch[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (t == 0u) {
    var val = p.sign * scratch[0];
    if (p.accumulate == 1u) { val = val + outBuf[p.tsIndex]; }
    outBuf[p.tsIndex] = val;
  }
}
`;

// Convenience map for engine.js.
export const SHADERS = {
  updateH: updateH_wgsl,
  updateE: updateE_wgsl,
  cpmlUpdatePsi: cpmlUpdatePsi_wgsl,
  cpmlCorrectField: cpmlCorrectField_wgsl,
  injectVoltage: injectVoltage_wgsl,
  sampleVoltage: sampleVoltage_wgsl,
  sampleCurrent: sampleCurrent_wgsl,
};
