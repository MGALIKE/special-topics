// binding.cpp — node-addon-api glue exposing the CUDA FDTD solver to JS.
//
// Exposes:
//   deviceCount()                              -> number (CUDA devices)
//   init(dims)                                 -> void   (allocate device fields)
//   uploadCoeff(which, Float32Array)           -> void
//   uploadCpml(faceId, obj)                    -> void
//   uploadVSource(dir, Int32Array, Float32Array, Float32Array) -> void
//   uploadSVoltage(dir, Int32Array, csvf)      -> void
//   uploadSCurrent(hxObj, hyObj, hzObj)        -> void
//   runBatch(tsStart, count)                   -> { voltage:Float32Array, current:Float32Array }
//   readField(which, Float32Array)             -> void
//   destroy()                                  -> void
//
// All heavy data crosses as typed arrays (zero-copy views into V8 buffers).
// The JS driver (cudaBackend.js) casts the engine's f64 arrays to f32 before
// calling these.

#include <napi.h>
#include <vector>
#include "fdtd_cuda.h"

static void Throw(const Napi::Env& env, const char* ctx) {
    std::string msg = std::string(ctx) + ": " + fdtd_last_error();
    throw Napi::Error::New(env, msg);
}

static Napi::Value DeviceCount(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int c = 0;
    fdtd_device_count(&c); // never throws; 0 if no driver
    return Napi::Number::New(env, c);
}

static Napi::Value Init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Object o = info[0].As<Napi::Object>();
    FdtdDims d;
    d.nx = o.Get("nx").As<Napi::Number>().Int32Value();
    d.ny = o.Get("ny").As<Napi::Number>().Int32Value();
    d.nz = o.Get("nz").As<Napi::Number>().Int32Value();
    d.dx = o.Get("dx").As<Napi::Number>().FloatValue();
    d.dy = o.Get("dy").As<Napi::Number>().FloatValue();
    d.dz = o.Get("dz").As<Napi::Number>().FloatValue();
    d.dt = o.Get("dt").As<Napi::Number>().FloatValue();
    d.numberOfTimeSteps = o.Get("numberOfTimeSteps").As<Napi::Number>().Int32Value();
    if (fdtd_init(&d) != FDTD_OK) Throw(env, "init");
    return env.Undefined();
}

static const float* f32ptr(const Napi::Value& v, int* lenOut) {
    Napi::Float32Array a = v.As<Napi::Float32Array>();
    if (lenOut) *lenOut = (int)a.ElementLength();
    return a.Data();
}
static const int* i32ptr(const Napi::Value& v, int* lenOut) {
    Napi::Int32Array a = v.As<Napi::Int32Array>();
    if (lenOut) *lenOut = (int)a.ElementLength();
    return a.Data();
}

static Napi::Value UploadCoeff(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int which = info[0].As<Napi::Number>().Int32Value();
    int n = 0;
    const float* p = f32ptr(info[1], &n);
    if (fdtd_upload_coeff(which, p, n) != FDTD_OK) Throw(env, "uploadCoeff");
    return env.Undefined();
}

// Helper: pull a Float32Array field from an object, or null if absent/empty.
static const float* optF32(const Napi::Object& o, const char* key, int* lenOut) {
    *lenOut = 0;
    if (!o.Has(key)) return nullptr;
    Napi::Value v = o.Get(key);
    if (v.IsNull() || v.IsUndefined()) return nullptr;
    return f32ptr(v, lenOut);
}

static Napi::Value UploadCpml(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int faceId = info[0].As<Napi::Number>().Int32Value();
    Napi::Object o = info[1].As<Napi::Object>();
    int nc = o.Get("nc").As<Napi::Number>().Int32Value();
    int dummy;
    const float* b_e = optF32(o, "b_e", &dummy);
    const float* a_e = optF32(o, "a_e", &dummy);
    const float* b_m = optF32(o, "b_m", &dummy);
    const float* a_m = optF32(o, "a_m", &dummy);
    int n0,n1,n2,n3;
    const float* c0 = optF32(o, "CPsi0", &n0);
    const float* c1 = optF32(o, "CPsi1", &n1);
    const float* c2 = optF32(o, "CPsi2", &n2);
    const float* c3 = optF32(o, "CPsi3", &n3);
    int m_start  = o.Get("m_start").As<Napi::Number>().Int32Value();
    int e_start  = o.Get("e_start").As<Napi::Number>().Int32Value();
    int ascending= o.Get("ascending").As<Napi::Number>().Int32Value();
    if (fdtd_upload_cpml(faceId, nc, b_e,a_e,b_m,a_m,
            c0,n0, c1,n1, c2,n2, c3,n3,
            m_start, e_start, ascending) != FDTD_OK) Throw(env, "uploadCpml");
    return env.Undefined();
}

static Napi::Value UploadVSource(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int dir = info[0].As<Napi::Number>().Int32Value();
    int cnt=0, ncoef=0, nsteps=0;
    const int*   idx  = i32ptr(info[1], &cnt);
    const float* coef = f32ptr(info[2], &ncoef);
    const float* wave = f32ptr(info[3], &nsteps);
    if (fdtd_upload_vsource(dir, idx, coef, cnt, wave, nsteps) != FDTD_OK) Throw(env, "uploadVSource");
    return env.Undefined();
}

static Napi::Value UploadSVoltage(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int dir = info[0].As<Napi::Number>().Int32Value();
    int cnt=0;
    const int* idx = i32ptr(info[1], &cnt);
    float csvf = info[2].As<Napi::Number>().FloatValue();
    if (fdtd_upload_svoltage(dir, idx, cnt, csvf) != FDTD_OK) Throw(env, "uploadSVoltage");
    return env.Undefined();
}

// Each arg is { idx:Int32Array, w:Float32Array } (may be empty).
static void readGatherList(const Napi::Object& o, const int** idx, const float** w, int* n) {
    *idx=nullptr; *w=nullptr; *n=0;
    if (o.IsNull() || o.IsUndefined()) return;
    if (!o.Has("idx")) return;
    Napi::Int32Array ia = o.Get("idx").As<Napi::Int32Array>();
    Napi::Float32Array wa = o.Get("w").As<Napi::Float32Array>();
    *idx = ia.Data(); *w = wa.Data(); *n = (int)ia.ElementLength();
}

static Napi::Value UploadSCurrent(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const int *hxI,*hyI,*hzI; const float *hxW,*hyW,*hzW; int hxN,hyN,hzN;
    readGatherList(info[0].As<Napi::Object>(), &hxI,&hxW,&hxN);
    readGatherList(info[1].As<Napi::Object>(), &hyI,&hyW,&hyN);
    readGatherList(info[2].As<Napi::Object>(), &hzI,&hzW,&hzN);
    if (fdtd_upload_scurrent(hxI,hxW,hxN, hyI,hyW,hyN, hzI,hzW,hzN) != FDTD_OK) Throw(env, "uploadSCurrent");
    return env.Undefined();
}

static Napi::Value RunBatch(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int tsStart = info[0].As<Napi::Number>().Int32Value();
    int count   = info[1].As<Napi::Number>().Int32Value();
    std::vector<float> sv(count, 0.0f), si(count, 0.0f);
    if (fdtd_run_batch(tsStart, count, sv.data(), si.data()) != FDTD_OK) Throw(env, "runBatch");
    Napi::Float32Array vOut = Napi::Float32Array::New(env, count);
    Napi::Float32Array iOut = Napi::Float32Array::New(env, count);
    memcpy(vOut.Data(), sv.data(), count*sizeof(float));
    memcpy(iOut.Data(), si.data(), count*sizeof(float));
    Napi::Object r = Napi::Object::New(env);
    r.Set("voltage", vOut);
    r.Set("current", iOut);
    return r;
}

static Napi::Value ReadField(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int which = info[0].As<Napi::Number>().Int32Value();
    Napi::Float32Array a = info[1].As<Napi::Float32Array>();
    if (fdtd_read_field(which, a.Data(), (int)a.ElementLength()) != FDTD_OK) Throw(env, "readField");
    return env.Undefined();
}

static Napi::Value Destroy(const Napi::CallbackInfo& info) {
    fdtd_destroy();
    return info.Env().Undefined();
}

static Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
    exports.Set("deviceCount",   Napi::Function::New(env, DeviceCount));
    exports.Set("init",          Napi::Function::New(env, Init));
    exports.Set("uploadCoeff",   Napi::Function::New(env, UploadCoeff));
    exports.Set("uploadCpml",    Napi::Function::New(env, UploadCpml));
    exports.Set("uploadVSource", Napi::Function::New(env, UploadVSource));
    exports.Set("uploadSVoltage",Napi::Function::New(env, UploadSVoltage));
    exports.Set("uploadSCurrent",Napi::Function::New(env, UploadSCurrent));
    exports.Set("runBatch",      Napi::Function::New(env, RunBatch));
    exports.Set("readField",     Napi::Function::New(env, ReadField));
    exports.Set("destroy",       Napi::Function::New(env, Destroy));
    return exports;
}

NODE_API_MODULE(fdtd_cuda, InitModule)
