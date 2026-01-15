import math
import cupy as cp

# ---- CUDA kernels (float32 / float64) ----
_kv_f32_src = r'''
extern "C" __device__ float sqrtf(float);
extern "C" __device__ float expf(float);
extern "C" __device__ float powf(float, float);

extern "C" __global__
void kv_realpos_f32(const float* x, float* out, int n,
                    float v, float sinpv, float gamma_vp1, float gamma_1mv,
                    float z_switch, int series_terms, int asym_terms)
{
    int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    float z = x[i];
    if (z == 0.0f) { out[i] = 1.0f/0.0f; return; }      // +inf
    if (!(z > 0.0f)) { out[i] = 0.0f/0.0f; return; }    // NaN

    const float pi = 3.14159265358979323846f;

    if (z > z_switch) {
        float mu = 4.0f * v * v;
        float term = 1.0f;
        float s = 1.0f;
        for (int k = 1; k <= asym_terms; ++k) {
            float a = (mu - (2.0f*k - 1.0f)*(2.0f*k - 1.0f)) / (k * 8.0f);
            term = term * (a / z);
            s += term;
        }
        out[i] = sqrtf(pi / (2.0f * z)) * expf(-z) * s;
        return;
    }

    float t  = 0.25f * z * z;   // (z^2)/4
    float z2 = 0.5f  * z;       // z/2

    float term_p = powf(z2,  v) / gamma_vp1;   // (z/2)^v / Gamma(v+1)
    float term_m = powf(z2, -v) / gamma_1mv;   // (z/2)^-v / Gamma(1-v)

    float I_p = term_p;
    float I_m = term_m;

    for (int k = 0; k < series_terms - 1; ++k) {
        float kp1 = (float)(k + 1);
        term_p *= t / (kp1 * (v + kp1));
        term_m *= t / (kp1 * (kp1 - v));
        I_p += term_p;
        I_m += term_m;
    }

    out[i] = (pi * 0.5f) * (I_m - I_p) / sinpv;
}
'''



_kv_f64_src = r'''
extern "C" __device__ double sqrt(double);
extern "C" __device__ double exp(double);
extern "C" __device__ double pow(double, double);

extern "C" __global__
void kv_realpos_f64(const double* x, double* out, int n,
                    double v, double sinpv, double gamma_vp1, double gamma_1mv,
                    double z_switch, int series_terms, int asym_terms)
{
    int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;

    double z = x[i];
    if (z == 0.0) { out[i] = 1.0/0.0; return; }       // +inf
    if (!(z > 0.0)) { out[i] = 0.0/0.0; return; }     // NaN

    const double pi = 3.141592653589793238462643383279502884;

    if (z > z_switch) {
        double mu = 4.0 * v * v;
        double term = 1.0;
        double s = 1.0;
        for (int k = 1; k <= asym_terms; ++k) {
            double a = (mu - (2.0*k - 1.0)*(2.0*k - 1.0)) / (k * 8.0);
            term = term * (a / z);
            s += term;
        }
        out[i] = sqrt(pi / (2.0 * z)) * exp(-z) * s;
        return;
    }

    double t  = 0.25 * z * z;
    double z2 = 0.5  * z;

    double term_p = pow(z2,  v) / gamma_vp1;
    double term_m = pow(z2, -v) / gamma_1mv;

    double I_p = term_p;
    double I_m = term_m;

    for (int k = 0; k < series_terms - 1; ++k) {
        double kp1 = (double)(k + 1);
        term_p *= t / (kp1 * (v + kp1));
        term_m *= t / (kp1 * (kp1 - v));
        I_p += term_p;
        I_m += term_m;
    }

    out[i] = (pi * 0.5) * (I_m - I_p) / sinpv;
}
'''


_kv_f32 = cp.RawKernel(_kv_f32_src, "kv_realpos_f32")
_kv_f64 = cp.RawKernel(_kv_f64_src, "kv_realpos_f64")


def kv_realpos(v: float, x, *, z_switch=25.0, series_terms=60, asym_terms=12):
    """
    K_v(x) for real order v (scalar) and real positive x (CuPy array).
    Fast path suitable for von Kármán usage.

    Notes:
      - v must NOT be an integer (reflection formula singular). v=5/6 is fine.
      - x must be >0. x==0 -> inf, x<0 -> NaN.
    """
    v = float(abs(v))
    x = cp.asarray(x)

    # force float32/float64 output matching x
    if x.dtype not in (cp.float32, cp.float64):
        x = x.astype(cp.float32)

    out = cp.empty_like(x)

    sinpv = math.sin(math.pi * v)
    if abs(sinpv) < 1e-12:
        raise ValueError("v is (near) an integer; this fast real-x implementation uses a reflection formula.")

    gamma_vp1 = math.gamma(v + 1.0)
    gamma_1mv = math.gamma(1.0 - v)

    threads = 256
    n = x.size
    blocks = (n + threads - 1) // threads

    if x.dtype == cp.float64:
        _kv_f64((blocks,), (threads,),
                (x, out, n,
                 float(v), float(sinpv), float(gamma_vp1), float(gamma_1mv),
                 float(z_switch), int(series_terms), int(asym_terms)))
    else:
        _kv_f32((blocks,), (threads,),
                (x, out, n,
                 cp.float32(v), cp.float32(sinpv), cp.float32(gamma_vp1), cp.float32(gamma_1mv),
                 cp.float32(z_switch), cp.int32(series_terms), cp.int32(asym_terms)))
    return out