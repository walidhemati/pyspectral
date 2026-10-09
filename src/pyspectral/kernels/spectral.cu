// Fused pointwise kernels for the pseudo-spectral Navier-Stokes solver (CUDA C, NVRTC).
//
// Data layout: every work field lives in a complex buffer of fsc = n^(d-1) * (n/2+1) modes.
// The same buffer, viewed as real, holds the padded physical field (last axis n+2) used by
// in-place cuFFT R2C/C2R transforms, so the real field stride is 2*fsc.
//
// One RK4 stage = C2R (batched, in place) -> phys kernel (in place) -> R2C (batched, in place)
// -> post kernel. The post kernel evaluates the nonlinear term from the transformed products,
// applies 2/3-rule dealiasing and (3D) the Leray projection, performs the integrating-factor
// RK4 stage update and directly writes the next stage's FFT inputs, so no field is ever
// re-read by a separate "pre" kernel.
//
// Integrating factor: L(k) = -(nu k^2 + alpha [k^2 <= ka2]) + A [klo^2 <= k^2 <= khi^2] is
// separable apart from the band terms, so
// exp(L theta dt) = t[|kx|] t[|ky|] (t[kz]) * (drag? ea : 1) * (band? eb : 1)
// with a 1D table t[m] = exp(-nu m^2 theta dt) (theta = 1 and 1/2): no transcendental math
// per mode, which matters for fp64 on consumer GPUs.

#ifdef USE_DOUBLE
typedef double real;
typedef double2 cplx;
#else
typedef float real;
typedef float2 cplx;
#endif

typedef long long i64;

__device__ __forceinline__ cplx cmk(real x, real y) { cplx c; c.x = x; c.y = y; return c; }
__device__ __forceinline__ cplx cadd(cplx a, cplx b) { return cmk(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ cplx csc(real s, cplx a) { return cmk(s * a.x, s * a.y); }
__device__ __forceinline__ cplx cmuli(cplx a) { return cmk(-a.y, a.x); }   // i * a
__device__ __forceinline__ cplx cmulmi(cplx a) { return cmk(a.y, -a.x); }  // -i * a
__device__ __forceinline__ cplx cfma(real s, cplx a, cplx b) {             // s*a + b
    return cmk(s * a.x + b.x, s * a.y + b.y);
}

struct IF {
    real E, Eh;
};

// Integrating-factor values for one mode.
__device__ __forceinline__ IF if_factors(const real* __restrict__ tab, int nh1, int ax, int ay,
                                         int az, real k2, real ea1, real eah, real eb1, real ebh,
                                         real klo2, real khi2, real ka2) {
    IF f;
    bool drag = k2 <= ka2;
    f.E = tab[ax] * tab[ay] * (drag ? ea1 : (real)1);
    f.Eh = tab[nh1 + ax] * tab[nh1 + ay] * (drag ? eah : (real)1);
    if (az >= 0) {
        f.E *= tab[az];
        f.Eh *= tab[nh1 + az];
    }
    if (k2 >= klo2 && k2 <= khi2) {
        f.E *= eb1;
        f.Eh *= ebh;
    }
    return f;
}

// RK4-IF stage update for one mode/component; returns the next stage state.
__device__ __forceinline__ cplx rk_stage(int stage, real dt, IF f, cplx N, cplx* u0p, cplx* accp) {
    cplx u = *u0p;
    cplx nxt;
    if (stage == 0) {
        *accp = csc(f.E, N);
        nxt = csc(f.Eh, cfma((real)0.5 * dt, N, u));
    } else if (stage == 1) {
        *accp = cfma((real)2 * f.Eh, N, *accp);
        nxt = cfma((real)0.5 * dt, N, csc(f.Eh, u));
    } else if (stage == 2) {
        *accp = cfma((real)2 * f.Eh, N, *accp);
        nxt = cfma(dt * f.Eh, N, csc(f.E, u));
    } else {
        cplx a = cadd(*accp, N);
        nxt = cfma(dt / (real)6, a, csc(f.E, u));
        *u0p = nxt;
    }
    return nxt;
}

// ----------------------------------------------------------------------------------- 2D
// Streamfunction velocity of a vorticity mode, scaled for the unnormalised C2R:
// u_hat = i ky w/k^2, v_hat = -i kx w/k^2.
__device__ __forceinline__ void write_uv2d(cplx w, real kx, real ky, real k2, bool keep,
                                           real scale, cplx* W, i64 idx, i64 fsc) {
    real ik2 = (k2 > 0 && keep) ? scale / k2 : (real)0;
    cplx psi = csc(ik2, w);
    W[idx] = csc(ky, cmuli(psi));
    W[fsc + idx] = csc(kx, cmulmi(psi));
}

extern "C" __global__ void pre2d(const cplx* __restrict__ u0, cplx* W, int n, int nh, i64 fsc,
                                 real scale) {
    i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsc) return;
    int j = (int)(idx % nh);
    int i = (int)(idx / nh);
    int kxi = i < n / 2 ? i : i - n;
    real kx = (real)kxi, ky = (real)j;
    bool keep = (3 * abs(kxi) < n) && (3 * j < n);
    write_uv2d(u0[idx], kx, ky, kx * kx + ky * ky, keep, scale, W, idx, fsc);
}

// physical space, in place: (u, v) -> (v^2 - u^2, u v)
extern "C" __global__ void phys2d(real* W, i64 fsr) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= fsr) return;
    real u = W[p], v = W[fsr + p];
    W[p] = (v - u) * (v + u);
    W[fsr + p] = u * v;
}

extern "C" __global__ void post2d(cplx* W, cplx* u0, cplx* acc, const real* __restrict__ tab,
                                  int n, int nh, i64 fsc, int stage, real dt, real ea1, real eah,
                                  real eb1, real ebh, real klo2, real khi2, real ka2,
                                  real scale) {
    i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsc) return;
    int j = (int)(idx % nh);
    int i = (int)(idx / nh);
    int kxi = i < n / 2 ? i : i - n;
    real kx = (real)kxi, ky = (real)j;
    real k2 = kx * kx + ky * ky;
    bool keep = (3 * abs(kxi) < n) && (3 * j < n);
    cplx N = cmk(0, 0);
    if (keep) {
        cplx A = W[idx], B = W[fsc + idx];
        N = cfma(kx * ky, A, csc(kx * kx - ky * ky, B));
    }
    IF f = if_factors(tab, nh, abs(kxi), j, -1, k2, ea1, eah, eb1, ebh, klo2, khi2, ka2);
    cplx nxt = rk_stage(stage, dt, f, N, u0 + idx, acc + idx);
    write_uv2d(nxt, kx, ky, k2, keep, scale, W, idx, fsc);
}

// ----------------------------------------------------------------------------------- 3D
__device__ __forceinline__ void decode3d(i64 idx, int n, int nh, int& kxi, int& kyi, int& kz) {
    kz = (int)(idx % nh);
    i64 t = idx / nh;
    int j = (int)(t % n);
    int i = (int)(t / n);
    kxi = i < n / 2 ? i : i - n;
    kyi = j < n / 2 ? j : j - n;
}

// write next-stage FFT inputs: u_hat (3) and, for the rotational form, omega_hat = i k x u_hat
__device__ __forceinline__ void write_pre3d(cplx a, cplx b, cplx c, real kx, real ky, real kz,
                                            bool keep, int rot, real scale, cplx* W, i64 idx,
                                            i64 fsc) {
    real s = keep ? scale : (real)0;
    a = csc(s, a);
    b = csc(s, b);
    c = csc(s, c);
    W[idx] = a;
    W[fsc + idx] = b;
    W[2 * fsc + idx] = c;
    if (rot) {
        W[3 * fsc + idx] = cmuli(cadd(csc(ky, c), csc(-kz, b)));
        W[4 * fsc + idx] = cmuli(cadd(csc(kz, a), csc(-kx, c)));
        W[5 * fsc + idx] = cmuli(cadd(csc(kx, b), csc(-ky, a)));
    }
}

extern "C" __global__ void pre3d(const cplx* __restrict__ u0, cplx* W, int n, int nh, i64 fsc,
                                 int rot, real scale) {
    i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsc) return;
    int kxi, kyi, kz;
    decode3d(idx, n, nh, kxi, kyi, kz);
    bool keep = (3 * abs(kxi) < n) && (3 * abs(kyi) < n) && (3 * kz < n);
    write_pre3d(u0[idx], u0[fsc + idx], u0[2 * fsc + idx], (real)kxi, (real)kyi, (real)kz, keep,
                rot, scale, W, idx, fsc);
}

// physical space, in place, Basdevant traceless stress: (u,v,w) -> (uu-q, vv-q, uv, uw, vw)
extern "C" __global__ void phys3d_basd(real* W, i64 fsr) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= fsr) return;
    real u = W[p], v = W[fsr + p], w = W[2 * fsr + p];
    real q = (u * u + v * v + w * w) * ((real)1 / (real)3);
    W[p] = u * u - q;
    W[fsr + p] = v * v - q;
    W[2 * fsr + p] = u * v;
    W[3 * fsr + p] = u * w;
    W[4 * fsr + p] = v * w;
}

// physical space, in place, rotational form: (u, omega) -> u x omega
extern "C" __global__ void phys3d_rot(real* W, i64 fsr) {
    i64 p = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= fsr) return;
    real u = W[p], v = W[fsr + p], w = W[2 * fsr + p];
    real ox = W[3 * fsr + p], oy = W[4 * fsr + p], oz = W[5 * fsr + p];
    W[p] = v * oz - w * oy;
    W[fsr + p] = w * ox - u * oz;
    W[2 * fsr + p] = u * oy - v * ox;
}

extern "C" __global__ void post3d(cplx* W, cplx* u0, cplx* acc, const real* __restrict__ tab,
                                  int n, int nh, i64 fsc, int rot, int stage, real dt, real ea1,
                                  real eah, real eb1, real ebh, real klo2, real khi2, real ka2,
                                  real scale) {
    i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsc) return;
    int kxi, kyi, kz;
    decode3d(idx, n, nh, kxi, kyi, kz);
    real kx = (real)kxi, ky = (real)kyi, kzr = (real)kz;
    real k2 = kx * kx + ky * ky + kzr * kzr;
    bool keep = (3 * abs(kxi) < n) && (3 * abs(kyi) < n) && (3 * kz < n);
    cplx N0 = cmk(0, 0), N1 = cmk(0, 0), N2 = cmk(0, 0);
    if (keep) {
        if (rot) {
            N0 = W[idx];
            N1 = W[fsc + idx];
            N2 = W[2 * fsc + idx];
        } else {
            cplx t11 = W[idx], t22 = W[fsc + idx], t12 = W[2 * fsc + idx];
            cplx t13 = W[3 * fsc + idx], t23 = W[4 * fsc + idx];
            cplx t33 = cmk(-(t11.x + t22.x), -(t11.y + t22.y));
            // N_i = -i k_j T_ij
            N0 = cmulmi(cadd(cadd(csc(kx, t11), csc(ky, t12)), csc(kzr, t13)));
            N1 = cmulmi(cadd(cadd(csc(kx, t12), csc(ky, t22)), csc(kzr, t23)));
            N2 = cmulmi(cadd(cadd(csc(kx, t13), csc(ky, t23)), csc(kzr, t33)));
        }
        // Leray projection N <- N - k (k.N)/k^2
        real ik2 = k2 > 0 ? (real)1 / k2 : (real)0;
        cplx d = csc(ik2, cadd(cadd(csc(kx, N0), csc(ky, N1)), csc(kzr, N2)));
        N0 = cfma(-kx, d, N0);
        N1 = cfma(-ky, d, N1);
        N2 = cfma(-kzr, d, N2);
    }
    IF f = if_factors(tab, nh, abs(kxi), abs(kyi), kz, k2, ea1, eah, eb1, ebh, klo2, khi2,
                      ka2);
    cplx a = rk_stage(stage, dt, f, N0, u0 + idx, acc + idx);
    cplx b = rk_stage(stage, dt, f, N1, u0 + fsc + idx, acc + fsc + idx);
    cplx c = rk_stage(stage, dt, f, N2, u0 + 2 * fsc + idx, acc + 2 * fsc + idx);
    write_pre3d(a, b, c, kx, ky, kzr, keep, rot, scale, W, idx, fsc);
}
