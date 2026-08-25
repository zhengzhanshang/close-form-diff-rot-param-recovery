#!/usr/bin/env python3
"""
refine_lsq.py — re-run the closed-form (s2, s4) recovery on an ALREADY-TRAINED
PINN field, scanning the denoising settings. No GPU, no retraining: it reads
checkpoints/pinn_fields.h5 written by PINN_s2s4_recovery.py and runs in seconds.

Why this works
--------------
The PDE residual is EXACTLY linear in (s2, s4):  R = c0 + s2*c2 + s4*c4,
so recovery is a linear least-squares solve. The only difficulty is that the
residual needs DERIVATIVES of the field, and a PINN field has small-amplitude
high-frequency wiggles that differentiation amplifies (~35x in practice).
Two denoisers fix that, and both exploit the physics:
  * SVD rank truncation — a single mode is separable, F(t,mu) = T(t)*prof(mu),
    i.e. EXACTLY rank-1. Noise is full-rank, so a low-rank truncation removes
    it in BOTH t and mu at once.
  * Truncated Chebyshev differentiation in mu — the physical mu-structure is
    low-order; high Chebyshev modes are noise.

Usage
-----
    python3 refine_lsq.py                       # scan, report best
    python3 refine_lsq.py --ckpt checkpoints    # explicit path
    python3 refine_lsq.py --svd 1 --cheb 30     # single setting
"""
import argparse, json, os
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints")
ap.add_argument("--input", default=None, help="input.json (for m, G, eps_reg); "
                                              "default: read from the h5 attrs")
ap.add_argument("--svd", type=int, default=None, help="single svd_rank (0=off)")
ap.add_argument("--cheb", type=int, default=None, help="single cheb_trunc")
ap.add_argument("--edge-trim", type=int, default=None)
a = ap.parse_args()

fp = os.path.join(a.ckpt, "pinn_fields.h5")
with h5py.File(fp, "r") as f:
    mu = f["mu"][:]; t = f["t"][:]
    at = dict(f.attrs)
    D = {}
    for n in ("u_theta", "u_phi", "h"):
        g = f[n]
        D[n] = dict(pred=g["pred_r"][:] + 1j*g["pred_i"][:],
                    true=g["true_r"][:] + 1j*g["true_i"][:])

m  = int(at["m"]); G = float(at["G"])
S2_TRUE = float(at["s2_true"]); S4_TRUE = float(at["s4_true"])
EDGE_TRIM = a.edge_trim if a.edge_trim is not None else int(at.get("edge_trim", 8))
EPS_REG = 0.05
if a.input and os.path.exists(a.input):
    P = json.load(open(a.input)); EPS_REG = float(P.get("eps_reg", 0.05))

N_t, N_mu = len(t), len(mu)
mu_sorted = np.sort(mu)
PDE_MU_MAX = float(mu_sorted[-EDGE_TRIM - 1])
_dt_unif = float(np.median(np.diff(t)))
_order = np.argsort(mu); _inv = np.argsort(_order); _mus = mu[_order]

print("="*70)
print(f"refine_lsq  |  {fp}")
print(f"  m={m}  G={G}  true s2={S2_TRUE}  s4={S4_TRUE}  "
      f"edge_trim={EDGE_TRIM}  N_t={N_t} N_mu={N_mu}")
print("="*70)


def cheb_trunc_D(x, K):
    x = np.asarray(x, float); N = len(x)
    th = np.arccos(np.clip(x, -1, 1)); kk = np.arange(N)
    C = (2.0/N)*np.cos(np.outer(kk, th)); C[0] *= 0.5
    Dm = np.zeros((N, N))
    for k in range(N-1):
        Dm[k, k+1] = 2*(k+1)
    for k in range(N-3, -1, -1):
        Dm[k, :] += Dm[k+2, :]
    Dm[0, :] *= 0.5
    Tr = np.diag((np.arange(N) < K).astype(float))
    return np.cos(np.outer(th, kk)) @ Dm @ Tr @ C


def svd_trunc(A, r):
    if r is None or r <= 0 or r >= min(A.shape):
        return A
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    return (U[:, :r]*S[:r]) @ Vh[:r, :]


def lsq(Uth, Uph, H, K, r):
    Uth, Uph, H = svd_trunc(Uth, r), svd_trunc(Uph, r), svd_trunc(H, r)
    Dmu = cheb_trunc_D(_mus, K)
    dmu = lambda F: (F[:, _order] @ Dmu.T)[:, _inv]
    def dt(A):
        d = np.zeros_like(A)
        d[2:-2] = (-A[4:] + 8*A[3:-1] - 8*A[1:-3] + A[:-4])/(12*_dt_unif)
        return d
    mu2 = mu[None, :]
    st = np.sqrt(np.maximum(1-mu2**2, 0)); ist = 1/(st+EPS_REG)
    a2, a4 = -mu2**2, -mu2**4; imm = 1j*m
    c0_1 = dt(Uth) - 2*mu2*Uph - G*st*dmu(H)
    c0_2 = dt(Uph) + 2*mu2*Uth + G*ist*imm*H
    c0_5 = dt(H) + ist*(mu2*Uth - (1-mu2**2)*dmu(Uth) + imm*Uph)
    c2_1 = a2*imm*Uth - 2*mu2*a2*Uph
    c2_2 = a2*imm*Uph + (2*mu2*a2 + 2*mu2*(1-mu2**2))*Uth
    c2_5 = a2*imm*H
    c4_1 = a4*imm*Uth - 2*mu2*a4*Uph
    c4_2 = a4*imm*Uph + (2*mu2*a4 + 4*mu2**3*(1-mu2**2))*Uth
    c4_5 = a4*imm*H
    msk = np.zeros((N_t, N_mu), bool)
    msk[:, np.abs(mu) <= PDE_MU_MAX] = True
    msk[:2, :] = False; msk[-2:, :] = False
    def S(x1, x2, x5):
        v = np.concatenate([x1[msk], x2[msk], x5[msk]])
        return np.concatenate([v.real, v.imag])
    b = S(c0_1, c0_2, c0_5)
    A = np.column_stack([S(c2_1, c2_2, c2_5), S(c4_1, c4_2, c4_5)])
    sol, *_ = np.linalg.lstsq(A, -b, rcond=None)
    res = float(np.linalg.norm(A@sol + b)/(np.linalg.norm(b)+1e-30))
    return float(sol[0]), float(sol[1]), res


# singular spectrum: how many modes does the field really have?
for n in ("u_theta", "u_phi", "h"):
    sv = np.linalg.svd(D[n]["true"], compute_uv=False)[:5]
    print(f"  SVD spectrum {n:8s} (true): " + " ".join(f"{v/sv[0]:.1e}" for v in sv))
print()

def report(tag, key, K, r):
    s2, s4, res = lsq(D["u_theta"][key], D["u_phi"][key], D["h"][key], K, r)
    e2 = abs(s2-S2_TRUE)/abs(S2_TRUE)*100; e4 = abs(s4-S4_TRUE)/abs(S4_TRUE)*100
    print(f"  {tag:22s} s2={s2:+.5f} ({e2:5.2f}%)  s4={s4:+.5f} ({e4:6.2f}%)  "
          f"resid={res:.3e}")
    return e2+e4, (K, r), (s2, s4)

if a.svd is not None or a.cheb is not None:
    K = a.cheb if a.cheb is not None else 30
    r = a.svd if a.svd is not None else 4
    print(f"single setting: cheb_trunc={K} svd_rank={r}")
    report("TRUE field", "true", K, r)
    report("PINN field", "pred", K, r)
else:
    print("--- scan over (cheb_trunc, svd_rank) on the PINN field ---")
    best = None
    for K in (20, 30, 40, 60):
        for r in (0, 1, 2, 4):
            try:
                sc, cfg, vals = report(f"cheb={K:<3d} svd={r:<2d}", "pred", K, r)
                if best is None or sc < best[0]:
                    best = (sc, cfg, vals)
            except Exception as e:
                print(f"  cheb={K} svd={r}: failed ({e})")
    print()
    K, r = best[1]
    print("="*70)
    print(f"BEST on the PINN field: cheb_trunc={K}, svd_rank={r}")
    print(f"   s2={best[2][0]:.6f} (true {S2_TRUE})   s4={best[2][1]:.6f} (true {S4_TRUE})")
    print(f"   -> put  \"cheb_trunc\": {K}, \"svd_rank\": {r}  in input.json")
    print("-"*70)
    print("reference — same settings on the TRUE field (the achievable ceiling):")
    report("TRUE field", "true", K, r)
    print("="*70)