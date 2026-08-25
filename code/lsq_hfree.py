#!/usr/bin/env python3
"""
lsq_hfree.py — closed-form (s2, s4) recovery from VELOCITIES ONLY.
h is never observed and never used, anywhere.

WHY THIS EXISTS
---------------
The `--velocity-only` flag in lsq_from_data.py drops the continuity equation's
rows but still uses the true h inside eq1/eq2's pressure terms, so it only ever
proved that eq5 is REDUNDANT — not that h is dispensable. On solar data h is not
observable at all, so that gap had to be closed before any solar claim.

It is closed analytically. For a single azimuthal mode, d/dphi -> i*m makes h
appear ALGEBRAICALLY in eq2 (verified: eq2 contains no derivative of h), so eq2
inverts for h in terms of the velocities. Substituting into eq1 gives a relation
in the velocities alone.

WHAT THE SYMBOLIC DERIVATION ESTABLISHED (sympy, 2026-08-13)
------------------------------------------------------------
  * h itself is EXACTLY LINEAR in (s2, s4)  -- monomials (0,0),(0,1),(1,0)
  * h-free eq1 is EXACTLY LINEAR in (s2, s4) -- so the closed-form least squares
    survives elimination:   R = c0 + s2*c2 + s4*c4, velocities only.
  * h-free eq5 is QUADRATIC (it contains s2^2, s2*s4, s4^2), because eq5 holds
    Omega_d*h and h already carries s2/s4. eq5 therefore CANNOT be used in the
    closed form and is not implemented here.

So the h-free route yields ONE linear equation family (eq1), not three: eq2 is
consumed by the elimination and eq5 goes quadratic. That is still enormously
over-determined -- one equation at every (mu, t) point, two unknowns -- but
whether it is well CONDITIONED is an empirical question this script answers
(watch the reported cond).

THE COST
--------
h-free eq1 needs a MIXED SECOND derivative d2(u_phi)/dmu dt, where the original
three-equation form needed only first derivatives. Second derivatives amplify
noise harder, so on noisy data expect this route to degrade faster -- which is
precisely where a PINN's smoothing earns its place. The two routes COMPOSE:
reconstruct with the PINN, then recover with this h-free solve.

USAGE
-----
    # on clean simulation snapshots (start here -- truth is known)
    python3 lsq_hfree.py --input <data-dir>/input.json --tmin 10 --scan

    # on a PINN checkpoint (PINN-smoothed field, then h-free recovery)
    python3 lsq_hfree.py --ckpt checkpoints --input <data-dir>/input.json --scan

    # verify the hand-transcribed coefficients against sympy (needs sympy)
    python3 lsq_hfree.py --selftest

Flags: --cheb / --svd for a single setting, --scan to sweep them,
       --tmin/--tmax for the window, --field pred|true with --ckpt.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import h5py

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--input", default=None, help="input.json (m, G, s2, s4, eps_reg)")
ap.add_argument("--ckpt", default=None,
                help="a PINN checkpoint dir holding pinn_fields.h5; if given, the "
                     "field comes from there instead of raw snapshots")
ap.add_argument("--field", choices=["pred", "true"], default="pred",
                help="with --ckpt: PINN field (pred) or the field it was given (true)")
ap.add_argument("--tmin", type=float, default=-np.inf)
ap.add_argument("--tmax", type=float, default=np.inf)
ap.add_argument("--edge-trim", type=int, default=None)
ap.add_argument("--cheb", type=int, default=30)
ap.add_argument("--svd", type=int, default=0)
ap.add_argument("--scan", action="store_true", help="sweep cheb x svd")
ap.add_argument("--selftest", action="store_true",
                help="check the numpy coefficients against a fresh sympy "
                     "derivation on random data, then exit")


# ── the h-free eq1 coefficients ──────────────────────────────────────────────
# R = c0 + s2*c2 + s4*c4, from eliminating h between eq1 and eq2.
# Generated from sympy (sp.pycode) rather than hand-copied; --selftest re-derives
# symbolically and compares, so a transcription slip cannot go unnoticed.
#   U=u_theta  V=u_phi  Um=d_mu u_theta  Ut=d_t u_theta
#   Vm=d_mu u_phi  Vt=d_t u_phi  Vmt=d_mu d_t u_phi
def hfree_coeffs(mu, m, eps, U, V, Um, Ut, Vm, Vt, Vmt):
    sq = np.sqrt(np.maximum(1 - mu**2, 0.0))
    j = 1j
    c0 = (-2*j*U*eps*sq + 4*j*U*mu**2 - 2*j*U
          - 2*j*Um*eps*mu*sq + 2*j*Um*mu**3 - 2*j*Um*mu
          - j*Vmt*eps*sq + j*Vmt*mu**2 - j*Vmt
          + j*Vt*mu + m*(Ut - 2*V*mu)) / m
    c2 = (12*j*U*eps*mu**2*sq - 2*j*U*eps*sq
          - 16*j*U*mu**4 + 16*j*U*mu**2 - 2*j*U
          + 4*j*Um*eps*mu**3*sq - 2*j*Um*eps*mu*sq
          - 4*j*Um*mu**5 + 6*j*Um*mu**3 - 2*j*Um*mu
          - m*mu*(j*U*m*mu + 2*V*eps*sq - 5*V*mu**2 + 2*V
                  + Vm*eps*mu*sq - Vm*mu**3 + Vm*mu)) / m
    c4 = mu**2*(30*j*U*eps*mu**2*sq - 12*j*U*eps*sq
                - 36*j*U*mu**4 + 46*j*U*mu**2 - 12*j*U
                + 6*j*Um*eps*mu**3*sq - 4*j*Um*eps*mu*sq
                - 6*j*Um*mu**5 + 10*j*Um*mu**3 - 4*j*Um*mu
                - m*mu*(j*U*m*mu + 4*V*eps*sq - 7*V*mu**2 + 4*V
                        + Vm*eps*mu*sq - Vm*mu**3 + Vm*mu)) / m
    return c0, c2, c4


def selftest():
    """Re-derive symbolically and compare to the numpy transcription."""
    import sympy as sp
    mu_s, t_s, m_s, G_s, eps_s, s2_s, s4_s = sp.symbols(
        'mu t m G eps s2 s4', real=True)
    Uth = sp.Function('Uth')(mu_s, t_s)
    Uph = sp.Function('Uph')(mu_s, t_s)
    Hh = sp.Function('H')(mu_s, t_s)
    Om = -s2_s*mu_s**2 - s4_s*mu_s**4
    st = sp.sqrt(1-mu_s**2)
    ist = 1/(st+eps_s)
    corf_th = 2*(1+Om)*mu_s
    corf_phi = 2*(1+Om)*mu_s + (1-mu_s**2)*(2*s2_s*mu_s + 4*s4_s*mu_s**3)
    eq1 = sp.diff(Uth, t_s) + Om*sp.I*m_s*Uth - corf_th*Uph - G_s*st*sp.diff(Hh, mu_s)
    eq2 = sp.diff(Uph, t_s) + Om*sp.I*m_s*Uph + corf_phi*Uth + G_s*ist*sp.I*m_s*Hh
    Hsol = sp.solve(eq2, Hh)[0]
    e = sp.expand(eq1.subs(sp.Derivative(Hh, mu_s), sp.diff(Hsol, mu_s)).subs(Hh, Hsol))
    P = sp.Poly(e, s2_s, s4_s)
    mons = sorted(P.monoms())
    print("h-free eq1 monomials in (s2,s4):", mons)
    assert all(sum(k) <= 1 for k in mons), "NOT LINEAR — derivation broken"
    print("  -> LINEAR confirmed")

    U_, V_, Um_, Ut_, Vm_, Vt_, Vmt_ = sp.symbols('U V Um Ut Vm Vt Vmt')
    sub = {sp.Derivative(Uth, mu_s): Um_, sp.Derivative(Uth, t_s): Ut_,
           sp.Derivative(Uph, mu_s): Vm_, sp.Derivative(Uph, t_s): Vt_,
           sp.Derivative(Uph, mu_s, t_s): Vmt_, sp.Derivative(Uph, t_s, mu_s): Vmt_,
           Uth: U_, Uph: V_}
    syms = [mu_s, m_s, eps_s, U_, V_, Um_, Ut_, Vm_, Vt_, Vmt_]
    fns = [sp.lambdify(syms, P.coeff_monomial(k).subs(sub), "numpy")
           for k in (sp.Integer(1), s2_s, s4_s)]

    rng = np.random.default_rng(0)
    vals = [rng.uniform(-0.9, 0.9), 5.0, 0.05] + \
           [rng.standard_normal() + 1j*rng.standard_normal() for _ in range(7)]
    sym = [f(*vals) for f in fns]
    num = hfree_coeffs(*vals)
    ok = True
    for nm, a, b in zip(("c0", "c2", "c4"), sym, num):
        d = abs(a-b)/max(abs(a), 1e-30)
        print(f"  {nm}: sympy={a:.6e}  numpy={b:.6e}  rel.diff={d:.2e}")
        ok &= d < 1e-10
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


a = ap.parse_args()
if a.selftest:
    sys.exit(selftest())
if not a.input:
    sys.exit("--input is required (needs m, G, s2, s4, eps_reg)")

P = json.load(open(a.input))
m = int(P["m"])
S2_TRUE, S4_TRUE = float(P["s2"]), float(P["s4"])
EPS = float(P.get("eps_reg", 0.05))
EDGE = a.edge_trim if a.edge_trim is not None else int(P.get("edge_trim", 8))

# ── load the field ───────────────────────────────────────────────────────────
if a.ckpt:
    fp = os.path.join(a.ckpt, "pinn_fields.h5")
    with h5py.File(fp, "r") as f:
        mu = f["mu"][:]; t = f["t"][:]
        k = a.field
        Uth = f["u_theta"][f"{k}_r"][:] + 1j*f["u_theta"][f"{k}_i"][:]
        Uph = f["u_phi"][f"{k}_r"][:] + 1j*f["u_phi"][f"{k}_i"][:]
    src = f"{fp} [{a.field}]"
else:
    d = os.path.dirname(os.path.abspath(a.input))
    sd = P.get("snapshots_dir", "snapshots")
    sd = sd if os.path.isabs(sd) else os.path.normpath(os.path.join(d, sd))
    files = sorted(glob.glob(os.path.join(sd, "snapshots_s*.h5")) or
                   glob.glob(os.path.join(sd, "*.h5")))
    if not files:
        sys.exit(f"no snapshots in {sd}")
    tc, Uc, Vc = [], [], []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            tt = f["scales/sim_time"][:]
            for nm in f["scales"]:
                if "mu" in nm.lower():
                    arr = np.array(f["scales"][nm][:])
                    if arr.ndim == 1:
                        mu = arr.ravel()
            nphi = f["tasks/u_theta"].shape[1]
            n = min(len(tt), f["tasks/u_theta"].shape[0])
            msk = (tt[:n] >= a.tmin) & (tt[:n] <= a.tmax)
            if not msk.any():
                continue
            mm = lambda nm: np.fft.fft(f[f"tasks/{nm}"][:n], axis=1)[:, m, :]/nphi
            tc.append(tt[:n][msk]); Uc.append(mm("u_theta")[msk]); Vc.append(mm("u_phi")[msk])
    t = np.concatenate(tc); Uth = np.concatenate(Uc); Uph = np.concatenate(Vc)
    o = np.argsort(t); t, Uth, Uph = t[o], Uth[o], Uph[o]
    src = sd

N_t, N_mu = len(t), len(mu)
mus = np.sort(mu)
MU_MAX = float(mus[-EDGE-1])
order = np.argsort(mu); inv = np.argsort(order); msrt = mu[order]
dt_u = float(np.median(np.diff(t)))

print("="*74)
print(f"lsq_hfree  |  {src}")
print(f"  m={m}  true s2={S2_TRUE}  s4={S4_TRUE}  eps_reg={EPS}  edge_trim={EDGE}")
print(f"  N_t={N_t}  N_mu={N_mu}  t in [{t.min():.1f}, {t.max():.1f}]")
print("  VELOCITIES ONLY — h is never observed and never used")
print("  (one linear equation family: h-free eq1. eq5 is quadratic after")
print("   elimination and cannot be used in closed form.)")
print("="*74)


def cheb_D(x, K):
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
    U_, S_, Vh_ = np.linalg.svd(A, full_matrices=False)
    return (U_[:, :r]*S_[:r]) @ Vh_[:r, :]


def dt4(A):
    d = np.zeros_like(A)
    d[2:-2] = (-A[4:] + 8*A[3:-1] - 8*A[1:-3] + A[:-4])/(12*dt_u)
    return d


def solve(K, r):
    U_, V_ = svd_trunc(Uth, r), svd_trunc(Uph, r)
    D = cheb_D(msrt, K)
    dmu = lambda F: (F[:, order] @ D.T)[:, inv]
    Um, Ut = dmu(U_), dt4(U_)
    Vm, Vt = dmu(V_), dt4(V_)
    Vmt = dmu(dt4(V_))                      # the mixed second derivative
    mu2 = mu[None, :]
    c0, c2, c4 = hfree_coeffs(mu2, m, EPS, U_, V_, Um, Ut, Vm, Vt, Vmt)
    msk = np.zeros((N_t, N_mu), bool)
    msk[:, np.abs(mu) <= MU_MAX] = True
    msk[:2, :] = False; msk[-2:, :] = False   # dt4 stencil edges
    S = lambda x: np.concatenate([x[msk].real, x[msk].imag])
    b = S(c0)
    A = np.column_stack([S(c2), S(c4)])
    sol, *_ = np.linalg.lstsq(A, -b, rcond=None)
    cond = float(np.linalg.cond(A))
    res = float(np.linalg.norm(A@sol + b)/(np.linalg.norm(b)+1e-30))
    return float(sol[0]), float(sol[1]), cond, res


def report(K, r):
    s2, s4, cond, res = solve(K, r)
    e2 = abs(s2-S2_TRUE)/abs(S2_TRUE)*100
    e4 = abs(s4-S4_TRUE)/abs(S4_TRUE)*100
    print(f"  cheb={K:<3d} svd={r:<2d}  s2={s2:+.5f} ({e2:6.2f}%)  "
          f"s4={s4:+.5f} ({e4:7.2f}%)  cond={cond:7.2f}  resid={res:.3e}")
    return e2+e4, (K, r), (s2, s4, cond)


if a.scan:
    print("  setting        s2                 s4                 cond      resid")
    best = None
    for K in (20, 30, 40, 60):
        for r in (0, 1, 2, 4):
            try:
                sc, cfg, vals = report(K, r)
                if best is None or sc < best[0]:
                    best = (sc, cfg, vals)
            except Exception as ex:
                print(f"  cheb={K} svd={r}: failed ({ex})")
    print("="*74)
    print(f"BEST (truth-informed, for validation only): cheb={best[1][0]}, svd={best[1][1]}")
    print(f"   s2={best[2][0]:.6f}  s4={best[2][1]:.6f}  cond={best[2][2]:.2f}")
    print("   NOTE: 'best of scan' is a MINIMUM over many noisy estimates and")
    print("   flatters itself as noise rises — for a degradation curve use a")
    print("   FIXED setting or a truth-free selector instead.")
else:
    report(a.cheb, a.svd)
print("="*74)