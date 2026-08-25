#!/usr/bin/env python3
"""
select_settings.py — choose (cheb_trunc, svd_rank) WITHOUT knowing the answer.

THE PROBLEM THIS SOLVES
-----------------------
refine_lsq.py reports a "BEST" setting, but it picks it by minimising the error
against the true s2/s4. On simulation data that is fine — it tells you what the
reconstruction is capable of. On solar data it is impossible: there is no true
s2/s4 to compare against, and the residual cannot substitute for it. Measured
on the G=0.25 clean PINN field:

    cheb=30, svd=1   residual 4.490e-1   ->  s4 error  3.66%
    cheb=60, svd=1   residual 4.487e-1   ->  s4 error  0.22%

Residuals agreeing to 0.07%, errors differing by 17x. On a PINN field the
residual is dominated by network wiggle noise that barely moves between
settings, so it has no discriminating power at all.

This script scores settings using only quantities computable without the truth.

THE TWO SIGNALS
---------------
1. WINDOW STABILITY (primary). Split the time series into disjoint sub-windows
   and recover s2/s4 in each. The physical parameters are constants: a setting
   that resolves the derivatives correctly returns the same numbers from every
   window, while a setting whose derivative is corrupted drifts, because the
   corruption depends on which part of the (decaying) signal is being
   differentiated. Reported as (max-min)/|mean| across windows.

2. EQUATION-SUBSET CONSISTENCY (secondary, and genuinely independent). Recover
   s2/s4 twice: once from the full three-equation stack, once from the two
   momentum equations alone. The three equations weight the mu-derivatives
   differently — eq5 carries nu4_h*d4h/dmu4 and dmu(u_theta), eq1 carries
   dmu(h) and nu4*d4u/dmu4 — so a truncation that mangles high-order
   derivatives corrupts them by different amounts and the two answers separate.
   With a good setting they agree. This is exactly the redundancy the
   velocity-only test established: dropping eq5 should not change the answer,
   so any change is diagnostic of a bad setting rather than of real physics.

HONEST LIMITATION — READ THIS
-----------------------------
Stability is necessary but not sufficient. A setting that is *consistently*
biased — one that truncates the same real signal out of every window and every
equation — scores well while being wrong. The two signals are chosen to be
sensitive to different failure modes, which makes that harder, but neither
detects a uniform bias. Treat the output as "the defensible choice given no
truth", not as a proof of correctness. It should be validated on simulation
data first (where the truth IS known, and this script reports the comparison)
before it is trusted on solar data. That validation is what --report-truth
does, and it is the whole point of running this now.

WHERE AND HOW TO RUN IT
-----------------------
No GPU. Runs in seconds. Run it in the PINN run directory — the one containing
checkpoints/ — on a login node or your laptop:

    cd .../PINN_1923/test1                # the dir holding checkpoints/
    python3 select_settings.py --input ../s2=0.19_s4=0.23/input.json

--input is REQUIRED for checkpoints written before nu4/nu4_h were saved into
the h5 attrs. Without them the hyperdiffusion terms silently default to zero
and every number here is garbage; the script refuses to run in that case.

    --windows 3        number of disjoint sub-windows (default 3)
    --cheb 20 30 40 60 truncations to scan
    --svd 0 1 2 4      ranks to scan
    --field pred|true  which field to score (default pred = the PINN field)
    --report-truth     also print the true errors, for validation ONLY.
                       Use on simulation data to check the selector works.
                       There is nothing to pass on solar data.
"""
import argparse
import json
import os
import sys

import numpy as np
import h5py

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="checkpoints",
                help="directory holding pinn_fields.h5 (default: checkpoints)")
ap.add_argument("--input", default=None,
                help="input.json — REQUIRED unless the checkpoint carries "
                     "nu4/nu4_h in its attrs")
ap.add_argument("--windows", type=int, default=3,
                help="number of disjoint time sub-windows (default 3)")
ap.add_argument("--cheb", type=int, nargs="+", default=[20, 30, 40, 60])
ap.add_argument("--svd", type=int, nargs="+", default=[0, 1, 2, 4])
ap.add_argument("--field", choices=["pred", "true"], default="pred",
                help="score the PINN field (pred) or the simulation field (true)")
ap.add_argument("--edge-trim", type=int, default=None)
ap.add_argument("--report-truth", action="store_true",
                help="ALSO print true errors — validation only, not available "
                     "on solar data")
ap.add_argument("--score", choices=["eq", "window", "combined"], default="eq",
                help="which signal to rank by. DEFAULT 'eq' (equation-subset "
                     "consistency): measured rank-correlation +0.92 with true "
                     "error on the first real checkpoint, versus +0.03 for "
                     "'window' and +0.08 for 'combined'. Use the others only "
                     "to reproduce that comparison.")
a = ap.parse_args()

fp = os.path.join(a.ckpt, "pinn_fields.h5")
if not os.path.exists(fp):
    sys.exit(f"not found: {fp}\nRun this in the PINN run directory (the one "
             f"containing checkpoints/).")

with h5py.File(fp, "r") as f:
    mu = f["mu"][:]
    t = f["t"][:]
    at = dict(f.attrs)
    D = {}
    for n in ("u_theta", "u_phi", "h"):
        g = f[n]
        D[n] = dict(pred=g["pred_r"][:] + 1j * g["pred_i"][:],
                    true=g["true_r"][:] + 1j * g["true_i"][:])

m = int(at["m"])
G = float(at["G"])
S2_TRUE = float(at["s2_true"])
S4_TRUE = float(at["s4_true"])
EDGE_TRIM = a.edge_trim if a.edge_trim is not None else int(at.get("edge_trim", 8))
EPS_REG = 0.05
NU4 = float(at.get("nu4", np.nan))
NU4_H = float(at.get("nu4_h", np.nan))

if a.input:
    if not os.path.exists(a.input):
        sys.exit(f"--input given but not found: {a.input}")
    P = json.load(open(a.input))
    EPS_REG = float(P.get("eps_reg", 0.05))
    NU4 = float(P.get("nu4", 0.0))
    NU4_H = float(P.get("nu4_h", 0.0))

# Refuse rather than silently produce garbage. Omitting the hyperdiffusion
# terms is precisely the bug that made a whole GPU run meaningless: the
# feasibility check returned s4 with the WRONG SIGN on data that recovers to
# 0.12% once the terms are included.
if not np.isfinite(NU4) or not np.isfinite(NU4_H):
    sys.exit(
        "ABORT: nu4 / nu4_h are not in the checkpoint attrs and no --input was\n"
        "  given. They are PRESENT IN THE DATA, so the residual needs them; a\n"
        "  silent default of zero makes every number here meaningless.\n"
        "  Re-run with:  --input <data-dir>/input.json")

N_t, N_mu = len(t), len(mu)
mu_sorted = np.sort(mu)
PDE_MU_MAX = float(mu_sorted[-EDGE_TRIM - 1])
_order = np.argsort(mu)
_inv = np.argsort(_order)
_mus = mu[_order]

print("=" * 74)
print(f"select_settings  |  {fp}   [field = {a.field}]")
print(f"  m={m}  G={G}  nu4={NU4:.3e}  nu4_h={NU4_H:.3e}  edge_trim={EDGE_TRIM}")
print(f"  N_t={N_t}  N_mu={N_mu}  t in [{t.min():.1f}, {t.max():.1f}]")
print(f"  scoring on {a.windows} disjoint sub-windows")
print("=" * 74)


def cheb_trunc_D(x, K):
    x = np.asarray(x, float)
    N = len(x)
    th = np.arccos(np.clip(x, -1, 1))
    kk = np.arange(N)
    C = (2.0 / N) * np.cos(np.outer(kk, th))
    C[0] *= 0.5
    Dm = np.zeros((N, N))
    for k in range(N - 1):
        Dm[k, k + 1] = 2 * (k + 1)
    for k in range(N - 3, -1, -1):
        Dm[k, :] += Dm[k + 2, :]
    Dm[0, :] *= 0.5
    Tr = np.diag((np.arange(N) < K).astype(float))
    return np.cos(np.outer(th, kk)) @ Dm @ Tr @ C


def svd_trunc(A, r):
    if r is None or r <= 0 or r >= min(A.shape):
        return A
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r, :]


_Dmu_cache = {}


def lsq(Uth, Uph, H, tt, K, r, drop_eq5=False):
    """Closed-form (s2,s4) on one time slice. Identical formulation to
    lsq_from_data.py, hyperdiffusion terms included."""
    n_t = len(tt)
    if n_t < 9:                      # dt4 stencil needs a usable interior
        return None
    Uth, Uph, H = svd_trunc(Uth, r), svd_trunc(Uph, r), svd_trunc(H, r)
    if K not in _Dmu_cache:
        _Dmu_cache[K] = cheb_trunc_D(_mus, K)
    Dmu = _Dmu_cache[K]
    D4 = Dmu @ Dmu @ Dmu @ Dmu
    dmu = lambda F: (F[:, _order] @ Dmu.T)[:, _inv]
    dmu4 = lambda F: (F[:, _order] @ D4.T)[:, _inv]
    dt_u = float(np.median(np.diff(tt)))

    def dt(A):
        d = np.zeros_like(A)
        d[2:-2] = (-A[4:] + 8 * A[3:-1] - 8 * A[1:-3] + A[:-4]) / (12 * dt_u)
        return d

    mu2 = mu[None, :]
    st = np.sqrt(np.maximum(1 - mu2 ** 2, 0))
    ist = 1 / (st + EPS_REG)
    a2, a4 = -mu2 ** 2, -mu2 ** 4
    imm = 1j * m
    c0_1 = dt(Uth) - 2 * mu2 * Uph - G * st * dmu(H) + NU4 * dmu4(Uth)
    c0_2 = dt(Uph) + 2 * mu2 * Uth + G * ist * imm * H + NU4 * dmu4(Uph)
    c0_5 = dt(H) + ist * (mu2 * Uth - (1 - mu2 ** 2) * dmu(Uth) + imm * Uph) \
        + NU4_H * dmu4(H)
    c2_1 = a2 * imm * Uth - 2 * mu2 * a2 * Uph
    c2_2 = a2 * imm * Uph + (2 * mu2 * a2 + 2 * mu2 * (1 - mu2 ** 2)) * Uth
    c2_5 = a2 * imm * H
    c4_1 = a4 * imm * Uth - 2 * mu2 * a4 * Uph
    c4_2 = a4 * imm * Uph + (2 * mu2 * a4 + 4 * mu2 ** 3 * (1 - mu2 ** 2)) * Uth
    c4_5 = a4 * imm * H
    msk = np.zeros((n_t, N_mu), bool)
    msk[:, np.abs(mu) <= PDE_MU_MAX] = True
    msk[:2, :] = False
    msk[-2:, :] = False

    def S(x1, x2, x5):
        parts = [x1[msk], x2[msk]] + ([] if drop_eq5 else [x5[msk]])
        v = np.concatenate(parts)
        return np.concatenate([v.real, v.imag])

    b = S(c0_1, c0_2, c0_5)
    A = np.column_stack([S(c2_1, c2_2, c2_5), S(c4_1, c4_2, c4_5)])
    sol, *_ = np.linalg.lstsq(A, -b, rcond=None)
    return float(sol[0]), float(sol[1])


# ── disjoint sub-windows ────────────────────────────────────────────────────
edges = np.linspace(0, N_t, a.windows + 1).astype(int)
slices = [slice(edges[i], edges[i + 1]) for i in range(a.windows)]
print("sub-windows:", ", ".join(
    f"t=[{t[s][0]:.0f},{t[s][-1]:.0f}] ({len(t[s])} pts)" for s in slices))
print()

key = a.field
Uth_all, Uph_all, H_all = D["u_theta"][key], D["u_phi"][key], D["h"][key]


def rel_spread(vals):
    vals = np.asarray(vals, float)
    denom = max(abs(float(np.mean(vals))), 1e-30)
    return float((vals.max() - vals.min()) / denom)


rows = []
for K in a.cheb:
    for r in a.svd:
        # full-window recovery = the number you would actually quote
        full = lsq(Uth_all, Uph_all, H_all, t, K, r)
        if full is None:
            continue
        # signal 1: stability across disjoint windows
        w2, w4 = [], []
        ok = True
        for s in slices:
            out = lsq(Uth_all[s], Uph_all[s], H_all[s], t[s], K, r)
            if out is None:
                ok = False
                break
            w2.append(out[0])
            w4.append(out[1])
        if not ok:
            continue
        spread = rel_spread(w2) + rel_spread(w4)
        # signal 2: full stack vs momentum-only
        vel = lsq(Uth_all, Uph_all, H_all, t, K, r, drop_eq5=True)
        disagree = (abs(vel[0] - full[0]) / max(abs(full[0]), 1e-30)
                    + abs(vel[1] - full[1]) / max(abs(full[1]), 1e-30))
        rows.append(dict(K=K, r=r, s2=full[0], s4=full[1],
                         spread=spread, disagree=disagree,
                         w2=w2, w4=w4))

if not rows:
    sys.exit("no setting produced a usable fit — try fewer --windows")

# Scoring. MEASURED on the first real checkpoint (G=0.25, case 0.19/0.23, clean,
# 16 settings), Spearman rank correlation against the true total error:
#     equation-subset consistency ... +0.92   (strong)
#     window stability ............. +0.03   (none)
# Window stability turned out to measure sub-window sampling noise, not setting
# quality: each sub-window holds only ~100 points, so even the BEST setting's
# per-window parameters wander by ~10%. Used alone it picks a setting with 14%
# s4 error. It is therefore reported as a diagnostic but NOT scored by default.
# Naively summing the two is also wrong: window spread spans 0.3-8.4 while eq
# disagreement spans 0.005-0.4, so the useless signal numerically swamps the
# good one (the sum scores only +0.08).
SCORERS = {
    "eq":       lambda d: d["disagree"],
    "window":   lambda d: d["spread"],
    "combined": lambda d: d["spread"] + d["disagree"],
}
for d in rows:
    d["score"] = SCORERS[a.score](d)

rows.sort(key=lambda d: d["score"])

print(f"ranking by: {a.score}"
      + ("   (equation-subset consistency — the validated signal)" if a.score == "eq"
         else "   (WARNING: weak signal, see --help)"))
print()
print(f"{'cheb':>5} {'svd':>4} | {'s2':>9} {'s4':>9} | "
      f"{'win spread':>10} {'eq disagree':>11} {'SCORE':>9}"
      + ("   | true err s2/s4" if a.report_truth else ""))
print("-" * (74 + (22 if a.report_truth else 0)))
for d in rows:
    line = (f"{d['K']:>5} {d['r']:>4} | {d['s2']:>9.5f} {d['s4']:>9.5f} | "
            f"{d['spread']:>10.3e} {d['disagree']:>11.3e} {d['score']:>9.3e}")
    if a.report_truth:
        e2 = abs(d["s2"] - S2_TRUE) / abs(S2_TRUE) * 100
        e4 = abs(d["s4"] - S4_TRUE) / abs(S4_TRUE) * 100
        line += f"   | {e2:6.2f}% {e4:6.2f}%"
    print(line)

best = rows[0]
print()
print("=" * 74)
print(f"TRUTH-FREE PICK:  cheb_trunc={best['K']}, svd_rank={best['r']}")
print(f"   s2 = {best['s2']:.6f}     s4 = {best['s4']:.6f}")
print(f"   eq disagreement = {best['disagree']:.3e}   "
      f"(window spread = {best['spread']:.3e}, diagnostic only)")
print(f"   per-window s2: " + "  ".join(f"{v:.5f}" for v in best["w2"]))
print(f"   per-window s4: " + "  ".join(f"{v:.5f}" for v in best["w4"]))
print("   (no true value was used to choose this)")

if a.report_truth:
    print("-" * 74)
    print("VALIDATION ONLY — this section needs the truth and does not exist")
    print("on solar data. It answers: does the truth-free rule pick well?")
    e2b = abs(best["s2"] - S2_TRUE) / abs(S2_TRUE) * 100
    e4b = abs(best["s4"] - S4_TRUE) / abs(S4_TRUE) * 100
    print(f"   truth-free pick  cheb={best['K']:<3} svd={best['r']:<2} "
          f"-> s2 {e2b:.2f}%  s4 {e4b:.2f}%")
    opt = min(rows, key=lambda d: (abs(d["s2"] - S2_TRUE) / abs(S2_TRUE)
                                   + abs(d["s4"] - S4_TRUE) / abs(S4_TRUE)))
    e2o = abs(opt["s2"] - S2_TRUE) / abs(S2_TRUE) * 100
    e4o = abs(opt["s4"] - S4_TRUE) / abs(S4_TRUE) * 100
    print(f"   truth-optimal    cheb={opt['K']:<3} svd={opt['r']:<2} "
          f"-> s2 {e2o:.2f}%  s4 {e4o:.2f}%")
    if best["K"] == opt["K"] and best["r"] == opt["r"]:
        print("   VERDICT: the truth-free rule picked the optimal setting.")
    else:
        cost2, cost4 = e2b - e2o, e4b - e4o
        print(f"   VERDICT: different setting; costs {cost2:+.2f}pp on s2, "
              f"{cost4:+.2f}pp on s4 vs the oracle.")
        print("   Small cost => the rule is usable. Large cost => it is not,")
        print("   and no solar number should be quoted until a better")
        print("   selector exists.")
print("=" * 74)