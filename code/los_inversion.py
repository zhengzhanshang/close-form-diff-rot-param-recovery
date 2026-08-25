#!/usr/bin/env python3
"""
los_inversion.py — synthetic Dopplergrams from Dedalus fields, then a LINEAR
inversion back to the three m-mode fields the closed-form recovery needs.

WHY THIS EXISTS
---------------
The solar chain has a missing link: HMI gives ONE scalar per pixel (the
line-of-sight velocity), while the s2/s4 recovery needs THREE fields
(u_theta, u_phi, and w = dh/dt, which is a genuine Doppler observable near
disk centre). Nothing bridged those.

Before reaching for a network, note the structure. The projection is

    v_LOS = w*(r_hat.e) + u_theta*(theta_hat.e) + u_phi*(phi_hat.e)

which is LINEAR in the three fields with KNOWN geometric coefficients. For a
single azimuthal mode each field is described by one complex amplitude per
(mu, t), so at fixed (mu, t) every visible longitude contributes ONE linear
equation in SIX real unknowns. With ~Nphi/2 visible longitudes that is hugely
over-determined, and the inversion DECOUPLES per (mu, t) — a small dense least
squares, no optimisation, no training.

GEOMETRY (general B0, observer at central meridian phi=0)
---------------------------------------------------------
    e = (cos B0, 0, sin B0)
    r_hat.e     =  st*cos(phi)*cos(B0) + mu*sin(B0)      <- vertical  (w)
    theta_hat.e =  mu*cos(phi)*cos(B0) - st*sin(B0)      <- u_theta
    phi_hat.e   = -sin(phi)*cos(B0)                      <- u_phi
with mu = cos(theta), st = sin(theta) = sqrt(1-mu^2). Visible: r_hat.e > 0.

*** THE CONDITIONING PROBLEM THIS IS BUILT TO MEASURE ***
At B0 = 0 the u_theta coefficient is mu*cos(phi), which is IDENTICALLY ZERO at
the equator: theta_hat is perpendicular to the line of sight there for every
visible longitude, so u_theta is formally unobservable at mu=0. The sectoral
Rossby mode PEAKS at the equator. A non-zero B0 breaks the degeneracy
(theta_hat.e -> -sin(B0) at the equator), so --b0 is exposed rather than fixed:
B0=0 is the WORST case, not merely the conservative one.

Watch the reported per-mu condition number. That is the number that decides
whether a linear inversion suffices or whether a physics-constrained method
(where a PINN could genuinely earn its place) is needed.

USAGE
-----
    # round-trip validation on data whose answer is known
    python3 los_inversion.py --input <data>/input.json --tmin 10 \
            --b0 0 --noise 0.1 --out los_ckpt

    # then run the SAME closed-form recovery used everywhere else:
    python3 lsq_from_data.py --input <data>/input.json --ckpt los_ckpt \
            --field pred --tmin 10 --mask-poles 60
    #   --field true = the direct m-mode fields (the ceiling)
    #   --field pred = what the LOS inversion recovered  <- the measurement

    python3 los_inversion.py --selftest      # geometry + round-trip checks
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
ap.add_argument("--input", default=None, help="input.json (m, s2, s4, G, nu4...)")
ap.add_argument("--snapshots", default=None, help="override snapshots dir")
ap.add_argument("--tmin", type=float, default=-np.inf)
ap.add_argument("--tmax", type=float, default=np.inf)
ap.add_argument("--b0", type=float, default=0.0,
                help="heliographic latitude of disk centre, DEGREES. B0=0 makes "
                     "u_theta unobservable at the equator (see module docstring); "
                     "the real Sun swings +-7.25 deg over a year.")
ap.add_argument("--noise", type=float, default=0.0,
                help="Gaussian noise on v_LOS, as a fraction of its RMS. This is "
                     "instrument noise on the ACTUAL observable, which is more "
                     "honest than adding noise to the three fields separately.")
ap.add_argument("--noise-seed", type=int, default=0)
ap.add_argument("--ridge", type=float, default=0.0,
                help="Tikhonov ridge (relative to the largest singular value) for "
                     "the per-(mu,t) solve. 0 = plain least squares. Raise only if "
                     "the reported condition numbers demand it.")
ap.add_argument("--omega-floor", type=float, default=0.05,
                help="when integrating w -> h in the temporal Fourier domain, zero "
                     "out |omega| below this (as a fraction of the mode frequency). "
                     "The DC part of h is NOT recoverable from dh/dt — a real "
                     "limitation of the observable, not a numerical choice.")
ap.add_argument("--amp-frac", type=float, default=0.1,
                help="fit the mode rate lambda only where |w| exceeds this "
                     "fraction of its peak. The decaying tail is noise-dominated "
                     "and would otherwise control the fit (57%% error at 10%% "
                     "noise vs 0.16%% with this on).")
ap.add_argument("--lambda-re", type=float, default=None,
                help="override the fitted growth rate Re(lambda), e.g. from "
                     "postrun_analyze, instead of fitting it")
ap.add_argument("--lambda-im", type=float, default=None,
                help="override the fitted angular frequency Im(lambda)")
ap.add_argument("--out", default="los_ckpt", help="output dir for los_fields.h5")
ap.add_argument("--selftest", action="store_true")


def proj(mu, phi, b0):
    """(r_hat.e, theta_hat.e, phi_hat.e) for observer e=(cos b0, 0, sin b0)."""
    st = np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    cb, sb = np.cos(b0), np.sin(b0)
    p_w = st*np.cos(phi)*cb + mu*sb
    p_th = mu*np.cos(phi)*cb - st*sb
    p_ph = -np.sin(phi)*cb
    return p_w, p_th, p_ph


def design(mu_val, phi_vis, m, b0):
    """Rows of the linear system at one mu, for unknowns
       [Re w, Im w, Re u_th, Im u_th, Re u_ph, Im u_ph].
    Real-space field = 2*Re[F_hat * exp(i m phi)], matching the numpy-FFT
    convention F_hat = FFT[field][m]/Nphi used everywhere else in the project."""
    p_w, p_th, p_ph = proj(mu_val, phi_vis, b0)
    c, s = np.cos(m*phi_vis), np.sin(m*phi_vis)
    return np.column_stack([2*p_w*c, -2*p_w*s,
                            2*p_th*c, -2*p_th*s,
                            2*p_ph*c, -2*p_ph*s])


def selftest():
    rng = np.random.default_rng(0)
    m, Nphi = 5, 256
    phi = 2*np.pi*np.arange(Nphi)/Nphi
    ok = True

    # 1) FFT convention: field = 2*Re[F_hat e^{i m phi}] with F_hat = FFT[field][m]/N
    A = rng.standard_normal() + 1j*rng.standard_normal()
    f = 2*np.real(A*np.exp(1j*m*phi))
    A_fft = np.fft.fft(f)[m]/Nphi
    d = abs(A_fft-A)/abs(A)
    print(f"  FFT convention round-trip rel.err {d:.2e}  {'OK' if d<1e-12 else 'FAIL'}")
    ok &= d < 1e-12

    # 2) forward -> inverse recovers the amplitudes exactly (noiseless)
    for b0deg in (0.0, 7.25):
        b0 = np.deg2rad(b0deg)
        for lat in (0.0, 30.0, 60.0):
            mu = np.sin(np.deg2rad(lat))
            W, Th, Ph = (rng.standard_normal()+1j*rng.standard_normal() for _ in range(3))
            p_w, p_th, p_ph = proj(mu, phi, b0)
            v = (2*np.real(W*np.exp(1j*m*phi))*p_w
                 + 2*np.real(Th*np.exp(1j*m*phi))*p_th
                 + 2*np.real(Ph*np.exp(1j*m*phi))*p_ph)
            vis = p_w > 0
            M = design(mu, phi[vis], m, b0)
            x, *_ = np.linalg.lstsq(M, v[vis], rcond=None)
            got = np.array([x[0]+1j*x[1], x[2]+1j*x[3], x[4]+1j*x[5]])
            want = np.array([W, Th, Ph])
            cond = np.linalg.cond(M)
            err = np.abs(got-want).max()/np.abs(want).max()
            # B0=0 is a KNOWN, ALGEBRAIC degeneracy: p_th = (mu/st)*p_w there,
            # so the w and u_theta columns are proportional at EVERY latitude
            # and only the combination st*w + mu*u_theta is observable. Expect
            # failure at B0=0; require success once B0 != 0.
            expect_bad = (b0deg == 0.0)
            flag = ("DEGENERATE (expected at B0=0)" if expect_bad
                    else ("OK" if err < 1e-8 else "FAIL"))
            print(f"  b0={b0deg:>5}° lat={lat:>4.0f}°  cond={cond:9.2e}  "
                  f"rel.err {err:.2e}  {flag}")
            if not expect_bad:
                ok &= err < 1e-8

    # 3) w -> h for a DECAYING mode (the real case): h ~ exp(-i*om*t + sig*t)
    t = np.linspace(0, 60, 400)
    lam_true = -1j*0.4586 + (-0.0586)
    h = np.exp(lam_true*t)
    w = lam_true*h
    h_rec, lam, _ = integrate_w(w[:, None], t)
    d = np.abs(h_rec[:, 0]-h).max()/np.abs(h).max()
    dl = abs(lam-lam_true)/abs(lam_true)
    print(f"  w -> h: lambda rel.err {dl:.2e}, h rel.err {d:.2e}  "
          f"{'OK' if d < 1e-8 else 'FAIL'}")
    ok &= d < 1e-8
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _fit_lambda(F, t, amp_frac):
    """Complex mode rate from the rank-1 temporal mode of F, using only the
    high-amplitude samples (the decaying tail is noise-dominated)."""
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    u1 = U[:, 0]*S[0]
    amp = np.abs(u1)
    keep = amp > amp_frac*amp.max()
    if keep.sum() < 12:
        keep = amp >= np.sort(amp)[-12]
    tt = t[keep]
    lam = (np.polyfit(tt, np.log(amp[keep]), 1)[0]
           + 1j*np.polyfit(tt, np.unwrap(np.angle(u1[keep])), 1)[0])
    # rank-1 dominance = truth-free SNR proxy. A single-column field is
    # trivially rank-1, so treat it as maximally dominant rather than erroring.
    dom = float(S[0]/max(S[1], 1e-30)) if len(S) > 1 else np.inf
    return lam, dom, int(keep.sum())


def integrate_w(w, t, amp_frac=0.1, lam_override=None, others=None):
    """h from w = dh/dt for a field dominated by ONE complex mode: h = w/lambda.

    CRITICAL: lambda is fitted from the CLEANEST available field, not from w.

    All three fields share the SAME mode rate, and w is by far the worst
    recovered from a line-of-sight inversion -- it is the smallest contributor
    to v_LOS (RMS(w) ~ 0.24*RMS(u_theta)) AND it is half of the near-parallel
    (w, u_theta) column pair that the B0=0 degeneracy lives in. Fitting lambda
    from w therefore fails exactly when it matters: at 10% LOS noise the rank-1
    mode of w IS the noise, so amplitude thresholding finds no decay to threshold
    against (it kept 262/380 samples) and lambda came out 16x too small.

    Fields are ranked by rank-1 dominance S0/S1, a truth-free SNR proxy, and
    lambda is taken from the winner.

    NOTE this fixes lambda only. If w itself is badly recovered, h = w/lambda is
    still badly recovered -- a correct lambda cannot denoise w.

    Returns (h, lambda, diagnostic string).
    """
    if lam_override is not None:
        return w/lam_override, lam_override, "SUPPLIED"
    cands = [("w", w)] + list(others or [])
    best = None
    for nm, F in cands:
        try:
            lam, dom, nk = _fit_lambda(F, t, amp_frac)
        except Exception:
            continue
        if best is None or dom > best[1]:
            best = (lam, dom, nk, nm)
    if best is None:
        raise RuntimeError("no field usable for the mode-rate fit")
    lam, dom, nk, nm = best
    if abs(lam) < 1e-30:
        raise RuntimeError("degenerate mode rate")
    return w/lam, lam, f"fitted from {nm} (rank-1 dominance {dom:.1f}, {nk}/{len(t)} samples)"


a = ap.parse_args()
if a.selftest:
    sys.exit(selftest())
if not a.input:
    sys.exit("--input is required")

P = json.load(open(a.input))
m = int(P["m"])
S2_TRUE = float(P["s2"]); S4_TRUE = float(P["s4"])   # also used by any --log block
b0 = np.deg2rad(a.b0)

snap = a.snapshots or P.get("snapshots_dir", "snapshots")
if not os.path.isabs(snap):
    snap = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(a.input)), snap))
files = sorted(glob.glob(os.path.join(snap, "snapshots_s*.h5")) or
               glob.glob(os.path.join(snap, "*.h5")))
if not files:
    sys.exit(f"no snapshots in {snap}")

tc, Uc, Vc, Hc = [], [], [], []
mu = None
for fp in files:
    with h5py.File(fp, "r") as f:
        if mu is None:
            for nm in f["scales"]:
                if "mu" in nm.lower():
                    arr = np.array(f["scales"][nm][:])
                    if arr.ndim == 1:
                        mu = arr.ravel()
        tt = f["scales/sim_time"][:]
        n = min(len(tt), f["tasks/u_theta"].shape[0])
        k = (tt[:n] >= a.tmin) & (tt[:n] <= a.tmax)
        if not k.any():
            continue
        tc.append(tt[:n][k])
        Uc.append(f["tasks/u_theta"][:n][k])
        Vc.append(f["tasks/u_phi"][:n][k])
        Hc.append(f["tasks/h"][:n][k])
t = np.concatenate(tc)
Uth = np.concatenate(Uc); Uph = np.concatenate(Vc); Hh = np.concatenate(Hc)
o = np.argsort(t)
t, Uth, Uph, Hh = t[o], Uth[o], Uph[o], Hh[o]
N_t, Nphi, N_mu = Uth.shape
phi = 2*np.pi*np.arange(Nphi)/Nphi
dt = float(np.median(np.diff(t)))

print("=" * 74)
print(f"los_inversion  |  {snap}")
print(f"  m={m}  B0={a.b0:g} deg  N_t={N_t}  Nphi={Nphi}  N_mu={N_mu}")
print(f"  t in [{t.min():.1f}, {t.max():.1f}]   noise on v_LOS = {a.noise:g}")
print("=" * 74)

# ── forward: w = dh/dt, then project onto the line of sight ─────────────────
W = np.zeros_like(Hh)
W[2:-2] = (-Hh[4:] + 8*Hh[3:-1] - 8*Hh[1:-3] + Hh[:-4])/(12*dt)
W[:2] = W[2]; W[-2:] = W[-3]                     # edges never enter the fit anyway

p_w, p_th, p_ph = proj(mu[None, :], phi[:, None], b0)      # (Nphi, N_mu)
v = W*p_w[None] + Uth*p_th[None] + Uph*p_ph[None]
visible = p_w > 0                                          # (Nphi, N_mu)
print(f"  visible disk: {100*visible.mean():.0f}% of the (phi,mu) grid")

if a.noise > 0:
    rms = np.sqrt(np.mean(v[:, visible]**2))
    rng = np.random.default_rng(a.noise_seed)
    v = v + a.noise*rms*rng.standard_normal(v.shape)
    print(f"  [noise {a.noise:g}] added to v_LOS at {a.noise:g}x its RMS "
          f"(seed {a.noise_seed})")

# ── inverse: one small least squares per (mu, t) ────────────────────────────
Wr = np.zeros((N_t, N_mu), complex)
Tr = np.zeros((N_t, N_mu), complex)
Pr = np.zeros((N_t, N_mu), complex)
conds = np.full(N_mu, np.nan)
for j in range(N_mu):
    vis = visible[:, j]
    if vis.sum() < 8:
        continue
    M = design(mu[j], phi[vis], m, b0)
    conds[j] = np.linalg.cond(M)
    if a.ridge > 0:
        U_, S_, Vt_ = np.linalg.svd(M, full_matrices=False)
        S_inv = S_/(S_**2 + (a.ridge*S_[0])**2)
        X = (Vt_.T*S_inv) @ U_.T @ v[:, vis, j].T
    else:
        X, *_ = np.linalg.lstsq(M, v[:, vis, j].T, rcond=None)
    Wr[:, j] = X[0] + 1j*X[1]
    Tr[:, j] = X[2] + 1j*X[3]
    Pr[:, j] = X[4] + 1j*X[5]

good = np.isfinite(conds)
print("\n  per-mu condition number of the 6-column design matrix:")
print(f"{'latitude':>10} {'cond':>12}")
for lat in (0, 15, 30, 45, 60, 75):
    j = int(np.argmin(np.abs(np.abs(mu) - np.sin(np.deg2rad(lat)))))
    print(f"{lat:>9}° {conds[j]:>12.3e}")
print(f"   median {np.nanmedian(conds):.3e}   max {np.nanmax(conds):.3e}")
if np.nanmax(conds) > 1e6:
    print("   *** ILL-CONDITIONED — a plain inversion cannot separate the fields.")
    if abs(a.b0) < 1e-6:
        print("       At B0=0 this is EXPECTED AND GLOBAL, not just at the equator:")
        print("       p_th = (mu/st)*p_w exactly, so the w and u_theta columns are")
        print("       PROPORTIONAL at every latitude and only the combination")
        print("       st*w + mu*u_theta is observable. B0=0 is a SINGULAR geometry.")
        print("       cond ~ 100/B0[deg]; use --b0 7.25 (or any |B0| >~ 1 deg).")
    else:
        print("       Try a larger |--b0|, or --ridge, and compare.")

# ── w -> h, and the direct m-modes as the reference ──────────────────────────
mm = lambda F: np.fft.fft(F, axis=1)[:, m, :]/Nphi
Uth_true, Uph_true, H_true = mm(Uth), mm(Uph), mm(Hh)
_lam_ov = (None if (a.lambda_re is None or a.lambda_im is None)
           else a.lambda_re + 1j*a.lambda_im)
H_rec, lam, _lam_src = integrate_w(Wr, t, amp_frac=a.amp_frac,
                                   lam_override=_lam_ov,
                                   others=[("u_theta", Tr), ("u_phi", Pr)])
print(f"\n  w -> h: lambda = {lam.real:+.5f} {lam.imag:+.5f}i   [{_lam_src}]")
print("      (all three fields share one mode rate; w is the worst-recovered,")
print("       so lambda is taken from whichever field is cleanest)")

def relerr(A, B):
    """Relative L2 over the INTERIOR time rows only.
    The first/last two rows have no valid 4th-order dh/dt stencil and are padded
    above; the downstream closed-form recovery masks exactly those rows, so they
    never enter any fit. Including them would report a ~12% error that no
    calculation ever sees."""
    A, B = A[2:-2], B[2:-2]
    return float(np.linalg.norm(A-B)/(np.linalg.norm(B)+1e-30))


print("\n  inversion accuracy vs the direct m-modes (relative L2, interior rows):")
print(f"    u_theta {relerr(Tr, Uth_true):.3e}")
print(f"    u_phi   {relerr(Pr, Uph_true):.3e}")
print(f"    h       {relerr(H_rec, H_true):.3e}")
print("    (edge rows excluded: no valid dh/dt stencil there, and the recovery"
      " masks them)")

os.makedirs(a.out, exist_ok=True)
fp_out = os.path.join(a.out, "pinn_fields.h5")   # same name/format as the PINN ckpt
with h5py.File(fp_out, "w") as f:
    f.attrs.update(dict(m=m, G=float(P["G"]),
                        s2_true=float(P["s2"]), s4_true=float(P["s4"]),
                        nu4=float(P.get("nu4", 0.0)), nu4_h=float(P.get("nu4_h", 0.0)),
                        edge_trim=int(P.get("edge_trim", 8)),
                        noise=a.noise, noise_seed=a.noise_seed,
                        b0_deg=a.b0, ridge=a.ridge, source="los_inversion"))
    f.create_dataset("mu", data=mu)
    f.create_dataset("t", data=t)
    for nm, pred, true in (("u_theta", Tr, Uth_true),
                           ("u_phi", Pr, Uph_true),
                           ("h", H_rec, H_true)):
        g = f.create_group(nm)
        g.create_dataset("pred_r", data=np.real(pred)); g.create_dataset("pred_i", data=np.imag(pred))
        g.create_dataset("true_r", data=np.real(true)); g.create_dataset("true_i", data=np.imag(true))
print(f"\n  wrote {fp_out}")
print("  now run the SAME closed-form recovery on both fields:")
print(f"    lsq_from_data.py --input {a.input} --ckpt {a.out} --field true "
      f"--tmin {a.tmin if np.isfinite(a.tmin) else 10} --mask-poles 60")
print(f"    lsq_from_data.py --input {a.input} --ckpt {a.out} --field pred "
      f"--tmin {a.tmin if np.isfinite(a.tmin) else 10} --mask-poles 60")
print("=" * 74)