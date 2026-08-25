#!/usr/bin/env python3
"""
lsq_from_data.py — closed-form (s2, s4) recovery read DIRECTLY from Dedalus
snapshots. No PINN, no GPU, no DeepXDE. Runs in seconds on a laptop.

Why this exists
---------------
The PDE residual is EXACTLY linear in (s2, s4):   R = c0 + s2*c2 + s4*c4
so recovery is a linear least-squares solve, not an optimisation. On clean,
dense simulation data you do not need a network at all — this script IS the
method. (The PINN earns its place only on sparse/noisy solar data, where the
field must be reconstructed before it can be differentiated.)

Use it to validate the METHOD across every (s2, s4) case in seconds, then
spend GPU time on the PINN reconstruction for only one or two cases.

Usage
-----
    python3 lsq_from_data.py --input ../s2=0.19_s4=0.23/input.json
    python3 lsq_from_data.py --input input.json --scan
    python3 lsq_from_data.py --input input.json --tmin 50 --tmax 536
"""
import argparse, glob, json, os, sys, datetime
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="input.json",
                help="Dedalus input.json (reads m, s2, s4, G, and t_win_min/max if present)")
ap.add_argument("--snapshots", default=None,
                help="snapshots dir (default: 'snapshots' next to input.json)")
ap.add_argument("--tmin", type=float, default=None, help="override t_win_min")
ap.add_argument("--tmax", type=float, default=None, help="override t_win_max")
ap.add_argument("--edge-trim", type=int, default=8)
ap.add_argument("--cheb", type=int, default=20,
                help="Chebyshev truncation for d/dmu. Default 20. **THE RIGHT VALUE "
                     "IS REGIME-DEPENDENT — this default is tuned for DEGRADED "
                     "(noisy / limb-weighted / masked) data, the solar-facing case.** "
                     "Under limb-weight+mask-poles, cheb=20 is the only survivor: 30 "
                     "and 40 are destroyed (s4 err 140%/313% at noise 0.1). But on "
                     "CLEAN G=0.25 data the opposite holds — cheb=20 gave residual "
                     "0.48 and cheb=30 gave 6.2e-3, because nu4_h>0 puts a 4th "
                     "mu-derivative in the residual that lives in the high modes 20 "
                     "truncates away. Use 30 for clean data, 20 for degraded. "
                     "Re-scan whenever the highest derivative order changes.")
ap.add_argument("--svd", type=int, default=2,
                help="SVD rank truncation (0=off). Default 2: chosen TRUTH-FREE by "
                     "pooled median eq-disagreement across the limb+mask runs, and it "
                     "has the best worst case of the good group (max s4 err 1.73% vs "
                     "6.54% for rank 3). NOTE rank 2 is for NOISY fields — on CLEAN "
                     "data svd=0 wins, since denoising can only destroy signal when "
                     "there is no noise to remove.")
ap.add_argument("--scan", action="store_true", help="scan svd_rank x cheb_trunc")
ap.add_argument("--dropout", type=float, default=0.0, metavar="FRAC",
                help="AXIS 3, IRREGULAR SAMPLING. Randomly DISCARD this fraction of "
                     "snapshots, leaving a NON-UNIFORM time grid. This is the "
                     "degradation the closed-form LSQ is structurally least able to "
                     "absorb: the Chebyshev mu-matrix needs a full regular grid, and "
                     "the time derivative is a fixed-dt stencil. --tstride keeps the "
                     "grid uniform (it only coarsens it), so it does NOT test this. "
                     "Real data has dropped frames and bad-quality rejections.")
ap.add_argument("--dropout-seed", type=int, default=0,
                help="RNG seed for --dropout, so a run is reproducible")
ap.add_argument("--gap", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                help="AXIS 3, IRREGULAR SAMPLING. Remove a CONTIGUOUS block of time "
                     "[T0,T1] — instrument downtime / a data outage, which is the "
                     "realistic failure mode rather than random dropout. Composes "
                     "with --dropout.")
ap.add_argument("--nonuniform-dt", choices=["auto", "fornberg", "fixed"], default="auto",
                help="how to differentiate in time. 'fixed' is the classic 4th-order "
                     "stencil, valid ONLY on a uniform grid. 'fornberg' solves the "
                     "local Vandermonde system on the ACTUAL sample times, so it stays "
                     "4th-order on an arbitrary grid. 'auto' (default) uses fixed when "
                     "the grid is uniform and fornberg when it is not. Keep this on "
                     "auto: it gives the LSQ its best shot on irregular data, so a "
                     "failure there is a REAL limitation and not just the wrong "
                     "formula being applied.")
ap.add_argument("--tstride", type=int, default=1,
                help="use every Nth snapshot (temporal-resolution convergence test: "
                     "if results move between --tstride 1 and 2, dt4 is under-resolved)")
ap.add_argument("--no-hyper", action="store_true",
                help="drop the nu4/nu4_h hyperviscosity terms from the residual "
                     "(reproduces the old, incomplete model — for bias comparison only)")
ap.add_argument("--eq-subset", default=None, metavar="DIGITS",
                help="which equations enter the stacked system, as digits from "
                     "{1,2,5}: 1=u_theta momentum, 2=u_phi momentum, 5=continuity/h. "
                     "Default 125 (all). '12' == --velocity-only. Use '1' or '2' "
                     "alone to test whether the fit is really being carried by ONE "
                     "equation: in the LOS-inversion runs eq1's residual saturates "
                     "(0.92 at noise 0.01, 1.00 at 0.05) while eq2's stays low "
                     "(0.05, 0.21), which would mean the apparent two-equation "
                     "redundancy is illusory and the error bars are optimistic.")
ap.add_argument("--velocity-only", action="store_true",
                help="OPTION 1 (partial solar-transfer check): drop the continuity "
                     "equation's (eq5/h) ROWS from the LSQ system, recovering s2/s4 "
                     "from the two momentum equations alone. Tests whether eq5 is "
                     "REDUNDANT for identifiability. NOTE: this is NOT the full "
                     "'h unobserved' test — eq1/eq2 still use the true h field "
                     "internally in their known terms (via -G*sinth*dmu(h) and "
                     "+G*ist*im*h, the pressure coupling). The full test, where h is "
                     "never observed anywhere, needs the PINN's latent-h "
                     "reconstruction and is a PINN-stage task (option 2).")
ap.add_argument("--mask-poles", type=float, default=0.0, metavar="LAT_DEG",
                help="AXIS 3 degradation test (solar-specific). Discard data poleward "
                     "of |latitude| = LAT_DEG before the fit, i.e. keep only "
                     "|mu| <= sin(LAT_DEG). Models HMI's inability to see the poles "
                     "(foreshortening + line-of-sight projection near the limb). "
                     "E.g. --mask-poles 60 keeps the equatorward |lat|<60 band "
                     "(|mu|<0.866). Sweep 70/60/50/40 to see recovery degrade as the "
                     "observable latitude band shrinks. This COMPOSES with the existing "
                     "PDE_MU_MAX equator-band mask (the stricter of the two wins). "
                     "Default 0 = no pole mask.")
ap.add_argument("--noise", type=float, default=0.0,
                help="AXIS 3 degradation test. Add Gaussian noise to the REAL-SPACE "
                     "(theta,phi) snapshots BEFORE the azimuthal FFT, at amplitude "
                     "NOISE * (per-field real-space RMS). This is where a telescope's "
                     "noise sits: broadband in phi, so the m-mode FFT then suppresses "
                     "most of it -- injecting after the FFT would be free denoising the "
                     "Sun never gives. Sweep e.g. --noise 0.001 0.01 0.1 (separate runs) "
                     "to see recovery degrade; the point at which it breaks is where the "
                     "PINN (which reconstructs a smooth field before differentiating) "
                     "earns its place. Default 0 = clean, unchanged behaviour.")
ap.add_argument("--limb-weight", choices=["off", "best", "mean"], default="off",
                help="AXIS 3, SOLAR-FACING degradation (supersedes uniform --noise as "
                     "the realistic test). Scales the injected noise per CHANNEL by the "
                     "inverse of that channel's line-of-sight sensitivity at B0=0. "
                     "Geometry: cos(rho) = cos(lat)*cos(phi'), LOS velocity = "
                     "w*cos(rho) + u_horiz*sin(rho); extracting a component divides by "
                     "its projection, so noise scales as 1/sensitivity. h is carried by "
                     "w = dh/dt (the free-surface kinematic condition), hence the "
                     "vertical channel. NOTE the limb is a great circle passing through "
                     "the EQUATOR at phi'=+-90, not only through the poles, so the "
                     "horizontal channels reach full sensitivity at EVERY latitude — "
                     "the asymmetry is one-sided. "
                     "'best' = best view over the visible disk (h ~ 1/cos(lat) degrading "
                     "poleward, velocities uniform); 'mean' = averaged over visible "
                     "phi' (pessimistic: also penalises velocities mildly near the "
                     "equator). Requires --noise > 0.")
ap.add_argument("--limb-floor", type=float, default=0.1,
                help="floor on the sensitivity so the 1/sensitivity noise does not "
                     "diverge where a projection -> 0 (the h channel exactly at the "
                     "poles). Default 0.1 = at most a 10x noise amplification.")
ap.add_argument("--noise-seed", type=int, default=0,
                help="RNG seed for --noise, so a noisy run is reproducible. Change it "
                     "to check the recovery is not a lucky draw (run the same --noise at "
                     "several seeds; the spread in s2/s4 is the noise-induced uncertainty).")
ap.add_argument("--ckpt", default=None,
                help="read the field from a PINN checkpoint dir (holding "
                     "pinn_fields.h5) INSTEAD of from raw snapshots. This lets the "
                     "IDENTICAL recovery — same mask-poles, cheb, svd, Fornberg dt and "
                     "eq-consistency selector — be applied to the PINN field and to the "
                     "raw field it was trained on, so a PINN-vs-raw comparison differs "
                     "in the FIELD alone. Degradation flags (--noise/--limb-weight/"
                     "--tstride) are IGNORED here: that degradation is already baked "
                     "into the checkpoint by the PINN run.")
ap.add_argument("--field", choices=["pred", "true"], default="pred",
                help="with --ckpt: 'pred' = the PINN's reconstruction, 'true' = the "
                     "(already degraded) field it was given. Run both for the "
                     "like-for-like comparison.")
ap.add_argument("--log", default=None,
                help="path to save the report "
                     "(default: lsq_from_data_s2=<s2>_s4=<s4>[_scan].log; use 'none' to disable)")
a = ap.parse_args()

P = json.load(open(a.input))
m = int(P["m"]); G = float(P["G"])
S2_TRUE = float(P["s2"]); S4_TRUE = float(P["s4"])
EPS_REG = float(P.get("eps_reg", 0.05))
# Hyperviscosity coefficients. These are KNOWN constants, so they belong in c0
# (the s2/s4-independent part), NOT as extra unknowns. Absent => 0 => the old
# G=0.01 behaviour is reproduced byte-for-byte.
NU4   = 0.0 if a.no_hyper else float(P.get("nu4",   0.0))
NU4_H = 0.0 if a.no_hyper else float(P.get("nu4_h", 0.0))
VEL_ONLY = a.velocity_only   # OPTION 1: drop eq5 (continuity) rows from the LSQ
# --eq-subset generalises --velocity-only; resolve both into ONE equation set.
if a.eq_subset is not None:
    _bad = set(a.eq_subset) - set("125")
    if _bad:
        raise SystemExit(f"--eq-subset: unknown equation(s) {sorted(_bad)}; "
                         f"use digits from 1,2,5")
    EQ_SET = set(a.eq_subset)
    if not EQ_SET:
        raise SystemExit("--eq-subset: pick at least one equation")
elif VEL_ONLY:
    EQ_SET = {"1", "2"}
else:
    EQ_SET = {"1", "2", "5"}
MASK_POLES = a.mask_poles    # AXIS 3: discard |lat|>MASK_POLES deg (0 = keep all)
tmin = a.tmin if a.tmin is not None else float(P.get("t_win_min", -np.inf))
tmax = a.tmax if a.tmax is not None else float(P.get("t_win_max", np.inf))
snap = a.snapshots
if snap is None:
    snap = P.get("snapshots_dir", "snapshots")
    if not os.path.isabs(snap):
        snap = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(a.input)), snap))

# ── Save a copy of everything printed, to a log file ──────────────────────────
class _Tee:
    """Mirror stdout to a file so the whole report is saved verbatim."""
    def __init__(self, path):
        self.f = open(path, "w")
    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)
    def flush(self):
        sys.__stdout__.flush(); self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass

log_path = a.log
if log_path is None:
    log_path = (f"lsq_from_data_s2={S2_TRUE:g}_s4={S4_TRUE:g}"
                f"{'_scan' if a.scan else ''}.log")
_tee = None
if str(log_path).lower() != "none":
    _tee = _Tee(log_path)
    sys.stdout = _tee

print("="*72)
print(f"lsq_from_data  |  {a.input}")
print(f"  run at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  m={m}  G={G}  true s2={S2_TRUE}  s4={S4_TRUE}")
print(f"  nu4={NU4:.3e}  nu4_h={NU4_H:.3e}"
      + ("   [--no-hyper: terms DROPPED from residual]" if a.no_hyper else ""))
print(f"  snapshots: {snap}")
print(f"  window: t=[{tmin}, {tmax}]   edge_trim={a.edge_trim}")
# The cheb=20 / svd=2 defaults are tuned for DEGRADED data. On clean data they
# are the wrong regime: cheb=20 gave residual 0.48 vs cheb=30's 6.2e-3 (the 4th
# mu-derivative from nu4_h lives in the modes 20 truncates), and svd>0 can only
# destroy signal when there is no noise to remove. Say so rather than let a
# clean-data run quietly return a bad number.
if (not a.scan and a.noise == 0 and a.limb_weight == "off"
        and (a.cheb == 20 or a.svd > 0)):
    print("  [NOTE] this looks like a CLEAN run (no --noise, no --limb-weight) but "
          "the")
    print("         degraded-data defaults are in use (cheb=20, svd=2). On clean "
          "data")
    print("         prefer --cheb 30 --svd 0; cheb=20 gave residual 0.48 there "
          "vs 6.2e-3.")
if a.noise > 0:
    print(f"  [--noise {a.noise:g}] Gaussian noise added to real-space (phi,mu) fields")
    print(f"      before the m-FFT, at {a.noise:g}x each field's RMS, seed={a.noise_seed}.")
    print(f"      AXIS 3 degradation test — broadband phi noise, FFT-then-SVD/cheb denoise.")
    if a.limb_weight != "off":
        # NOTE: this banner runs BEFORE the snapshots are loaded, so `mu` does
        # not exist yet — report the cap analytically (it is 1/floor by
        # construction) rather than measuring it off the grid.
        print(f"  [--limb-weight {a.limb_weight}] centre-to-limb sensitivity at B0=0, "
              f"floor={a.limb_floor:g}")
        print(f"      h (vertical, w=dh/dt): noise x 1/cos(lat), growing poleward, "
              f"capped at {1.0/a.limb_floor:.1f}x")
        print(f"      u_theta,u_phi (horizontal): "
              + ("uniform — full sensitivity reachable at every latitude via the limb"
                 if a.limb_weight == "best" else
                 "mild equatorward penalty (phi'-averaged)"))
if VEL_ONLY and a.eq_subset is None:
    print("  [--velocity-only] eq5 (continuity/h) rows DROPPED; s2/s4 from momentum")
    print("      eqns only. Partial solar-transfer check — h still used inside eq1/eq2.")
elif EQ_SET != {"1", "2", "5"}:
    _nm = {"1": "u_theta momentum", "2": "u_phi momentum", "5": "continuity/h"}
    print(f"  [equations {''.join(sorted(EQ_SET))}] stacking only: "
          + ", ".join(_nm[d] for d in sorted(EQ_SET)))
    if len(EQ_SET) == 1:
        print("      SINGLE EQUATION — no cross-equation redundancy at all.")
        print("      Diagnostic only, not a recommended configuration.")
if MASK_POLES > 0:
    print(f"  [--mask-poles {MASK_POLES:g}] keeping |lat|<{MASK_POLES:g} deg "
          f"(|mu|<{np.sin(np.deg2rad(MASK_POLES)):.3f}); poleward data discarded.")
    print("      AXIS 3 degradation test — models HMI's missing polar coverage.")
if not np.isfinite(tmax):
    print("  [WARNING] t_win_max not set — if this run has a growth onset, the")
    print("            blow-up region will corrupt the fit. Set it from postrun_analyze.")
print("="*72)


def _scale(f, sub):
    for n in f["scales"].keys():
        if sub in n.lower():
            arr = np.array(f["scales"][n][:])
            if arr.ndim == 1:
                return arr.flatten()
    raise KeyError(sub)


if a.ckpt:
    # ---- PINN checkpoint path -------------------------------------------
    # pinn_fields.h5 already stores the m-mode projected complex fields on the
    # (t, mu) grid, i.e. exactly what the recovery below consumes. Nothing is
    # re-degraded: noise/limb-weight/tstride were applied by the PINN run.
    _cf = os.path.join(a.ckpt, "pinn_fields.h5")
    if not os.path.exists(_cf):
        raise FileNotFoundError(f"{_cf} not found")
    with h5py.File(_cf, "r") as _f:
        mu = _f["mu"][:]; t = _f["t"][:]
        _k = a.field
        Uth = _f["u_theta"][f"{_k}_r"][:] + 1j*_f["u_theta"][f"{_k}_i"][:]
        Uph = _f["u_phi"][f"{_k}_r"][:]   + 1j*_f["u_phi"][f"{_k}_i"][:]
        H   = _f["h"][f"{_k}_r"][:]       + 1j*_f["h"][f"{_k}_i"][:]
        _at = dict(_f.attrs)
    print(f"  [--ckpt] field='{a.field}' from {_cf}")
    print(f"      the checkpoint was made with: noise={_at.get('noise','?')} "
          f"seed={_at.get('noise_seed','?')} limb={_at.get('limb_weight','?')} "
          f"tstride={_at.get('tstride','?')}")
    if a.noise > 0 or a.limb_weight != "off" or a.tstride > 1:
        print("      [NOTE] degradation flags are IGNORED with --ckpt — the field is "
              "already degraded")
    _m = (t >= tmin) & (t <= tmax)
    t, Uth, Uph, H = t[_m], Uth[_m], Uph[_m], H[_m]
else:
  files = sorted(glob.glob(os.path.join(snap, "snapshots_s*.h5")) or
               glob.glob(os.path.join(snap, "*.h5")))
  if not files:
    raise FileNotFoundError(f"no .h5 in {snap}")
  with h5py.File(files[0], "r") as f0:
      mu = _scale(f0, "mu")
      have = list(f0["tasks"].keys())
  for r in ("u_theta", "u_phi", "h"):
      if r not in have:
          raise RuntimeError(f"tasks/{r} missing — found {have}")

  tc, Uc, Vc, Hc = [], [], [], []
  for fp in files:
      with h5py.File(fp, "r") as f:
          tt = f["scales/sim_time"][:]
          # A RUNNING Dedalus job may be mid-write: the task datasets can carry
          # one more row than the sim_time scale (or vice versa) because HDF5
          # has not flushed both yet. Truncate to the common length and drop the
          # partial tail instead of crashing, so the recovery can be run against
          # a live run's snapshots without waiting for it to finish.
          lens = [len(tt)] + [f[f"tasks/{n}"].shape[0]
                              for n in ("u_theta", "u_phi", "h")]
          n_ok = min(lens)
          if len(set(lens)) > 1:
              print(f"  [live file] {os.path.basename(fp)}: lengths {lens} -> "
                    f"using {n_ok} (job still writing; partial tail dropped)")
          tt = tt[:n_ok]
          k = (tt >= tmin) & (tt <= tmax)
          if not k.any():
              continue
          nphi = f["tasks/u_theta"].shape[1]
          if a.noise > 0:
              # Noise on the REAL-SPACE (t, phi, mu) field, before the m-FFT.
              # Amplitude is relative to each field's own real-space RMS, so the
              # same --noise means the same fractional corruption in every field.
              # A fresh, deterministic RNG per file keeps runs reproducible while
              # ensuring no two files share the identical noise draw.
              # Deterministic per-file offset so files differ but the run is
              # REPRODUCIBLE. Python's built-in hash() on str is randomized per
              # process (PYTHONHASHSEED), which would silently break --noise-seed;
              # use a stable hash of the filename instead.
              import zlib
              _off = zlib.crc32(os.path.basename(fp).encode()) & 0xffff
              _rng = np.random.default_rng(a.noise_seed + _off)

              # ── centre-to-limb sensitivity (B0 = 0) ──────────────────────────
              # cos(rho) = cos(lat)*cos(phi'), and mu = sin(lat) in this
              # convention, so cos(lat) = sqrt(1-mu^2).
              #   vertical  channel (h, carried by w = dh/dt): sensitivity cos(rho)
              #   horizontal channels (u_theta, u_phi)       : sensitivity sin(rho)
              # Noise scales as 1/sensitivity because extracting a component
              # divides the LOS signal by its projection.
              #
              # CRITICAL GEOMETRY: the limb is the great circle at rho=90deg, which
              # passes through the EQUATOR at phi'=+-90deg — not only through the
              # poles. So sin(rho)=1 is reachable at EVERY latitude, and the
              # horizontal channels are never blind. Only the vertical channel is
              # genuinely one-sided: max cos(rho) = cos(lat), which -> 0 at the
              # poles because a pole is never seen face-on.
              def _limb_weight(name):
                  if a.limb_weight == "off":
                      return 1.0
                  _clat = np.sqrt(np.maximum(1.0 - mu**2, 0.0))    # cos(latitude)
                  if a.limb_weight == "best":
                      # best view available anywhere on the visible disk
                      sens = _clat if name == "h" else np.ones_like(_clat)
                  else:                                            # "mean"
                      # averaged over visible phi' in (-90, 90)
                      php = np.linspace(-np.pi/2, np.pi/2, 181)[None, :]
                      cr = _clat[:, None]*np.cos(php)
                      sens = (cr.mean(1) if name == "h"
                              else np.sqrt(np.maximum(1-cr**2, 0.0)).mean(1))
                  sens = np.maximum(sens, a.limb_floor)
                  return (1.0/sens)[None, None, :]                 # broadcast over (t, phi)

              def mm(name):
                  raw = f[f"tasks/{name}"][:n_ok]                 # (n_ok, Nphi, Nmu) real
                  rms = np.sqrt(np.mean(raw**2))
                  raw = raw + a.noise * rms * _limb_weight(name) * _rng.standard_normal(raw.shape)
                  return np.fft.fft(raw, axis=1)[:, m, :]/nphi
          else:
              mm = lambda name: np.fft.fft(f[f"tasks/{name}"][:n_ok], axis=1)[:, m, :]/nphi
          tc.append(tt[k]); Uc.append(mm("u_theta")[k])
          Vc.append(mm("u_phi")[k]); Hc.append(mm("h")[k])
  if not tc:
      raise RuntimeError(f"no snapshots in t=[{tmin},{tmax}]")
  t = np.concatenate(tc); Uth = np.concatenate(Uc); Uph = np.concatenate(Vc); H = np.concatenate(Hc)
  o = np.argsort(t); t, Uth, Uph, H = t[o], Uth[o], Uph[o], H[o]
if a.tstride > 1 and not a.ckpt:
    t, Uth, Uph, H = t[::a.tstride], Uth[::a.tstride], Uph[::a.tstride], H[::a.tstride]
    print(f"  [tstride={a.tstride}] subsampled in time -> effective snap_dt "
          f"{np.median(np.diff(t)):.3f}   (grid stays UNIFORM)")

# ── irregular temporal sampling ──────────────────────────────────────────────
# Applied AFTER tstride so the two compose. Order matters only cosmetically.
if a.gap is not None and not a.ckpt:
    _g0, _g1 = a.gap
    _keep = ~((t >= _g0) & (t <= _g1))
    _nrm = int((~_keep).sum())
    t, Uth, Uph, H = t[_keep], Uth[_keep], Uph[_keep], H[_keep]
    print(f"  [--gap {_g0:g} {_g1:g}] removed a contiguous block: {_nrm} snapshots")
if a.dropout > 0 and not a.ckpt:
    _rng = np.random.default_rng(a.dropout_seed)
    _keep = _rng.random(len(t)) >= a.dropout
    _keep[0] = _keep[-1] = True          # keep the endpoints so the span is unchanged
    _nrm = int((~_keep).sum())
    t, Uth, Uph, H = t[_keep], Uth[_keep], Uph[_keep], H[_keep]
    print(f"  [--dropout {a.dropout:g}] discarded {_nrm} snapshots at random "
          f"(seed={a.dropout_seed})")
if (a.gap is not None or a.dropout > 0) and not a.ckpt:
    _d = np.diff(t)
    print(f"      time grid is now NON-UNIFORM: dt min/median/max = "
          f"{_d.min():.3f} / {np.median(_d):.3f} / {_d.max():.3f}  "
          f"(spread {_d.max()/max(_d.min(),1e-30):.1f}x)")
    print(f"      -> the fixed-dt 4th-order stencil is INVALID here; "
          f"--nonuniform-dt={a.nonuniform_dt} governs what is used instead")
N_t, N_mu = len(t), len(mu)
print(f"  loaded N_t={N_t}, N_mu={N_mu}, t in [{t[0]:.1f}, {t[-1]:.1f}]\n")

mu_sorted = np.sort(mu)
PDE_MU_MAX = float(mu_sorted[-a.edge_trim - 1])
_dt = float(np.median(np.diff(t)))

# ── time derivative on a possibly NON-UNIFORM grid ───────────────────────────
# The classic stencil (-f[i-2]+8f[i-1]-8f[i+1]+f[i+2])/(12*dt) assumes a single
# constant dt. Under --dropout/--gap that is simply false, and using it would
# make the LSQ fail for a trivial reason (wrong formula) rather than a real one.
# Fornberg's method instead solves, for each point, the local Vandermonde system
#     sum_j w_j * (t_j - t_i)^k = k! * delta_{k,1}
# over a 5-point neighbourhood, giving 4th-order-accurate weights on ANY grid.
# On a uniform grid it reproduces the classic stencil exactly (verified below).
_UNIFORM = bool((np.diff(t).max() - np.diff(t).min()) / max(_dt, 1e-30) < 1e-6)
_USE_FORNBERG = (a.nonuniform_dt == "fornberg" or
                 (a.nonuniform_dt == "auto" and not _UNIFORM))


def _fornberg_dt_matrix(tt, half=2):
    """First-derivative weights on an arbitrary grid; (2*half+1)-point stencil."""
    n = len(tt); w = 2*half + 1
    D = np.zeros((n, n))
    for i in range(n):
        lo = max(0, min(i - half, n - w))          # shift (one-sided) near the ends
        idx = np.arange(lo, lo + w)
        dx = tt[idx] - tt[i]
        V = np.vander(dx, w, increasing=True).T    # V[k,j] = dx_j**k
        rhs = np.zeros(w); rhs[1] = 1.0            # want sum_j w_j dx_j^k = k! d_{k1}
        D[i, idx] = np.linalg.solve(V, rhs)
    return D


_D_t = _fornberg_dt_matrix(t) if _USE_FORNBERG else None
if _USE_FORNBERG:
    print(f"  [dt] NON-UNIFORM grid -> Fornberg weights on the actual sample times "
          f"(4th-order on an arbitrary grid)")
elif not _UNIFORM:
    print(f"  [dt] WARNING: grid is non-uniform but --nonuniform-dt=fixed was forced "
          f"— the time derivative is WRONG; any failure is the formula, not the method")
_ord = np.argsort(mu); _inv = np.argsort(_ord); _mus = mu[_ord]

if MASK_POLES > 0:
    # Report what the fit actually KEEPS, measured off the grid. Deriving this
    # figure by hand has gone wrong before: "% of domain" is ambiguous between
    # grid columns (what the fit loses) and sphere area (a geometric measure),
    # and the two differ a lot because the Gauss-Chebyshev grid is uniform in
    # colatitude, so the poles are geometrically small but hold a
    # disproportionate share of grid points. Print both, measured.
    _keep = np.abs(mu) <= min(np.sin(np.deg2rad(MASK_POLES)), PDE_MU_MAX)
    _base = np.abs(mu) <= PDE_MU_MAX
    print(f"  [--mask-poles {MASK_POLES:g}] retained mu-columns: {_keep.sum()}/{_base.sum()}"
          f" = {100*_keep.sum()/max(_base.sum(),1):.0f}% of the fitted columns"
          f"   (sphere area kept = {100*np.sin(np.deg2rad(MASK_POLES)):.0f}%)")
    print("      quote the COLUMN figure for 'how much data the fit lost'.\n")


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


def lsq(Uth, Uph, H, K, r, diag=False, drop_eq5=None):
    Uth, Uph, H = svd_trunc(Uth, r), svd_trunc(Uph, r), svd_trunc(H, r)
    Dmu = cheb_trunc_D(_mus, K)
    dmu = lambda F: (F[:, _ord] @ Dmu.T)[:, _inv]
    # 4th mu-derivative for the biharmonic filters. Dmu already truncates its
    # input to K Chebyshev modes; since d/dmu of a K-mode field has <K modes,
    # chaining is equivalent to "truncate once, then differentiate 4x".
    D4 = Dmu @ Dmu @ Dmu @ Dmu
    dmu4 = lambda F: (F[:, _ord] @ D4.T)[:, _inv]
    def dt4(A):
        if _USE_FORNBERG:
            d = _D_t @ A
            d[:2] = 0; d[-2:] = 0        # same edge handling as the fixed stencil
            return d
        d = np.zeros_like(A)
        d[2:-2] = (-A[4:] + 8*A[3:-1] - 8*A[1:-3] + A[:-4])/(12*_dt)
        return d
    mu2 = mu[None, :]
    st = np.sqrt(np.maximum(1-mu2**2, 0)); ist = 1/(st+EPS_REG)
    a2, a4 = -mu2**2, -mu2**4; imm = 1j*m
    # Hyperviscosity enters the LHS with a PLUS sign, matching the Dedalus
    # equations:  ... + nu4*dmu(dmu(s_theta)) = 0   with  s_theta = d2/dmu2 u.
    hv_1 = NU4*dmu4(Uth)
    hv_2 = NU4*dmu4(Uph)
    hv_5 = NU4_H*dmu4(H)
    c0_1 = dt4(Uth) - 2*mu2*Uph - G*st*dmu(H) + hv_1
    c0_2 = dt4(Uph) + 2*mu2*Uth + G*ist*imm*H + hv_2
    c0_5 = dt4(H) + ist*(mu2*Uth - (1-mu2**2)*dmu(Uth) + imm*Uph) + hv_5
    c2_1 = a2*imm*Uth - 2*mu2*a2*Uph
    c2_2 = a2*imm*Uph + (2*mu2*a2 + 2*mu2*(1-mu2**2))*Uth
    c2_5 = a2*imm*H
    c4_1 = a4*imm*Uth - 2*mu2*a4*Uph
    c4_2 = a4*imm*Uph + (2*mu2*a4 + 4*mu2**3*(1-mu2**2))*Uth
    c4_5 = a4*imm*H
    msk = np.zeros((N_t, N_mu), bool)
    keep_mu = np.abs(mu) <= PDE_MU_MAX
    if MASK_POLES > 0:
        # Discard poleward of |lat|=MASK_POLES deg. |mu|=|sin(lat)|, so keep
        # |mu| <= sin(MASK_POLES). Composes with PDE_MU_MAX: the stricter wins.
        keep_mu = keep_mu & (np.abs(mu) <= np.sin(np.deg2rad(MASK_POLES)))
    msk[:, keep_mu] = True
    msk[:2, :] = False; msk[-2:, :] = False
    # OPTION 1 (--velocity-only): stack only eq1 & eq2 (the momentum equations),
    # dropping eq5 (continuity). h STILL enters via c0_1/c0_2's pressure terms
    # -G*st*dmu(H) and +G*ist*imm*H above -- so this is a redundancy test on eq5,
    # not a true 'h unobserved' test. The x5 argument is simply omitted from the
    # stack when VEL_ONLY is set.
    # drop_eq5 overrides the global per call, so the truth-free selector can ask
    # for BOTH stacks on the same field without touching the global.
    # drop_eq5 (used by the truth-free selector) still overrides, and is
    # expressed through the same equation set.
    _eqs = EQ_SET if drop_eq5 is None else (EQ_SET - {"5"} if drop_eq5 else EQ_SET | {"5"})
    def S(x1, x2, x5):
        parts = [x for d, x in (("1", x1), ("2", x2), ("5", x5)) if d in _eqs]
        v = np.concatenate([p[msk] for p in parts])
        return np.concatenate([v.real, v.imag])
    if diag:
        _rms = lambda F: float(np.sqrt(np.mean(np.abs(F[msk])**2)))
        print("  hyperviscosity term size (RMS, over the fitted mask):")
        print(f"    {'equation':12s} {'|hv|':>10s} {'|dt(field)|':>12s} {'ratio':>9s}")
        for nm, hv, dtf in (("(1) u_theta", hv_1, dt4(Uth)),
                            ("(2) u_phi  ", hv_2, dt4(Uph)),
                            ("(5) h      ", hv_5, dt4(H))):
            rh, rd = _rms(hv), _rms(dtf)
            print(f"    {nm:12s} {rh:10.3e} {rd:12.3e} {rh/(rd+1e-300):8.2%}")
        print("    (ratio >~1%: the term matters; dropping it biases s2/s4)")
        print()
    b = S(c0_1, c0_2, c0_5)
    A = np.column_stack([S(c2_1, c2_2, c2_5), S(c4_1, c4_2, c4_5)])
    sol, *_ = np.linalg.lstsq(A, -b, rcond=None)
    cond = float(np.linalg.cond(A))
    res = float(np.linalg.norm(A@sol + b)/(np.linalg.norm(b)+1e-30))
    if diag:
        # Per-equation residual. The LSQ lumps all three equations into one
        # solve, so a single number hides WHICH equation the model fails in.
        print("  per-equation residual  |R(s_fit)| / |c0|   (0 = model explains data):")
        print(f"    {'equation':12s} {'residual':>10s}")
        for nm, c0, c2, c4 in (("(1) u_theta", c0_1, c2_1, c4_1),
                               ("(2) u_phi  ", c0_2, c2_2, c4_2),
                               ("(5) h      ", c0_5, c2_5, c4_5)):
            R = (c0 + sol[0]*c2 + sol[1]*c4)[msk]
            rr = float(np.linalg.norm(R)/(np.linalg.norm(c0[msk])+1e-300))
            print(f"    {nm:12s} {rr:10.3f}")
        print("    all three high -> time-derivative / model-wide problem")
        print("    only (5) high  -> continuity/h-specific problem")
        print()
    return float(sol[0]), float(sol[1]), cond, res


sv = np.linalg.svd(Uth, compute_uv=False)[:5]
print("  SVD spectrum u_theta: " + " ".join(f"{v/sv[0]:.1e}" for v in sv))
print()

if a.scan:
    print(f"  {'setting':18s} {'s2':>10s} {'err%':>7s}  {'s4':>10s} {'err%':>7s}  "
          f"{'cond':>6s} {'resid':>9s} {'eq-disag':>10s}")
    best = None      # truth-INFORMED (validation only)
    tfree = None     # truth-FREE (the one you could use on solar data)
    for K in (20, 30, 40):
        for r in (0, 2, 3, 4):
            s2, s4, cd, rs = lsq(Uth, Uph, H, K, r)
            # TRUTH-FREE SCORE: equation-subset consistency. Recover again from
            # the momentum equations alone and compare. The three equations
            # weight the mu-derivatives differently (eq5 carries nu4_h*d4h/dmu4
            # and dmu(u_theta); eq1 carries dmu(h) and nu4*d4u/dmu4), so a
            # truncation that mangles high-order derivatives corrupts them
            # unequally and the two answers separate. The velocity-only result
            # already established eq5 is REDUNDANT on good data, so ANY
            # disagreement is diagnostic of a bad SETTING, not of real physics.
            # Uses no true value -- available on solar data, where the error is
            # unknowable.
            v2, v4, _, _ = lsq(Uth, Uph, H, K, r, drop_eq5=True)
            disag = (abs(v2-s2)/max(abs(s2), 1e-30)
                     + abs(v4-s4)/max(abs(s4), 1e-30))
            e2 = abs(s2-S2_TRUE)/abs(S2_TRUE)*100; e4 = abs(s4-S4_TRUE)/abs(S4_TRUE)*100
            print(f"  cheb={K:<3d} svd={r:<2d}    {s2:+10.5f} {e2:6.2f}%  "
                  f"{s4:+10.5f} {e4:6.2f}%  {cd:6.2f} {rs:9.3e} {disag:10.3e}")
            if best is None or e2+e4 < best[0]:
                best = (e2+e4, K, r, s2, s4)
            if tfree is None or disag < tfree[0]:
                tfree = (disag, K, r, s2, s4)
    print("\n" + "="*72)
    print(f"  TRUTH-FREE PICK (eq-consistency): cheb_trunc={tfree[1]}, svd_rank={tfree[2]}")
    print(f"    s2 = {tfree[3]:.6f}  (true {S2_TRUE},  {abs(tfree[3]-S2_TRUE)/abs(S2_TRUE)*100:.2f}%)")
    print(f"    s4 = {tfree[4]:.6f}  (true {S4_TRUE},  {abs(tfree[4]-S4_TRUE)/abs(S4_TRUE)*100:.2f}%)")
    print(f"    eq-disagreement = {tfree[0]:.3e}   (no true value was used to choose this)")
    print("-"*72)
    print(f"  truth-INFORMED BEST (VALIDATION ONLY — needs the answer, so it does")
    print(f"  NOT exist on solar data; also a minimum over 12 noisy estimates,")
    print(f"  which flatters itself more as noise rises): cheb={best[1]}, svd={best[2]}")
    print(f"    s2 = {best[3]:.6f}  ({abs(best[3]-S2_TRUE)/abs(S2_TRUE)*100:.2f}%)   "
          f"s4 = {best[4]:.6f}  ({abs(best[4]-S4_TRUE)/abs(S4_TRUE)*100:.2f}%)")
    _c2 = abs(tfree[3]-S2_TRUE)/abs(S2_TRUE)*100 - abs(best[3]-S2_TRUE)/abs(S2_TRUE)*100
    _c4 = abs(tfree[4]-S4_TRUE)/abs(S4_TRUE)*100 - abs(best[4]-S4_TRUE)/abs(S4_TRUE)*100
    if (tfree[1], tfree[2]) == (best[1], best[2]):
        print("  VERDICT: the truth-free rule picked the optimal setting.")
    else:
        print(f"  VERDICT: different setting; truth-free costs {_c2:+.2f}pp on s2, "
              f"{_c4:+.2f}pp on s4.")
        print("  Small cost => the rule is usable on solar data. Large cost => it is not.")
    print("="*72)
else:
    s2, s4, cd, rs = lsq(Uth, Uph, H, a.cheb, a.svd, diag=True)
    print("="*72)
    print(f"  RECOVERED (closed-form LSQ on the data; cheb={a.cheb}, svd={a.svd})")
    print(f"    s2 = {s2:.6f}   true {S2_TRUE}   ({abs(s2-S2_TRUE)/abs(S2_TRUE)*100:.2f}%)")
    print(f"    s4 = {s4:.6f}   true {S4_TRUE}   ({abs(s4-S4_TRUE)/abs(S4_TRUE)*100:.2f}%)")
    print(f"    cond([c2,c4]) = {cd:.2f}   residual = {rs:.3e}")
    print("="*72)

# ── Close the log ─────────────────────────────────────────────────────────────
if _tee is not None:
    sys.stdout = sys.__stdout__
    _tee.close()
    print(f"[saved] {log_path}")