#!/usr/bin/env python3
# evp_spherical_hd.py
# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue problem for the 2D spherical shallow-water HD model, at a single
# azimuthal wavenumber m.  Equations/BCs come from sw_model.py — NOT duplicated.
#
# What it gives you, in seconds instead of a 3.5 h IVP:
#   * omega for every mode at this (m, G, s2, s4, nu4, nu4_h) — Rossby AND
#     Poincare — so you can see what you are about to excite before you run.
#   * the STABILITY of the discretisation directly: any mode with Im(omega) > 0
#     is the numerical instability you have been bisecting with 3.5 h runs.
#   * a clean eigenvector IC (--write-ic) that excites ONE mode and nothing else.
#
# Sign convention (see sw_model.py):  q ~ exp(i m phi - i omega t)
#   Re(omega) < 0  -> RETROGRADE  -> Rossby
#   Re(omega) > 0  -> PROGRADE
#   Im(omega) > 0  -> GROWING     -> unstable
#
# Usage:
#   python3 evp_spherical_hd.py --input input.json
#   python3 evp_spherical_hd.py --input input.json --write-ic evp_ic.h5
#   python3 evp_spherical_hd.py --input input.json --nu4 3e-5 --nu4-h 1e-4
# ─────────────────────────────────────────────────────────────────────────────
import argparse, json, sys, datetime
import numpy as np
import dedalus.public as d3
import sw_model

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="input.json")
ap.add_argument("--m", type=int, default=None, help="override azimuthal wavenumber")
ap.add_argument("--nu4", type=float, default=None, help="override nu4 (stability scan)")
ap.add_argument("--nu4-h", type=float, default=None, help="override nu4_h (stability scan)")
ap.add_argument("--G", type=float, default=None, help="override G")
ap.add_argument("--nmodes", type=int, default=12, help="how many modes to print per branch")
ap.add_argument("--rtol", type=float, default=1e-4,
                help="resolution-check tolerance for rejecting spurious eigenvalues")
ap.add_argument("--no-resolution-check", action="store_true",
                help="skip the 1.5x resolution cross-check (faster, keeps spurious modes)")
ap.add_argument("--write-ic", default=None,
                help="write the selected eigenvector to this .h5 for the IVP to load")
ap.add_argument("--pick", default="rossby",
                help="which mode to export: 'rossby' (least-damped retrograde), "
                     "'unstable' (largest Im), or an integer index into the printed table")
ap.add_argument("--log", default=None)
a = ap.parse_args()

P = json.load(open(a.input))
if a.nu4 is not None:   P["nu4"] = a.nu4
if a.nu4_h is not None: P["nu4_h"] = a.nu4_h
if a.G is not None:     P["G"] = a.G
m = int(P["m"]) if a.m is None else a.m


class _Tee:
    def __init__(self, path): self.f = open(path, "w")
    def write(self, s): sys.__stdout__.write(s); self.f.write(s)
    def flush(self): sys.__stdout__.flush(); self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass


log_path = a.log or f"evp_m={m}_G={P['G']:g}_nu4={P['nu4']:g}_nu4h={P.get('nu4_h',0):g}.log"
_tee = None
if str(log_path).lower() != "none":
    _tee = _Tee(log_path); sys.stdout = _tee

print("=" * 72)
print(f"evp_spherical_hd  |  {a.input}")
print(f"  run at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  m={m}  G={P['G']}  s2={P['s2']}  s4={P['s4']}")
print(f"  nu4={P['nu4']:.3e}  nu4_h={P.get('nu4_h', 0.0):.3e}  Nmu={P['Nmu']}")
print("=" * 72)


def solve_at(Nmu):
    """Dense EVP solve at a given mu-resolution. Returns (evals, M, solver, sp)."""
    Q = dict(P); Q["Nmu"] = int(Nmu)
    M = sw_model.build(Q, mode="evp", m=m)
    problem = sw_model.make_problem(M)
    solver = problem.build_solver()
    sp = solver.subproblems[0]
    solver.solve_dense(sp)
    ev = np.array(solver.eigenvalues)
    return ev, M, solver, sp


ev_lo, M, solver, sp = solve_at(P["Nmu"])
good = np.isfinite(ev_lo)
print(f"  raw eigenvalues: {ev_lo.size}   finite: {good.sum()}")

if not a.no_resolution_check:
    # Standard spurious-mode rejection: an eigenvalue is PHYSICAL only if it
    # survives a resolution change. Tau/discretisation artefacts move; real
    # modes do not.
    Nhi = int(1.5 * P["Nmu"])
    ev_hi, *_ = solve_at(Nhi)
    ev_hi = ev_hi[np.isfinite(ev_hi)]
    keep = np.zeros(ev_lo.size, bool)
    for i, w in enumerate(ev_lo):
        if not np.isfinite(w):
            continue
        d = np.min(np.abs(ev_hi - w)) / max(abs(w), 1e-12)
        keep[i] = (d < a.rtol)
    print(f"  resolution check vs Nmu={Nhi} (rtol={a.rtol:g}): "
          f"{keep.sum()} of {good.sum()} finite modes are converged")
    good = keep

idx = np.where(good)[0]
ev = ev_lo[idx]
if ev.size == 0:
    print("\n  NO converged eigenvalues. Check the tau/BC setup before trusting anything.")
    sys.exit(1)

# ── Classification ───────────────────────────────────────────────────────────
# The discriminator is |Re(omega)| vs the INERTIAL frequency f = 2, NOT the sign.
# Poincare/inertia-gravity waves are SUPER-inertial (|omega| > f) and come in
# +/- pairs, so the retrograde branch is full of them. Rossby waves are
# SUB-inertial (|omega| < f) and are always retrograde.
F_INERTIAL = 2.0
sub = np.abs(np.real(ev)) < F_INERTIAL
rossby = idx[sub & (np.real(ev) < 0)]                       # sub-inertial, retrograde
sub_pro = idx[sub & (np.real(ev) >= 0)]                     # sub-inertial, prograde (odd)
poincare_r = idx[~sub & (np.real(ev) < 0)]                  # super-inertial, retrograde
poincare_p = idx[~sub & (np.real(ev) >= 0)]                 # super-inertial, prograde


def show(title, ii, key):
    if ii.size == 0:
        print(f"\n  {title}: none")
        return
    order = ii[np.argsort(key(ev_lo[ii]))]
    print(f"\n  {title}   (showing up to {a.nmodes})")
    print(f"    {'idx':>5s} {'Re(omega)':>12s} {'Im(omega)':>12s} {'T=2pi/|Re|':>11s} {'dir':>5s}")
    for j in order[:a.nmodes]:
        w = ev_lo[j]
        T = 2 * np.pi / abs(np.real(w)) if abs(np.real(w)) > 1e-12 else np.inf
        print(f"    {j:5d} {np.real(w):+12.6f} {np.imag(w):+12.6f} {T:11.4f} "
              f"{'pro' if np.real(w) > 0 else 'retro':>5s}")


# Least-damped first (largest Im = closest to neutral/growing)
show(f"ROSSBY  (sub-inertial |omega| < {F_INERTIAL:g}, retrograde)  <-- THE PHYSICS",
     rossby, lambda w: -np.imag(w))
show(f"POINCARE prograde  (|omega| > {F_INERTIAL:g})", poincare_p, lambda w: -np.imag(w))
show(f"POINCARE retrograde  (|omega| > {F_INERTIAL:g})", poincare_r, lambda w: -np.imag(w))
show(f"sub-inertial PROGRADE (unexpected - inspect if present)",
     sub_pro, lambda w: -np.imag(w))

# ── The stability verdict ────────────────────────────────────────────────────
# CRITICAL: the IVP integrates the operator AT THIS Nmu, including modes the
# resolution check rejects. A mode that exists at Nmu and vanishes at 1.5*Nmu is
# not "spurious and therefore harmless" — it is EXACTLY the numerical
# instability, and it will blow up the IVP. So the stability verdict is taken
# over ALL FINITE modes; the converged subset only tells you about the physics.
print("\n" + "-" * 72)
fin = np.where(np.isfinite(ev_lo))[0]
iall = fin[np.argmax(np.imag(ev_lo[fin]))]
wall = ev_lo[iall]
print(f"  STABILITY OF THE OPERATOR THE IVP ACTUALLY INTEGRATES (Nmu={P['Nmu']},")
print(f"  all {fin.size} finite modes, converged or not):")
print(f"    most unstable: idx={iall}  omega={wall:+.6f}")
if np.imag(wall) > 1e-8:
    print(f"    *** UNSTABLE: Im(omega)={np.imag(wall):+.4e} > 0")
    print(f"        e-folding time 1/Im = {1.0/np.imag(wall):.3f}")
    conv_set = set(idx.tolist())
    tag = "CONVERGED (physical)" if iall in conv_set else \
          "NOT converged -> a GRID-SCALE artefact of this discretisation"
    print(f"        this mode is {tag}")
    ngrow = int((np.imag(ev_lo[fin]) > 1e-8).sum())
    print(f"        {ngrow} of {fin.size} finite modes have Im(omega) > 0")
    print(f"        -> raise nu4/nu4_h (or Nmu) until this is negative.")
    print(f"        -> scan it here in seconds with --nu4 / --nu4-h. No IVP needed.")
else:
    print(f"    STABLE: every finite mode decays (max Im = {np.imag(wall):+.4e}).")
    print(f"    If the IVP still blows up, the growth is NOT a linear eigenvalue:")
    print(f"    suspect the timestepper, non-normal transient growth, or the taus.")

if not a.no_resolution_check:
    imax = idx[np.argmax(np.imag(ev_lo[idx]))]
    print(f"\n  (converged subset only — the PHYSICS: most unstable idx={imax}, "
          f"omega={ev_lo[imax]:+.6f})")
else:
    imax = iall

# Poincare sanity estimate
lest = m
print(f"\n  Poincare estimate sqrt(4 + G*l(l+1)) for l={lest}: "
      f"{np.sqrt(4 + P['G']*lest*(lest+1)):.4f}")
print("-" * 72)

# ── Eigenvector export ───────────────────────────────────────────────────────
if a.write_ic:
    if a.pick == "rossby":
        if rossby.size == 0:
            print("\n  --pick rossby: NO sub-inertial retrograde mode found. Nothing written.")
            sys.exit(1)
        sel = rossby[np.argmax(np.imag(ev_lo[rossby]))]   # least-damped Rossby
        why = "least-damped ROSSBY (sub-inertial, retrograde)"
    elif a.pick == "unstable":
        sel = imax; why = "largest Im(omega)"
    else:
        sel = int(a.pick); why = f"explicit index {sel}"

    w = ev_lo[sel]
    solver.set_state(sel, sp.subsystems[0])
    M.u_theta.change_scales(1); M.u_phi.change_scales(1); M.h.change_scales(1)
    uth = np.array(M.u_theta["g"]).ravel().copy()
    uph = np.array(M.u_phi["g"]).ravel().copy()
    hh = np.array(M.h["g"]).ravel().copy()

    # Normalise so max|Psi-equivalent amplitude| = IC_amp, using max|u| as proxy
    amp = float(P["IC_amp"])
    scale = max(np.max(np.abs(uth)), np.max(np.abs(uph)))
    if scale == 0:
        print("\n  selected eigenvector has zero velocity — refusing to write.")
        sys.exit(1)
    fac = amp / scale
    uth *= fac; uph *= fac; hh *= fac

    import h5py
    with h5py.File(a.write_ic, "w") as f:
        f.create_dataset("mu", data=np.asarray(M.mu_grid).ravel())
        f.create_dataset("u_theta_hat", data=uth)
        f.create_dataset("u_phi_hat", data=uph)
        f.create_dataset("h_hat", data=hh)
        f.attrs["omega_real"] = float(np.real(w))
        f.attrs["omega_imag"] = float(np.imag(w))
        f.attrs["m"] = m
        f.attrs["G"] = float(P["G"])
        f.attrs["s2"] = float(P["s2"]); f.attrs["s4"] = float(P["s4"])
        f.attrs["nu4"] = float(P["nu4"]); f.attrs["nu4_h"] = float(P.get("nu4_h", 0.0))
        f.attrs["Nmu"] = int(P["Nmu"])
        f.attrs["IC_amp"] = amp
    print(f"\n  [written] {a.write_ic}")
    print(f"    mode: idx={sel}  ({why})")
    print(f"    omega = {np.real(w):+.6f} {np.imag(w):+.6f}j")
    print(f"    T = {2*np.pi/abs(np.real(w)):.4f}   "
          f"-> for dt4 error < 0.1%% use snap_dt <~ {0.3/abs(np.real(w)):.4f}")
    print(f"    In input.json set:  \"ic_name\": \"evp\", \"evp_ic_file\": \"{a.write_ic}\"")

print("=" * 72)
if _tee is not None:
    sys.stdout = sys.__stdout__; _tee.close()
    print(f"[saved] {log_path}")
