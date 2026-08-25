# spherical-2d-hydro.py
# 2D Spherical Shallow-Water Rossby Wave Simulation — HYDRODYNAMIC (h formulation)
# Physical-domain IVP.  Induction equations omitted; b-fields removed.
# Prognostic height h' (reduced gravity G); no pressure/constraint.
#
# Coordinates: phi in [0, 2pi)  x  mu = cos(theta) in [-1, 1]
# Basis:       RealFourier(phi) x Chebyshev(mu)
#
# Variables (primitive, real): u_theta, u_phi, h   (all prognostic)
#
# Governing equations (primes dropped, normalised, t' = Omega_0 t):
#   (1)  dt u_θ + Ω_d ∂_φ u_θ − 2(1+Ω_d)μ u_φ = G sinθ ∂_μ h
#   (2)  dt u_φ + Ω_d ∂_φ u_φ + C_φ u_θ        = −(G/sinθ) ∂_φ h
#   (5)  dt h  + Ω_d ∂_φ h + (1/sinθ)[ μ u_θ − D u_θ + ∂_φ u_φ ] = 0
#
# with D = (1−μ²) ∂_μ ,  Ω_d = −s₂μ² − s₄μ⁴ ,
#      C_φ = 2(1+Ω_d)μ + (1−μ²)(2s₂μ + 4s₄μ³),  G = ε = gH₀/(Ω₀²R₀²).
#
# Tau placement:
#   eq (5) carries lift(tau_uth1,-1)+lift(tau_uth2,-2) → enforces u_theta(±1)=0.
#   h is prognostic and free at the poles (no BC); no pressure gauge needed.
#
# Usage:
#   python3 spherical-2d-hydro.py input.json [--restart]
#   mpiexec -n 4 python3 spherical-2d-hydro.py input.json

import json
import sys
import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
from scipy.special import lpmv
import sw_model

comm = MPI.COMM_WORLD

# ── Logging ───────────────────────────────────────────────────────────────────
if comm.rank == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s :: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logger_info.txt", mode="w"),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    for _log_name in ("solvers", "subsystems"):
        _lg = logging.getLogger(_log_name)
        _lg.setLevel(logging.INFO)
        _lg.propagate = False
        for _h in logging.getLogger().handlers:
            if _h not in _lg.handlers:
                _lg.addHandler(_h)
else:
    logging.basicConfig(level=logging.WARNING, force=True)
    logger = logging.getLogger(__name__)

# ── Arguments ─────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 spherical-2d-hydro.py input.json [--restart]")
    sys.exit(1)
restart = "--restart" in sys.argv
params  = json.load(open(sys.argv[1]))

# ── Parameters ────────────────────────────────────────────────────────────────
m_ic      = params["m"]          # dominant azimuthal wavenumber for IC
s2        = params["s2"]         # differential rotation: Omega_d = -s2*mu^2 - s4*mu^4
s4        = params["s4"]
G         = params["G"]          # reduced gravity  G = eps = g H0 / (Omega0^2 R0^2)
gamma     = params["gamma"]      # magnetic param (UNUSED in hydrodynamic case)
alpha_r   = params["alpha_r"]    # normalised alpha (UNUSED in hydrodynamic case)
nu4       = params["nu4"]        # mu-biharmonic hyperviscosity: +nu4 * d^4/dmu^4 on momentum
nu4_h     = params.get("nu4_h", 0.0)  # OPTIONAL mu-biharmonic on the h/continuity eqn (0=off).
                                      # Needed for large-G runs where the instability lives in h.
Nmu       = params["Nmu"]
Nphi      = params["Nphi"]
IC_amp    = params["IC_amp"]
mu_center = params["mu_center"]
mu_width  = params["mu_width"]
stop_sim_time = params["stop_sim_time"]
max_timestep  = params["max_timestep"]
snap_dt       = params["snap_dt"]
max_writes    = params["max_writes"]
run_name      = params["run_name"]
ic_name       = params["ic_name"]

# CFL controls (optional in JSON; safe defaults otherwise)
cfl_safety    = params.get("cfl_safety", 0.3)   # fraction of the CFL limit to use
cfl_cadence   = params.get("cfl_cadence", 10)   # recompute dt every N steps
min_timestep  = params.get("min_timestep", 1e-8)

snapshots_dir = "snapshots"
state_dir     = run_name + "_state"

timestepper = d3.SBDF2

# ── Model (bases, fields, equations, BCs) ─────────────────────────────────────
# Everything below the parameter block comes from sw_model.py, which is the
# SINGLE source of truth shared with evp_spherical_hd.py. Do not inline
# equations here — change them in sw_model.py and both solvers follow.
M = sw_model.build(params, mode="ivp")
problem = sw_model.make_problem(M)

dist      = M.dist
coords    = M.coords
phi_basis = M.phi_basis
mu_basis  = M.mu_basis
basis2d   = M.field_bases
phi_grid  = M.phi_grid
mu_grid   = M.mu_grid
u_theta, u_phi, h = M.u_theta, M.u_phi, M.h
st_f, ist_f, mu_f = M.st_f, M.ist_f, M.mu_f
dphi, dmu = M.dphi, M.dmu
use_h_hyper = M.use_h_hyper
eps_reg = sw_model.EPS_REG

# ── Solver ────────────────────────────────────────────────────────────────────
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# ── Initial conditions ────────────────────────────────────────────────────────
if ic_name == "none":
    # Balanced, non-divergent velocity seed from a streamfunction Psi.
    #   Psi(mu, phi) = profile(mu) * cos(m phi)
    # Velocity is u = k x grad(Psi)  (non-divergent by construction):
    #   u_theta = -(1/sin theta) d Psi/d phi
    #   u_phi   = - sin theta      d Psi/d mu
    # Height starts at h = 0.  Profile is normalised so max|Psi| = IC_amp.
    #
    # Two profiles are provided — comment out one, keep the other.

    # ---- (A) Associated Legendre P_l^m, sectoral l = m  (ACTIVE) ─────────────
    #   P_l^m vanishes at mu = ±1 for l >= m >= 1, so u vanishes at the poles.
    l_deg = int(params.get("l_deg", m_ic))     # sectoral mode: l = m
    prof = lpmv(m_ic, l_deg, mu_grid)          # (1, Nmu)
    prof_label = f"Legendre P_l^m (l={l_deg}, m={m_ic})"

    # ---- (B) Boundary-vanishing Gaussian  (comment (A), uncomment this) ──────
    #   raw bump minus the endpoint line, so profile(±1) = 0 exactly.
    #   (mu_width is the e-folding scale on (mu-mu_center)^2, = sqrt(2)*sigma.)
    # Graw = np.exp(-((mu_grid - mu_center)**2) / mu_width**2)
    # G_lo = np.exp(-((-1.0 - mu_center)**2) / mu_width**2)      # at mu = -1
    # G_hi = np.exp(-(( 1.0 - mu_center)**2) / mu_width**2)      # at mu = +1
    # baseline = G_lo + (G_hi - G_lo) * (mu_grid + 1.0) / 2.0    # endpoint line
    # prof = Graw - baseline                                      # zero at mu = ±1
    # prof_label = f"Gaussian (mu0={mu_center}, width={mu_width})"

    # normalise profile so that max|Psi| = IC_amp  (global max, MPI-safe)
    local_max = float(np.max(np.abs(prof))) if prof.size else 0.0
    prof_max  = comm.allreduce(local_max, op=MPI.MAX)
    if prof_max == 0.0:
        if comm.rank == 0:
            logger.error("IC profile is identically zero (check l>=m or mu_width).")
        sys.exit(1)
    prof = IC_amp * prof / prof_max

    # streamfunction field
    Psi = dist.Field(name="Psi", bases=basis2d)
    Psi["g"] = prof * np.cos(m_ic * phi_grid)

    # velocity from Psi — derivatives via Dedalus operators (MPI-safe);
    # metric factors are local, pointwise, and exact (sin theta > 0 on the grid).
    dPsi_dphi = dphi(Psi).evaluate(); dPsi_dphi.change_scales(1)
    dPsi_dmu  = dmu(Psi).evaluate();  dPsi_dmu.change_scales(1)
    st_grid = np.sqrt(np.maximum(1.0 - mu_grid**2, 0.0))

    u_theta["g"] = -(1.0 / st_grid) * dPsi_dphi["g"]
    u_phi["g"]   = -st_grid * dPsi_dmu["g"]
    h["g"]       = 0.0

    if comm.rank == 0:
        logger.info("IC: balanced streamfunction  %s  amp=%.4e", prof_label, IC_amp)
        logger.info("IC: u = k x grad(Psi)  (non-divergent);  h = 0")
elif ic_name == "evp":
    # Eigenvector IC from evp_spherical_hd.py — excites exactly ONE mode, so
    # there is no geostrophic-adjustment transient and no gravity-wave
    # contamination. The eigenvector q_hat(mu) is COMPLEX; the real physical
    # field is  Re[ q_hat(mu) * exp(i m phi) ]
    #         =  Re(q_hat)*cos(m phi) - Im(q_hat)*sin(m phi).
    import h5py
    evp_file = params["evp_ic_file"]
    with h5py.File(evp_file, "r") as f:
        mu_evp  = f["mu"][:]
        uth_hat = f["u_theta_hat"][:]
        uph_hat = f["u_phi_hat"][:]
        h_hat   = f["h_hat"][:]
        w_re    = float(f.attrs["omega_real"])
        w_im    = float(f.attrs["omega_imag"])
        m_evp   = int(f.attrs["m"])
        Nmu_evp = int(f.attrs["Nmu"])
        _attrs  = {k: f.attrs[k] for k in f.attrs}
    # The eigenvector is an eigenvector OF A SPECIFIC OPERATOR. If any parameter
    # differs, it is not an eigenmode of the system being integrated: it projects
    # onto several modes, radiates a transient, and quietly contaminates the data.
    # "Nearly an eigenvector" is the dangerous case — it looks fine and is not.
    _mismatch = []
    for _k, _mine in (("m", m_ic), ("Nmu", Nmu), ("G", G),
                      ("s2", s2), ("s4", s4), ("nu4", nu4), ("nu4_h", nu4_h)):
        if _k not in _attrs:
            continue
        _theirs = _attrs[_k]
        if not np.isclose(float(_theirs), float(_mine), rtol=1e-12, atol=0.0):
            _mismatch.append(f"    {_k:8s} EVP IC = {float(_theirs):<12g} "
                             f"input.json = {float(_mine):<12g}")
    if _mismatch:
        raise ValueError(
            "EVP IC was built for a DIFFERENT operator than this run:\n"
            + "\n".join(_mismatch)
            + f"\n  Regenerate it with matching parameters, e.g.:\n"
              f"    python3 evp_spherical_hd.py --input {sys.argv[1]} "
              f"--write-ic {evp_file} --pick rossby\n"
              "  (an eigenvector of the wrong operator is NOT an eigenmode here)"
        )
    # mu_grid here is this rank's local slice of the SAME analytic Chebyshev
    # grid the EVP used, so index by value rather than assuming layout.
    order = np.argsort(mu_evp)
    def _on_local(vec):
        return np.interp(mu_grid.ravel(), mu_evp[order], vec[order])
    cosmp = np.cos(m_ic * phi_grid)
    sinmp = np.sin(m_ic * phi_grid)
    for fld, hat in ((u_theta, uth_hat), (u_phi, uph_hat), (h, h_hat)):
        re = _on_local(np.real(hat))[None, :]
        im = _on_local(np.imag(hat))[None, :]
        fld["g"] = re * cosmp - im * sinmp
    if comm.rank == 0:
        logger.info("IC: EVP eigenvector from %s", evp_file)
        logger.info("IC: m=%d  omega=%+.6f %+.6fj  T=%.4f  (expect sigma=%+.6f)",
                    m_evp, w_re, w_im, 2*np.pi/abs(w_re) if w_re else np.inf, w_im)
else:
    ic_file = ic_name + "_state/" + ic_name + "_state_s1.h5"
    write, initial_time = solver.load_state(ic_file)
    if restart:
        solver.sim_time      = initial_time
        solver.stop_sim_time = initial_time + stop_sim_time
    if comm.rank == 0:
        logger.info("Loaded state from %s  t=%.4f", ic_file, solver.sim_time)

# ── Analysis ──────────────────────────────────────────────────────────────────
snapshots = solver.evaluator.add_file_handler(
    snapshots_dir, sim_dt=snap_dt, max_writes=max_writes, mode="overwrite"
)
for fld, name in ((u_theta, "u_theta"), (u_phi, "u_phi"), (h, "h")):
    snapshots.add_task(fld, name=name)

# Relative vorticity: zeta = ist * dphi(u_theta) + dmu(st * u_phi)
snapshots.add_task(ist_f*dphi(u_theta) + dmu(st_f*u_phi), name="vorticity")
# Absolute vorticity: planetary (2mu) + relative
snapshots.add_task(2*mu_f + ist_f*dphi(u_theta) + dmu(st_f*u_phi), name="abs_vorticity")
# Kinetic energy density
snapshots.add_task(u_theta**2 + u_phi**2, name="KE_density")

# Final-state checkpoint (for restart)
final_state = solver.evaluator.add_file_handler(
    state_dir, sim_dt=stop_sim_time - max_timestep, mode="overwrite"
)
final_state.add_tasks(solver.state)

# ── Flow diagnostics ──────────────────────────────────────────────────────────
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u_theta**2 + u_phi**2, name="KE")
flow.add_property(h**2, name="h2")

# ── CFL time-step control ─────────────────────────────────────────────────────
# The fastest signal is the shallow-water GRAVITY wave, c_g = sqrt(G) (normalised);
# the flow is a small linear perturbation, so advection is sub-dominant but is
# included for generality. The velocity components (u_theta, u_phi) are PHYSICAL
# spherical components, not conjugate to the (phi, mu) grid, so the metric enters
# explicitly:
#   phi-cell crossing rate: ist * (c_g + |u_phi|)   / dphi     (ist = 1/(sinθ+eps))
#   mu -cell crossing rate:  st * (c_g + |u_theta|) / dmu       (st  = sinθ)
# dt_CFL = cfl_safety / max_over_grid(sum of the two rates), capped by max_timestep.
cg   = float(np.sqrt(max(G, 1e-30)))                     # gravity-wave speed sqrt(G)
dphi = 2.0 * np.pi / Nphi
# analytic Gauss-Chebyshev (roots) mu grid -> identical on every rank (MPI-safe)
_mu_cheb = np.cos(np.pi * (2*np.arange(Nmu) + 1) / (2*Nmu))
_order   = np.argsort(_mu_cheb)
_mu_srt  = _mu_cheb[_order]
_dmu     = np.gradient(_mu_srt)                          # per-point mu spacing
_st_cheb = np.sqrt(np.maximum(1.0 - _mu_srt**2, 0.0))    # sin(theta)
_ist_cheb = 1.0 / (_st_cheb + eps_reg)                   # regularised 1/sin(theta)

def compute_timestep():
    """Adaptive CFL time step (gravity wave + advection), MPI-safe, capped."""
    # global max physical flow speeds (0 contribution if a rank holds no grid)
    umax_phi = comm.allreduce(
        float(np.max(np.abs(u_phi["g"]))) if u_phi["g"].size else 0.0, op=MPI.MAX)
    umax_th  = comm.allreduce(
        float(np.max(np.abs(u_theta["g"]))) if u_theta["g"].size else 0.0, op=MPI.MAX)
    inv_dt = (_ist_cheb * (cg + umax_phi) / dphi
              + _st_cheb * (cg + umax_th) / np.abs(_dmu))
    inv_dt_max = float(np.max(inv_dt))
    dt = cfl_safety / inv_dt_max if inv_dt_max > 0 else max_timestep
    return float(np.clip(dt, min_timestep, max_timestep))

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    # Initial CFL dt — compute_timestep() is an MPI collective (allreduce),
    # so ALL ranks must call it here, not just rank 0.
    timestep = compute_timestep()
    if comm.rank == 0:
        logger.info("Starting main loop (HYDRODYNAMIC, shallow-water h)")
        logger.info(
            "Params: m=%d  s2=%.4f  s4=%.4f  G=%.4g  nu4=%.2e  nu4_h=%.2e%s  "
            "(gamma, alpha_r ignored: %.1e, %.1e)",
            m_ic, s2, s4, G, nu4, nu4_h,
            "" if use_h_hyper else " (h-filter OFF)", gamma, alpha_r,
        )
        logger.info(
            "CFL: c_g=sqrt(G)=%.4g  safety=%.2f  cadence=%d  ->  initial dt=%.3e  (cap=%.3e)",
            cg, cfl_safety, cfl_cadence, timestep, max_timestep,
        )
    while solver.proceed:
        # Adaptive gravity-wave CFL; recomputed every cfl_cadence steps
        # (called by all ranks together — the collective stays synchronised).
        if solver.iteration % cfl_cadence == 0:
            timestep = compute_timestep()
        solver.step(timestep)
        if solver.iteration % 1000 == 0:
            ke  = flow.volume_integral("KE")   # MPI collective — all ranks call
            h2  = flow.volume_integral("h2")
            if comm.rank == 0:
                logger.info(
                    "Iter=%i  t=%.4f  dt=%.2e  KE=%.4e  <h²>=%.4e",
                    solver.iteration, solver.sim_time, timestep, ke, h2,
                )
except Exception:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()

if comm.rank == 0:
    logger.info("Simulation completed successfully.")