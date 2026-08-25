# sw_model.py
# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH for the 2D spherical shallow-water hydrodynamic model
# (h formulation).  Both solvers import from here:
#
#     spherical-2d-hd-s2s4.py   (IVP — time integration)
#     evp_spherical_hd.py       (EVP — eigenvalues + eigenvector ICs)
#
# The equation strings appear EXACTLY ONCE, in _add_equations() below.  Change
# the physics here and both solvers follow.  Nothing is duplicated.
#
# The two modes differ in only three substitutions:
#
#   mode="ivp"                         mode="evp"
#   --------------------------------   ------------------------------------
#   RealFourier(phi) x Chebyshev(mu)   Chebyshev(mu)         (1D, single m)
#   dtype = float64                    dtype = complex128
#   dt(A)   = d3.TimeDerivative(A)     dt(A)   = -1j*omega*A
#   dphi(A) = d3.Differentiate(A,phi)  dphi(A) = 1j*m*A
#
# The equations are otherwise IDENTICAL — the system is already linear, so the
# EVP needs no linearisation step.
#
# SIGN CONVENTION (matches measure_mode.py):
#   fields ~ exp(i*m*phi) * exp(-i*omega*t)
#   => Re(omega) > 0 is PROGRADE (eastward);  Re(omega) < 0 is RETROGRADE.
#   => Im(omega) > 0 is GROWING;              Im(omega) < 0 is DECAYING.
#   A Rossby wave MUST have Re(omega) < 0.  Poincare/inertia-gravity waves come
#   in +/- pairs at |omega| ~ sqrt(4 + G*l(l+1)).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import dedalus.public as d3

EPS_REG = 0.05      # pole regularisation for 1/sin(theta) — MUST match lsq_from_data.py
DEALIAS = 3 / 2


class Model:
    """Container for everything the equation strings reference."""
    pass


def build(params, mode, m=None):
    """
    Build bases, fields, coefficient fields and operators.

    params : dict from input.json
    mode   : "ivp" or "evp"
    m      : azimuthal wavenumber (EVP only; defaults to params["m"])
    """
    if mode not in ("ivp", "evp"):
        raise ValueError(f"mode must be 'ivp' or 'evp', got {mode!r}")

    M = Model()
    M.mode = mode
    M.params = params

    M.s2 = float(params["s2"])
    M.s4 = float(params["s4"])
    M.G = float(params["G"])
    M.nu4 = float(params["nu4"])
    M.nu4_h = float(params.get("nu4_h", 0.0))
    M.Nmu = int(params["Nmu"])
    M.Nphi = int(params["Nphi"])
    M.m = int(params["m"]) if m is None else int(m)
    M.use_h_hyper = (M.nu4_h > 0.0)

    # ── Bases ────────────────────────────────────────────────────────────────
    if mode == "ivp":
        coords = d3.CartesianCoordinates("phi", "mu")
        dist = d3.Distributor(coords, dtype=np.float64)
        phi_basis = d3.RealFourier(coords["phi"], size=M.Nphi,
                                   bounds=(0, 2 * np.pi), dealias=DEALIAS)
        mu_basis = d3.Chebyshev(coords["mu"], size=M.Nmu,
                                bounds=(-1, 1), dealias=DEALIAS)
        field_bases = (phi_basis, mu_basis)
        tau_bases = phi_basis          # taus vary with phi
        phi_grid, mu_grid = dist.local_grids(phi_basis, mu_basis)
        M.coords = coords
        M.phi_basis = phi_basis
        M.phi_grid = phi_grid
        dphi = lambda f: d3.Differentiate(f, coords["phi"])
        dmu = lambda f: d3.Differentiate(f, coords["mu"])
        dt = d3.TimeDerivative
        omega = None
    else:
        coord = d3.Coordinate("mu")
        dist = d3.Distributor(coord, dtype=np.complex128)
        mu_basis = d3.Chebyshev(coord, size=M.Nmu,
                                bounds=(-1, 1), dealias=DEALIAS)
        field_bases = mu_basis
        tau_bases = None               # taus are scalars in 1D
        mu_grid = dist.local_grid(mu_basis)
        M.coords = coord
        M.phi_basis = None
        M.phi_grid = None
        omega = dist.Field(name="omega")
        _mm = M.m
        dphi = lambda f: 1j * _mm * f
        dmu = lambda f: d3.Differentiate(f, coord)
        dt = lambda f: -1j * omega * f

    M.dist = dist
    M.mu_basis = mu_basis
    M.mu_grid = mu_grid
    M.field_bases = field_bases
    M.omega = omega

    def _F(name, bases=field_bases):
        return dist.Field(name=name, bases=bases) if bases is not None \
            else dist.Field(name=name)

    # ── Prognostic + auxiliary fields ────────────────────────────────────────
    u_theta = _F("u_theta")
    u_phi = _F("u_phi")
    h = _F("h")
    s_theta = _F("s_theta")
    s_phi = _F("s_phi")

    taus = {n: _F(n, tau_bases) for n in
            ("tau_ut1", "tau_ut2", "tau_ut3", "tau_ut4",
             "tau_up1", "tau_up2", "tau_up3", "tau_up4")}

    if M.use_h_hyper:
        s_h = _F("s_h")
        taus.update({n: _F(n, tau_bases) for n in
                     ("tau_h1", "tau_h2", "tau_h3", "tau_h4")})
    else:
        s_h = None

    # ── Metric / parameter fields (mu-dependent only) ────────────────────────
    mu_f = _F("mu_f", mu_basis);           mu_f["g"] = mu_grid
    one_minus_mu2 = _F("one_minus_mu2", mu_basis)
    one_minus_mu2["g"] = 1.0 - mu_grid**2
    st_f = _F("st_f", mu_basis)
    st_f["g"] = np.sqrt(np.maximum(1.0 - mu_grid**2, 0.0))
    ist_f = _F("ist_f", mu_basis)
    ist_f["g"] = 1.0 / (np.sqrt(np.maximum(1.0 - mu_grid**2, 0.0)) + EPS_REG)

    Omega_d_f = _F("Omega_d_f", mu_basis)
    Omega_d_f["g"] = -M.s2 * mu_grid**2 - M.s4 * mu_grid**4

    corf_th_f = _F("corf_th_f", mu_basis)
    corf_th_f["g"] = 2.0 * mu_grid * (1.0 - M.s2 * mu_grid**2 - M.s4 * mu_grid**4)

    corf_phi_f = _F("corf_phi_f", mu_basis)
    corf_phi_f["g"] = (
        2.0 * mu_grid * (1.0 - M.s2 * mu_grid**2 - M.s4 * mu_grid**4)
        + (1.0 - mu_grid**2) * (2.0 * M.s2 * mu_grid + 4.0 * M.s4 * mu_grid**3)
    )

    # ── Operators ────────────────────────────────────────────────────────────
    D_op = lambda f: one_minus_mu2 * dmu(f)
    lift_basis = mu_basis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    # ── Variable list (order matters for the solver) ─────────────────────────
    M.vars = [u_theta, u_phi, h, s_theta, s_phi,
              taus["tau_ut1"], taus["tau_ut2"], taus["tau_ut3"], taus["tau_ut4"],
              taus["tau_up1"], taus["tau_up2"], taus["tau_up3"], taus["tau_up4"]]
    if M.use_h_hyper:
        M.vars += [s_h, taus["tau_h1"], taus["tau_h2"],
                   taus["tau_h3"], taus["tau_h4"]]

    M.u_theta, M.u_phi, M.h = u_theta, u_phi, h
    M.dphi, M.dmu, M.dt = dphi, dmu, dt
    M.st_f, M.ist_f, M.mu_f = st_f, ist_f, mu_f

    # Namespace the equation strings are parsed against.
    M.ns = dict(
        u_theta=u_theta, u_phi=u_phi, h=h, s_theta=s_theta, s_phi=s_phi,
        mu_f=mu_f, one_minus_mu2=one_minus_mu2, st_f=st_f, ist_f=ist_f,
        Omega_d_f=Omega_d_f, corf_th_f=corf_th_f, corf_phi_f=corf_phi_f,
        G=M.G, nu4=M.nu4, nu4_h=M.nu4_h,
        dphi=dphi, dmu=dmu, dt=dt, D_op=D_op, lift=lift,
        **taus,
    )
    if M.use_h_hyper:
        M.ns["s_h"] = s_h
    if omega is not None:
        M.ns["omega"] = omega
    return M


def make_problem(M):
    """Create the IVP/EVP and attach the (shared) equations and BCs."""
    if M.mode == "ivp":
        problem = d3.IVP(M.vars, namespace=M.ns)
    else:
        problem = d3.EVP(M.vars, eigenvalue=M.omega, namespace=M.ns)
    _add_equations(problem, M)
    return problem


def _add_equations(problem, M):
    """
    THE equations. Identical for IVP and EVP — only dt/dphi differ, and those
    are supplied through the namespace by build().
    """
    # ── Eq (1): u_theta momentum (+ mu-biharmonic nu4 * d^4 u_theta/dmu^4) ───
    problem.add_equation(
        "dt(u_theta) + Omega_d_f*dphi(u_theta) - corf_th_f*u_phi "
        "- G*st_f*dmu(h) + nu4*dmu(dmu(s_theta)) "
        "+ lift(tau_ut1,-1) + lift(tau_ut2,-2) = 0"
    )
    # ── Eq (2): u_phi momentum (+ mu-biharmonic nu4 * d^4 u_phi/dmu^4) ───────
    problem.add_equation(
        "dt(u_phi) + Omega_d_f*dphi(u_phi) + corf_phi_f*u_theta "
        "+ G*ist_f*dphi(h) + nu4*dmu(dmu(s_phi)) "
        "+ lift(tau_up1,-1) + lift(tau_up2,-2) = 0"
    )
    # ── Auxiliary definitions: s = d^2/dmu^2 of each velocity ────────────────
    problem.add_equation(
        "s_theta - dmu(dmu(u_theta)) + lift(tau_ut3,-1) + lift(tau_ut4,-2) = 0"
    )
    problem.add_equation(
        "s_phi   - dmu(dmu(u_phi))   + lift(tau_up3,-1) + lift(tau_up4,-2) = 0"
    )
    # ── Eq (5): continuity (h prognostic), optional mu-biharmonic on h ───────
    if M.use_h_hyper:
        problem.add_equation(
            "dt(h) + Omega_d_f*dphi(h) "
            "+ ist_f*(mu_f*u_theta - D_op(u_theta) + dphi(u_phi)) "
            "+ nu4_h*dmu(dmu(s_h)) + lift(tau_h1,-1) + lift(tau_h2,-2) = 0"
        )
        problem.add_equation(
            "s_h - dmu(dmu(h)) + lift(tau_h3,-1) + lift(tau_h4,-2) = 0"
        )
    else:
        problem.add_equation(
            "dt(h) + Omega_d_f*dphi(h) "
            "+ ist_f*(mu_f*u_theta - D_op(u_theta) + dphi(u_phi)) = 0"
        )
    # ── BCs: clamped velocity, u = 0 and du/dmu = 0 at mu = +/-1 ─────────────
    problem.add_equation("u_theta(mu=-1) = 0")
    problem.add_equation("u_theta(mu=1)  = 0")
    problem.add_equation("dmu(u_theta)(mu=-1) = 0")
    problem.add_equation("dmu(u_theta)(mu=1)  = 0")
    problem.add_equation("u_phi(mu=-1) = 0")
    problem.add_equation("u_phi(mu=1)  = 0")
    problem.add_equation("dmu(u_phi)(mu=-1) = 0")
    problem.add_equation("dmu(u_phi)(mu=1)  = 0")
    # ── h-filter BCs (Neumann/free): only when the h-biharmonic is active ────
    if M.use_h_hyper:
        problem.add_equation("dmu(h)(mu=-1) = 0")
        problem.add_equation("dmu(h)(mu=1)  = 0")
        problem.add_equation("dmu(s_h)(mu=-1) = 0")
        problem.add_equation("dmu(s_h)(mu=1)  = 0")
