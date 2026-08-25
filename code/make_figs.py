#!/usr/bin/env python3
"""
make_figures.py -- figures for the closed-form Rossby-wave recovery paper.

TWO KINDS OF FIGURE IN HERE, and the difference matters:

  ANALYTIC (Fig. 1, 2): computed from scratch every run, from the projection
  geometry alone. Nothing is hard-coded, so these are fully reproducible and
  you can trust them as they stand.

  MEASURED (Fig. 3, 4, 5): the numbers are transcribed into the MEASURED
  block below, taken from the run logs. They are correct as transcribed, but
  a transcription is not a pipeline. BEFORE SUBMISSION, replace that block
  with a parser that reads the actual .log files, so the figures regenerate
  from data and cannot silently drift from the tables in the text.

A&A figure widths: 88 mm single column, 180 mm double column.

    python3 make_figures.py            # writes fig1..fig5 as PDF into ./figures
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1/25.4
COL1, COL2 = 88*MM, 180*MM
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# =====================================================================
# MEASURED RESULTS -- transcribed from run logs. REPLACE WITH A PARSER.
# =====================================================================
# End-to-end LOS recovery, (s2,s4)=(0.19,0.23), B0=7.25 deg, mask 60 deg,
# K=20, r=2. Errors in per cent. None where not run.
LOS_NOISE = [0.001, 0.01, 0.02, 0.05, 0.10]
LOS = {
    "eq1+2+5": {"s2": [2.30, 5.74, 14.53, 72.30, 273.0],
                "s4": [3.29, 13.43, 38.45, 200.40, 760.0]},
    "eq1+2":   {"s2": [0.13, 0.06, 0.07, 0.66, 3.05],
                "s4": [0.22, 0.03, 0.28, 1.44, 5.59]},
    "eq2":     {"s2": [None, 0.29, None, 0.29, 0.13],
                "s4": [None, 0.43, None, 0.41, 0.29]},
}
# 5-seed scatter at noise 0.05 (std dev, per cent)
LOS_ERRBAR = {"eq1+2": {"s2": (0.05, 0.28), "s4": (0.05, 1.12)},
              "eq2":   {"s2": (0.05, 0.12), "s4": (0.05, 0.33)}}

# Polar masking, clean data, (0.19,0.23). Cutoff latitude in degrees;
# 90 means no mask.
POLE_CUT = [90, 60, 40, 20]
POLE_S2 = [0.10, 0.07, 0.12, 0.16]
POLE_S4 = [0.12, 0.01, 0.17, 0.21]
POLE_COND = [7.2, 7.2, 9.0, 19.0]

# Cadence, with noise 0.1 + limb weighting + 60 deg mask.
CAD_PTS = [27.4, 13.7, 6.85]          # samples per wave period
CAD_S2 = [0.32, 0.00, 5.79]
CAD_S4 = [0.49, 0.44, 5.99]

# Irregular sampling, same configuration as the cadence baseline.
IRREG_LABEL = ["baseline", "20% dropout", "40% dropout", "gap\n(~3 periods)"]
IRREG_S2 = [0.32, 0.00, 0.00, 0.31]
IRREG_S4 = [0.49, 0.25, 0.11, 0.46]

# PINN training trajectory at tstride 4 (6.9 samples/period). Iteration in
# thousands; recovered s4 (true 0.23).
PINN_IT = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
           85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
PINN_S4_NF7 = [0.15990, 0.06506, 0.03150, 0.13903, 0.21538, 0.28184, 0.27278,
               0.29021, 0.36713, 0.33458, 0.30614, 0.52196, 0.83380, 1.02915,
               1.02145, 0.95019, 0.80533, 0.52428, 0.07026, -0.51253, -1.17458,
               -1.97630, -2.83168, -3.78752, -4.69100, -5.48818, -6.07457,
               -6.62381, -6.87050, -6.95395]
PINN_S4_NF9 = [0.66141, 0.70803, 0.68725, 0.48956, 0.20968, -0.16473, -0.65620,
               -0.93403, -1.02266, -0.84302, -0.43772, -0.24253, -0.43793,
               -0.45851, -0.53332, -0.59699, -0.72486, -0.82326, -0.99864,
               -1.24145, -1.48153, -1.77258, -2.03194, -2.29888, -2.46267,
               -2.62661, -2.69363, -2.71950, -2.71266, -2.68636]
S4_TRUE = 0.23
LSQ_S4_AT_TSTRIDE4 = 5.99      # per cent, the bar the network must beat
# =====================================================================


def proj(mu, phi, b0):
    """(r_hat.e, theta_hat.e, phi_hat.e) with observer e = (cos b0, 0, sin b0)."""
    st = np.sqrt(np.maximum(1 - mu**2, 0))
    cb, sb = np.cos(b0), np.sin(b0)
    return (st*np.cos(phi)*cb + mu*sb,
            mu*np.cos(phi)*cb - st*sb,
            -np.sin(phi)*cb)


def design(mu, phi, m, b0):
    pw, pt, pp = proj(mu, phi, b0)
    c, s = np.cos(m*phi), np.sin(m*phi)
    return np.column_stack([2*pw*c, -2*pw*s, 2*pt*c, -2*pt*s, 2*pp*c, -2*pp*s])


# ---------------------------------------------------------------------
# Fig. 1  ANALYTIC -- the B0 = 0 degeneracy and what it does to conditioning
# ---------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 0.36*COL2))

    # (a) ratio of the two projections, NORMALISED to its central-meridian
    #     value. At B0 = 0 this is identically 1 at every longitude and every
    #     latitude -- the two columns are proportional, so the system is
    #     singular. Any nonzero B0 breaks it.
    phi = np.linspace(-np.pi/2 + 2e-2, np.pi/2 - 2e-2, 400)
    for b0deg, ls, c in ((0.0, "-", "C3"), (1.0, "-.", "C1"), (7.25, "--", "C0")):
        b0 = np.deg2rad(b0deg)
        for j, lat0 in enumerate((15, 45, 70)):
            mu0 = np.sin(np.deg2rad(lat0))
            pw, pt, _ = proj(mu0, phi, b0)
            r = (pt/pw) / (pt/pw)[len(phi)//2]
            ax[0].plot(np.rad2deg(phi), r, ls, color=c,
                       lw=2.2 if b0deg == 0 else 1.1,
                       alpha=1.0 if b0deg == 0 else 0.85,
                       zorder=5 if b0deg == 0 else 2,
                       label=rf"$B_0={b0deg:g}^\circ$" if j == 0 else None)
    ax[0].set_xlabel("longitude from central meridian (deg)")
    ax[0].set_ylabel(r"$(\hat\theta\cdot\hat e)/(\hat r\cdot\hat e)$,"
                     "\n normalised to central meridian")
    ax[0].set_ylim(0.05, 1.18)
    ax[0].set_title("(a) three latitudes: 15, 45, 70 deg", loc="left")
    ax[0].legend(frameon=False, loc="lower center", ncol=3, fontsize=6)
    ax[0].text(0.5, 0.93,
               r"$B_0=0$: identically flat $\Rightarrow$ columns proportional",
               transform=ax[0].transAxes, ha="center", fontsize=6,
               style="italic", color="C3")

    # (b) condition number vs B0
    m, Nphi = 5, 256
    phi_all = 2*np.pi*np.arange(Nphi)/Nphi
    b0s = np.logspace(-3, np.log10(7.25), 60)
    for lat, c in zip((15, 30, 45, 60), ["C0", "C1", "C2", "C3"]):
        mu = np.sin(np.deg2rad(lat))
        conds = []
        for b0deg in b0s:
            b0 = np.deg2rad(b0deg)
            pw, _, _ = proj(mu, phi_all, b0)
            conds.append(np.linalg.cond(design(mu, phi_all[pw > 0], m, b0)))
        ax[1].loglog(b0s, conds, color=c, lw=1.4, alpha=0.8,
                     label=rf"${lat}^\circ$")
    ax[1].loglog(b0s, 100/b0s, "k:", lw=0.9, label=r"$100/B_0$")
    ax[1].axvspan(7.25, 10, color="0.9", zorder=0)
    ax[1].text(8.6, 8e1, "beyond\nsolar $|B_0|$", fontsize=6, color="0.45",
               ha="center", va="center")
    ax[1].set_xlim(1e-3, 10)
    ax[1].set_xlabel(r"$B_0$ (deg)")
    ax[1].set_ylabel("condition number")
    ax[1].set_title("(b) conditioning of the inversion", loc="left")
    ax[1].legend(frameon=False, title="latitude (curves coincide)", fontsize=6,
                 title_fontsize=6, loc="lower left")

    fig.subplots_adjust(wspace=0.32)
    fig.savefig(f"{OUT}/fig1_geometry_conditioning.pdf")
    plt.close(fig)
    print("fig1_geometry_conditioning.pdf   [analytic]")


# ---------------------------------------------------------------------
# Fig. 3  MEASURED -- end-to-end error vs LOS noise, by equation subset
# ---------------------------------------------------------------------
def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 0.38*COL2), sharey=True)
    style = {"eq1+2+5": ("C3", "o", "-"), "eq1+2": ("C0", "s", "-"),
             "eq2": ("C2", "D", "-")}
    for k, par in enumerate(("s2", "s4")):
        for name, (c, mk, ls) in style.items():
            x = [n for n, v in zip(LOS_NOISE, LOS[name][par]) if v is not None]
            y = [v for v in LOS[name][par] if v is not None]
            ax[k].loglog(x, y, ls, color=c, marker=mk, ms=3.5,
                         label=name if k == 0 else None)
            eb = LOS_ERRBAR.get(name, {}).get(par)
            if eb:
                xi, si = eb
                yi = LOS[name][par][LOS_NOISE.index(xi)]
                ax[k].errorbar([xi], [yi], yerr=[si], color=c, capsize=2, lw=1)
        ax[k].axhline(1.0, color="0.7", lw=0.6, ls=":")
        ax[k].set_xlabel("line-of-sight noise (fraction of RMS)")
        ax[k].set_title(rf"({'ab'[k]}) $s_{{{2 if par=='s2' else 4}}}$",
                        loc="left")
    ax[0].set_ylabel("relative error (per cent)")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].text(0.03, 0.60, "1 per cent", transform=ax[0].transAxes,
               fontsize=6, color="0.5")
    fig.subplots_adjust(wspace=0.10)
    fig.savefig(f"{OUT}/fig3_noise_eqsubset.pdf")
    plt.close(fig)
    print("fig3_noise_eqsubset.pdf   [MEASURED -- replace with log parser]")


# ---------------------------------------------------------------------
# Fig. 4  MEASURED -- the three degradations that are not noise
# ---------------------------------------------------------------------
def fig4():
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 0.34*COL2))

    # (a) polar masking
    ax[0].plot(POLE_CUT, POLE_S2, "o-", color="C0", ms=3.5, label=r"$s_2$")
    ax[0].plot(POLE_CUT, POLE_S4, "s-", color="C3", ms=3.5, label=r"$s_4$")
    ax[0].invert_xaxis()
    ax[0].set_ylim(0, 0.25)
    ax[0].set_xlabel("mask cutoff latitude (deg)")
    ax[0].set_ylabel("relative error (per cent)")
    ax[0].set_title("(a) polar masking", loc="left")
    ax[0].legend(frameon=False, loc="upper left")
    a0 = ax[0].twinx()
    a0.plot(POLE_CUT, POLE_COND, ":", color="0.55", lw=1.0)
    a0.set_ylabel("condition number", color="0.55", fontsize=6.5)
    a0.tick_params(colors="0.55", labelsize=6)
    a0.set_ylim(0, 22)

    # (b) cadence. LINEAR axis on purpose: one measured value rounds to
    # 0.00 per cent, which a log axis cannot show at all -- it would be
    # silently dropped.
    ax[1].plot(CAD_PTS, CAD_S2, "o-", color="C0", ms=3.5, label=r"$s_2$")
    ax[1].plot(CAD_PTS, CAD_S4, "s-", color="C3", ms=3.5, label=r"$s_4$")
    ax[1].axvline(10, color="0.6", lw=0.8, ls="--")
    ax[1].text(10.4, 4.6, "suggested\nfloor", fontsize=6, color="0.45",
               ha="left", va="center")
    ax[1].invert_xaxis()
    ax[1].set_ylim(-0.4, 6.6)
    ax[1].set_xlabel("samples per wave period")
    ax[1].set_title("(b) cadence", loc="left")
    ax[1].legend(frameon=False, loc="upper left")

    # (c) irregular sampling
    xs = np.arange(len(IRREG_LABEL))
    ax[2].bar(xs-0.18, IRREG_S2, 0.36, color="C0", label=r"$s_2$")
    ax[2].bar(xs+0.18, IRREG_S4, 0.36, color="C3", label=r"$s_4$")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(["base", "20%\ndrop", "40%\ndrop", "gap"],
                          fontsize=6.5)
    ax[2].set_ylim(0, 0.72)
    ax[2].set_ylabel("relative error (per cent)", fontsize=7)
    ax[2].set_title("(c) irregular sampling", loc="left")
    ax[2].legend(frameon=False, loc="upper right")

    fig.subplots_adjust(wspace=0.62)
    fig.savefig(f"{OUT}/fig4_degradations.pdf")
    plt.close(fig)
    print("fig4_degradations.pdf   [MEASURED -- replace with log parser]")


# ---------------------------------------------------------------------
# Fig. 5  MEASURED -- the network over-trains and destroys its own estimate
#
# IMPORTANT, and easy to get wrong: this trajectory is the PINN script's
# INTERNAL recovery (cheb=20, NO svd truncation, NO polar mask). The matched
# comparator is therefore the raw degraded field through that SAME internal
# recovery, which the run log reports as its feasibility check: 116.7 per cent.
# It is NOT the mitigated closed-form result (5.99 per cent), which uses
# svd=2 and a 60 deg mask and is simply a different estimator.
#
# The apples-to-apples comparison -- both fields through identical mitigated
# recovery -- is the single pair raw 5.99 vs network 28.19 per cent, quoted
# in the text and marked here for reference only.
# ---------------------------------------------------------------------
RAW_INTERNAL = 116.7    # raw degraded field, PINN-internal recovery
RAW_MITIGATED = 5.99    # raw degraded field, mitigated recovery (different estimator)
PINN_MITIGATED = 28.19  # network final checkpoint, mitigated recovery


def fig5():
    err7 = [abs(v-S4_TRUE)/S4_TRUE*100 for v in PINN_S4_NF7]
    err9 = [abs(v-S4_TRUE)/S4_TRUE*100 for v in PINN_S4_NF9]

    fig, ax = plt.subplots(figsize=(COL1, 0.80*COL1))
    ax.semilogy(PINN_IT, err7, "-", color="C0", label="network, 7 features")
    ax.semilogy(PINN_IT, err9, "-", color="C1", alpha=0.75,
                label="network, 9 features")
    ax.axhline(RAW_INTERNAL, color="C3", ls="--", lw=1.2,
               label="raw field, same recovery")
    ax.axhline(PINN_MITIGATED, color="0.55", ls=":", lw=1.0,
               label="network, final, mitigated")
    ax.axhline(RAW_MITIGATED, color="C2", ls="-.", lw=1.0,
               label="raw field, mitigated")
    i7 = int(np.argmin(err7))
    ax.plot(PINN_IT[i7], err7[i7], "o", color="C0", ms=4)
    ax.annotate("best point,\nthen diverges",
                xy=(PINN_IT[i7], err7[i7]), xytext=(46, 12),
                fontsize=6, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_ylim(3, 1e4)
    ax.set_xlabel("training iteration (thousands)")
    ax.set_ylabel(r"$s_4$ relative error (per cent)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.6, ncol=1)
    fig.savefig(f"{OUT}/fig5_pinn_trajectory.pdf")
    plt.close(fig)
    print("fig5_pinn_trajectory.pdf   [MEASURED -- note the two recovery scales]")


if __name__ == "__main__":
    fig1(); fig3(); fig4(); fig5()
    print(f"\nwrote 4 figures to {OUT}/")
    print("Fig. 1 is computed from scratch and is reproducible as-is.")
    print("Figs. 3-5 use transcribed numbers -- swap in a log parser")
    print("before submission so they cannot drift from the tables.")