#!/usr/bin/env python3
"""
postrun_analyze.py — per-run post-processing for the (s2,s4) sweep.

Combines fit_growth (stability + growth onset) and measure_mode (FFT mode
frequency) into ONE per-run report, and picks the clean PINN-training window
[tmin, onset - margin] automatically.  Emits a one-line summary and (optionally)
appends a row to a shared CSV so a whole sweep builds one table.

Run from inside a run directory:
  python3 ../postrun_analyze.py snapshots/*.h5 --log logger_info.txt --input input.json

Sweep usage (append every run to one table):
  python3 ../postrun_analyze.py snapshots/*.h5 --input input.json \\
      --table ../sweep_table.csv

Mirrors the methods/conventions of fit_growth.py and measure_mode.py so the
numbers match those tools exactly.
"""
import argparse, glob, os, re, json, datetime, csv
import numpy as np
import h5py

# ───────────────────────── KE / growth (fit_growth core) ─────────────────────
KE_LINE = re.compile(r"t=([-\d.eE+]+).*?KE=([-\d.eE+]+)")

def parse_ke(path):
    t, ke = [], []
    with open(path) as f:
        for ln in f:
            m = KE_LINE.search(ln)
            if m:
                t.append(float(m.group(1))); ke.append(float(m.group(2)))
    t, ke = np.array(t), np.array(ke)
    good = ke > 0
    return t[good], ke[good]

def linfit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intc), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = res[0] if len(res) else float(np.sum((y-(slope*x+intc))**2))
    ss_tot = float(np.sum((y-y.mean())**2)) + 1e-300
    return float(slope), 1.0 - ss_res/ss_tot

def find_onset(t, ke, r2min, min_efolds, min_pts):
    """Earliest i with slope>0, R2>=r2min, tail spanning >= min_efolds. -> onset time or None."""
    y = np.log(ke); N = len(t)
    for i in range(0, N - min_pts):
        seg = y[i:]
        if (seg.max() - seg.min()) < min_efolds:
            continue
        slope, r2 = linfit(t[i:], seg)
        if slope > 0 and r2 >= r2min:
            return t[i], 0.5*slope, r2
    return None

def envelope_check(t, ke):
    """Peak-envelope trend over the whole series (pre-onset stability)."""
    pk = [i for i in range(1, len(ke)-1) if ke[i] > ke[i-1] and ke[i] > ke[i+1]]
    if len(pk) < 4:
        return None
    pk = np.array(pk)
    slope, r2 = linfit(t[pk], np.log(ke[pk]))
    return 0.5*slope, r2, t[pk[0]], t[pk[-1]]

# ───────────────────── snapshot FFT (measure_mode core) ──────────────────────
def find_scale(f, sub):
    for name in f['scales'].keys():
        if sub in name.lower():
            a = np.array(f['scales'][name][:])
            if a.ndim == 1:
                return a
    return None

def load_fields(files, fields):
    paths = []
    for p in files:
        paths += glob.glob(p) or ([p] if os.path.exists(p) else [])
    paths = sorted(set(paths), key=lambda fp: h5py.File(fp,'r')['scales/sim_time'][0])
    if not paths:
        raise FileNotFoundError("no .h5 files")
    with h5py.File(paths[0],'r') as f:
        mu = find_scale(f,'mu'); phi = find_scale(f,'phi')
        avail = list(f['tasks'].keys())
    fields = [fld for fld in fields if fld in avail]
    t, data = [], {fld: [] for fld in fields}
    for fp in paths:
        with h5py.File(fp,'r') as f:
            t.append(f['scales/sim_time'][:])
            for fld in fields:
                data[fld].append(f['tasks'][fld][:])
    t = np.concatenate(t)
    data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    return mu, phi, t, data, fields

def parabolic(a, b, c):
    den = (a - 2*b + c)
    return (a - c)/(2*den) if den != 0 else 0.0

def measure_field(data_f, mu, phi, t, m, tmin, tmax, mu0):
    """FFT complex frequency of wavenumber m from one field. Returns (omega, sigma, T, mu_used)."""
    nphi = phi.size
    cm = np.fft.fft(data_f, axis=1)[:, int(m), :] / nphi     # (nt, nmu)
    n = min(cm.shape[0], t.size)
    cm, tt = cm[:n], t[:n]
    j = int(np.argmin(np.abs(mu - mu0))) if mu0 is not None else int(np.argmax(np.mean(np.abs(cm),axis=0)))
    c = cm[:, j]
    msk = np.ones_like(tt, bool)
    if tmin is not None: msk &= tt >= tmin
    if tmax is not None: msk &= tt <= tmax
    c, tw = c[msk], tt[msk]
    N = tw.size
    if N < 8:
        return None
    dt = np.median(np.diff(tw))
    good = np.abs(c) > 0
    sigma = float(np.polyfit(tw[good], np.log(np.abs(c[good])), 1)[0]) if good.sum() >= 2 else np.nan
    C = np.abs(np.fft.fft((c - c.mean())*np.hanning(N)))
    k = int(np.argmax(C))
    off = parabolic(C[(k-1)%N], C[k], C[(k+1)%N])
    f_peak = (k+off)/(N*dt)
    if f_peak > 0.5/dt: f_peak -= 1.0/dt
    omega = -2*np.pi*f_peak
    T = 2*np.pi/abs(omega) if omega != 0 else np.inf
    return omega, sigma, T, mu[j]

# ─────────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="snapshot .h5 files (glob)")
    ap.add_argument("--log", default="logger_info.txt", help="Dedalus KE log")
    ap.add_argument("--input", default="input.json")
    ap.add_argument("--fields", nargs="+", default=["u_theta","u_phi","h"])
    ap.add_argument("--tmin", type=float, default=50.0, help="skip initial transient")
    ap.add_argument("--margin", type=float, default=50.0,
                    help="cut the window this far BEFORE the growth onset")
    ap.add_argument("--mu0", type=float, default=None,
                    help="fix latitude mu for all fields (default: each field's max-amp mu)")
    ap.add_argument("--r2min", type=float, default=0.999)
    ap.add_argument("--min-efolds", type=float, default=5.0)
    ap.add_argument("--min-pts", type=int, default=8)
    ap.add_argument("--out", default="postrun_summary.txt")
    ap.add_argument("--table", default=None,
                    help="append a one-line row to this CSV (created with header if absent)")
    a = ap.parse_args()

    params = json.load(open(a.input)) if os.path.exists(a.input) else {}
    m  = int(params.get("m", 5))
    s2 = params.get("s2"); s4 = params.get("s4")
    G  = params.get("G");  nu4 = params.get("nu4")

    _lines = []
    def emit(s=""): print(s); _lines.append(s)

    emit("="*70)
    emit(f"  postrun_analyze   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    emit(f"  run: {os.path.abspath('.')}")
    emit(f"  m={m}  s2={s2}  s4={s4}  G={G}  nu4={nu4}")
    emit("="*70)

    # ---- growth / stability from the KE log ----
    onset_t, sigma_tail, verdict = None, None, "UNKNOWN"
    if os.path.exists(a.log):
        t, ke = parse_ke(a.log)
        if t.size:
            env = envelope_check(t, ke)
            res = find_onset(t, ke, a.r2min, a.min_efolds, a.min_pts)
            if env is not None:
                se, r2e, ta, tb = env
                emit(f"  envelope (pre-onset): sigma_env={se:+.5f}  R2={r2e:.4f}  "
                     f"peaks t=[{ta:.1f},{tb:.1f}]")
            if res is not None:
                onset_t, sigma_tail, r2t = res
                emit(f"  growth ONSET at t={onset_t:.1f}  (tail sigma={sigma_tail:+.5f}, R2={r2t:.4f})")
                verdict = "STABLE-then-GROWS"
            else:
                onset_t = None
                emit(f"  no growing tail found -> NEUTRAL/stable for full log")
                verdict = "STABLE"
        else:
            emit(f"  [warn] no KE lines parsed from {a.log}")
    else:
        emit(f"  [warn] log '{a.log}' not found -> skipping growth analysis")

    # ---- training window ----
    t_end_win = (onset_t - a.margin) if onset_t is not None else None
    win_tmin, win_tmax = a.tmin, t_end_win
    emit("-"*70)
    emit(f"  PINN window: t=[{win_tmin:.1f}, "
         f"{'end' if win_tmax is None else f'{win_tmax:.1f}'}]  "
         f"(tmin skip transient; tmax = onset - margin{a.margin:g})")

    # ---- FFT mode measurement over that window ----
    mu, phi, tt, data, fields = load_fields(a.files, a.fields)
    omegas, sigmas = [], []
    emit(f"  measured over window (fields: {', '.join(fields)}):")
    for fld in fields:
        r = measure_field(data[fld], mu, phi, tt, m, win_tmin, win_tmax, a.mu0)
        if r is None:
            emit(f"    {fld:9s}: too few samples in window"); continue
        omega, sig, T, muj = r
        omegas.append(omega); sigmas.append(sig)
        direction = "retro" if omega < 0 else "pro"
        emit(f"    {fld:9s}: omega={omega:+.5f}  T={T:8.4f}  sigma={sig:+.5f}  "
             f"mu={muj:+.4f}  ({direction})")

    omega_mean  = float(np.mean(omegas)) if omegas else float('nan')
    omega_spread= float(np.max(omegas)-np.min(omegas)) if len(omegas) > 1 else 0.0
    sigma_mean  = float(np.mean(sigmas)) if sigmas else float('nan')
    sigma_spread= float(np.max(sigmas)-np.min(sigmas)) if len(sigmas) > 1 else 0.0
    emit("-"*70)
    emit(f"  omega_mean={omega_mean:+.5f}  omega_spread={omega_spread:.2e}  "
         f"sigma_mean={sigma_mean:+.5f}  sigma_spread={sigma_spread:.2e}")
    # A CLEAN SINGLE MODE means the same coherent mode is seen in every field:
    # omega AND sigma must agree ACROSS FIELDS (small spread). It does NOT mean
    # the mode is neutral. The old test also required abs(sigma_mean) < 3e-3,
    # which encoded the G=0.01 expectation of a near-neutral mode -- but with
    # nu4_h damping (G=0.25) the physical mode decays at sigma ~ -0.04, and that
    # is not contamination. Cross-field spread is the right cleanliness test;
    # the decay rate itself is a separate physical quantity, reported not gated.
    clean = (omega_spread < 1e-3) and (sigma_spread < 3e-3)
    emit(f"  CLEAN SINGLE MODE: {'YES' if clean else 'NO — cross-field spread too large'}")
    if clean and abs(sigma_mean) > 3e-3:
        emit(f"  (single coherent mode, decaying at sigma={sigma_mean:+.4f} "
             f"-- expected when nu4_h damps; compare the EVP eigenvalue)")
    emit("="*70)

    with open(a.out, "w") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"[saved] {a.out}")

    # ---- append row to sweep table ----
    if a.table:
        header = ["run","m","s2","s4","G","nu4","onset_t","win_tmin","win_tmax",
                  "omega_mean","omega_spread","sigma_mean","sigma_spread","verdict","clean"]
        row = [os.path.basename(os.path.abspath(".")), m, s2, s4, G, nu4,
               f"{onset_t:.2f}" if onset_t else "none",
               f"{win_tmin:.1f}", f"{win_tmax:.1f}" if win_tmax else "end",
               f"{omega_mean:+.5f}", f"{omega_spread:.2e}", f"{sigma_mean:+.5f}",
               f"{sigma_spread:.2e}", verdict, "yes" if clean else "no"]
        new = not os.path.exists(a.table)
        with open(a.table, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(header)
            w.writerow(row)
        print(f"[table] appended row to {a.table}")


if __name__ == "__main__":
    main()