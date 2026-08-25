# Closed-form recovery of differential-rotation parameters from equatorial Rossby waves

Code and manuscript accompanying Shang, Zaqarashvili & Veronig (submitted to
*Astronomy & Astrophysics*).

The method recovers the differential-rotation parameters `s2` and `s4` from a
single equatorial Rossby mode in a shallow-water layer. Because the linearised
equations are exactly linear in those parameters, the recovery is a closed-form
linear least-squares solve rather than a search.

---

## Repository layout

```
paper/          manuscript source (ms_aa.tex, references.bib) and figure PDFs
code/           the analysis chain
figures/        make_figures.py -- regenerates every figure in the paper
logs/           run logs backing every number in the tables and figures
```

`aa.cls` and `aa.bst` are **not** included: they belong to EDP Sciences and must
be downloaded from
<https://www.aanda.org/for-authors/latex-issues/texnical-background>.

Simulation snapshots and neural-network checkpoints are **not** in git — they
are far too large. See *Data* below.

---

## The analysis chain

Run in this order. Every script takes `--help`.

| script | role |
|---|---|
| `spherical-2d-hd-s2s4.py` | Dedalus forward model: solves the linearised shallow-water system and writes snapshots |
| `evp_spherical_hd.py` | linear eigenvalue problem; supplies the initial condition |
| `postrun_analyse.py` | finds the growth onset, which sets the analysis window (`t_win_max`) |
| `lsq_from_data.py` | **the estimator.** Recovers `s2`/`s4`; carries every degradation flag |
| `los_inversion.py` | forward-projects fields to synthetic Dopplergrams, then inverts back |
| `select_settings.py` | truth-free choice of `(K, r)` on a network checkpoint |
| `refine_lsq.py` | post-hoc `(K, r)` scan on a saved field |
| `lsq_hfree.py` | layer-thickness elimination; the cross-check of Appendix B |
| `PINN_s2s4_recovery.py` | the physics-informed network of Sect. 6 |
| `sw_model.py` | shared model definitions |

### Reproducing the main result

```bash
# 1. recovery on clean simulation output (Table 1)
python3 code/lsq_from_data.py --input <case>/input.json --tmin 10 --cheb 30 --svd 0

# 2. degradation tests (Sect. 4, Fig. 1)
python3 code/lsq_from_data.py --input <case>/input.json --tmin 10 \
        --noise 0.1 --limb-weight best --mask-poles 60

# 3. synthetic Dopplergrams and the inversion (Sect. 5)
python3 code/los_inversion.py --input <case>/input.json --tmin 10 \
        --b0 7.25 --noise 0.05 --out los_ckpt

# 4. recovery from the inverted fields (Table 2)
python3 code/lsq_from_data.py --input <case>/input.json --ckpt los_ckpt \
        --field pred --tmin 10 --mask-poles 60 --eq-subset 2
#    ... and --field true for the control
```

Two conventions worth knowing before you read the code:

- **`--field true` is the control, not the truth.** It applies the estimator to
  the directly computed m-mode fields from the same checkpoint, which separates
  errors introduced by the inversion from errors in the estimator.
- **Equation labelling.** The paper calls the three equations eq1, eq2 and eq3.
  The code labels the continuity equation `5`, so `--eq-subset 2` is eq2 of the
  paper and `--eq-subset 5` is eq3.

### Recommended configuration

For line-of-sight-derived fields, as set out in the conclusions:

```
--eq-subset 2 --mask-poles 60 --cheb 20 --svd 2      with B0 != 0
```

`B0 = 0` is a **singular** geometry (Sect. 5.2) — the inversion has no solution
there, and `los_inversion.py` will say so.

---

## Figures

```bash
cd figures && python3 make_figures.py
```

Figure 3 (geometry and conditioning) is computed from the projection geometry
alone and reproduces exactly.

Figures 1, 2 and 4 currently read their numbers from a transcription block at
the top of `make_figures.py`, taken from the run logs in `logs/`. The values are
correct as transcribed, but a transcription is not a pipeline: **before the
paper is finalised, replace that block with a parser that reads `logs/`
directly**, so the figures cannot drift from the tables. The script says so on
every run.

---

## Data

Snapshots and checkpoints are too large for git. Deposit them on Zenodo and put
the DOI here; GitHub can mint one automatically from a release
(<https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content>).

What to deposit:

- Dedalus snapshots for the six `(s2, s4)` cases of Table 1
- the `los_*` checkpoints behind Table 2
- the network checkpoints of Sect. 6

What is already in git: `logs/`, which contains the console output of every run
quoted in the paper. Those logs are the provenance for the tables and figures,
and they are small.

---

## Requirements

See `requirements.txt`. The network additionally needs DeepXDE with a PyTorch
backend and a GPU; nothing else does.

---

## Citation

TODO: add the paper reference and the Zenodo DOI once available.

## License

TODO: choose one. MIT or BSD-3-Clause is usual for analysis code; CC-BY for the
manuscript text. Note that the license must not purport to cover `aa.cls`,
which is not redistributed here.
