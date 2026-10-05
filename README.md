# When Does a Diffusion Model Decide What to Draw?

Code for the paper **"When Does a Diffusion Model Decide What to Draw?"** by Snigdha Chandan Khilar.

A diffusion model starts from pure noise and removes it step by step. Somewhere along the way it becomes committed
to, say, a horse rather than a truck. This repository measures **when** that happens and **what a trained model gets
wrong about it**, on CIFAR-10, with two instruments:

* **Committors.** Pause generation, restart it many times from the same half-finished image, and count how often
  each class comes out.
* **Density transitions.** The noise level at which a classifier's opinion about many images splits from one group
  into two, together with how lopsided that split is (the cusp normal form).

The main findings: coarse decisions come before fine ones; along the class axis NVIDIA's EDM model commits like its
data except for class proportions; correcting those proportions inside the sampler fixes commitment; and an older
DDPM model also commits about 20% late. See [`paper/paper.pdf`](paper/paper.pdf).

---

## Repository layout

```
commitment/            Python package with the shared code
  density.py           density transitions sigma_bif and lopsidedness h       (NumPy/SciPy)
  stats.py             class-block bootstrap, Benjamini-Hochberg, Spearman-Brown, class-imbalance error
  calibration.py       monotone cross-fitted recalibration of log-odds, tilting identity
  committor.py         committors, decision levels, cross-over rates, committor fidelity
  testbed.py           the exactly solvable six-class Gaussian-mixture test bed
  classifiers.py       noise-conditional ResNet (FiLM) and VGG classifiers     (PyTorch)
  sampling.py          EDM schedule, reverse SDE with recording/branching, Heun, DDPM wrapper
  guidance.py          class-proportion corrections (first-order and exact)
  evaluation.py        ViT labelling, FID
notebooks/             the notebooks that produced every number in the paper (run on Kaggle, one T4 GPU)
  00 ... 12            main results, in run order
  appendix/            appendix results (linear directions, digits, small models, detectors)
  archive/             earlier development steps, kept for transparency; superseded and not needed
scripts/
  reproduce_testbed.py exact test-bed results on CPU
  make_figures.py      regenerates all figures from the recorded results
paper/                 LaTeX source, figures, figure script and the recorded per-pair results
tests/                 CPU tests, including a check that the package reproduces the notebook code exactly
```

## Installation

```bash
git clone https://github.com/nssprogrammer/diffusion-commit.git
cd diffusion-commit
pip install -e .            # CPU modules: numpy, scipy, scikit-learn, matplotlib
pip install -e ".[gpu]"     # adds torch, torchvision, transformers, diffusers, pytorch-fid
pytest -q                   # CPU tests (about 5 seconds)
```

## Quick checks on a laptop (CPU only)

```bash
python scripts/reproduce_testbed.py --quick   # test-bed band table, about 1 minute (full: drop --quick, ~10 min)
python scripts/make_figures.py                # all paper figures from the recorded results
```

## Reproducing the paper (GPU)

All GPU experiments were run as **Kaggle notebooks on a single NVIDIA T4 (16 GB)**, about 80 GPU-hours in total.
Each notebook is self-contained: it downloads CIFAR-10, the EDM code and checkpoints, the ViT and DDPM weights
itself. Turn on **GPU** and **Internet** and use **Save & Run All (commit)**.

Some notebooks read files written by earlier ones. On Kaggle, add the earlier notebook's output as an input
(**Add Input → Notebook Output**); the notebooks find files anywhere under `/kaggle/input/`. Elsewhere, put the
files in a folder and change `/kaggle/input` and `/kaggle/working` in the first cell.

| # | notebook | needs outputs of | writes | GPU time |
|---|---|---|---|---|
| 00 | `00_exact_testbed__step26` | none (CPU) | printed tables | ~10 min CPU |
| 01 | `01_pretrained_edm_main_run__step19` | none | `noisy_resnet_seed0.pt`, `step19_logits.npz` | ~5 h |
| 02 | `02_posthoc_asymmetry__step19b` | 01 | printed tables | CPU |
| 03 | `03_census_coordinate_null__step20` | 01 (classifier) | `step20_logits.npz`, `noisy_resnet30.pt`, `noisy_vgg.pt` | ~3.5 h |
| 04 | `04_census_calibrated_coordinate__step20b` | 03 | `step20b_calibrated.npz` | ~1.5 h |
| 05 | `05_interventions_and_sampler_checks__step21` | 01 | `step21_*.npz`, `step21_heun_fp32.npy` | ~8.5 h |
| 06 | `06_mass_to_place_and_statistics__step22` | 01, 03, 05 | printed tables | CPU, ~2 h |
| 07 | `07_committor_and_conditional_model__step23` | 01 | `step23_committor.npz`, `step23_conditional.npz` | ~5 h |
| 08 | `08_pair_decisions_and_location__step23b` | 01, 07 | printed tables | CPU |
| 09 | `09_corruptions_and_bands__step24` | 01, 05 | `step24_*.npz` | ~6.5 h |
| 10 | `10_direct_test_planted_errors_lags__step25` | 01, 07 | printed tables | CPU, ~2 h |
| 11 | `11_sampler_exact_tilt_ddpm__step27` | 01 | `step27_*.npz` | ~8 h (sections switchable) |
| 12 | `12_intervals__step27b` | 01, 11 | printed tables | CPU |

The file names keep the original step numbers (`step19_logits.npz`, ...) because the notebooks look files up by name.

## Where each result in the paper comes from

| paper | result | notebook |
|---|---|---|
| Fig. 1 | pitchfork vs fold (analytic) | `paper/make_all_figs.py` |
| §3.1, §5, Fig. 3a | committors: restarts, superclass/class decisions | 07 |
| §3.1, §5, Fig. 3b, decisions table | pair decisions by threshold and measuring stick | 08, 10 (part 4) |
| §3.1, App. C | committor estimators against exact committors | 00 |
| §3.2, Fig. 4, App. D | census classifier, density census, VGG check, warps, symmetric-pair noise floor | 03 |
| §3.2, App. D | recalibrated coordinate, tilting identity | 04 |
| §3.3, App. B | reverse SDE vs ODE on an exact three-class mixture (1.00 vs 1.36) | 01 |
| §3.4 | pipeline check on the data alone (slope 1.00) | 06 (part C) |
| §3.4 | imposed imbalance through guidance (slope 1.05) | 05 |
| §3.5, Fig. 2 | planted errors (power of the comparison) | 10 (part 3) |
| §5 | hierarchy 5.8x [3.9, 8.1], Benjamini-Hochberg 19/44 | 06 (part D) |
| §5, App. D | Gaussian (second-order) baseline | 03, 04 |
| §6.1, Fig. 5 | raw model vs data, frequencies, asymmetry vs over-production | 01, 02, 05 |
| §6.1, Fig. 6a | timing vs lopsidedness errors, cusp prediction, split halves | 06 (part A) |
| §6.2, models table | data reweighted to the model's class proportions (0.974 vs 0.981) | 10 (part 1) |
| §6.2 | rebalanced asymmetry against the Spearman-Brown ceiling | 06 (part B) |
| §6.3 | class-conditional EDM with equal labels | 07 (B), 08 |
| §6.4, sampler table | 256 / 512 / 1,024 steps in full precision, Heun, ViT | 11 (A), 05 |
| §6.4 | noise floor of the class-imbalance error | 12, `commitment.stats` |
| §6.5, Fig. 7 | committor fidelity, adjustment for class proportions, class proportions along generation | 08, 10 (part 2), 07 |
| §7, correction table | first-order correction (one round) | 05 |
| §7, correction table | exact correction, second round | 11 (B) |
| §7, band table | band-limited corrections on the exact test bed | 00, `scripts/reproduce_testbed.py` |
| §7 | the earlier single-split CIFAR-10 band test | 09 |
| §8 | DDPM, intervals | 11 (C), 12 |
| App. H, corruption figure | score scaling and noise-level mismatch | 09 |
| App. E | linear directions and soft modes on CIFAR-10 | appendix/A1 |
| Notes below | two independently trained census classifiers agree (correlation 0.99) | appendix/A2 |
| App. I, digits census | MNIST / Fashion-MNIST census, floors, exponent calibration | appendix/A3 |
| App. I, small models | transitions of small trained networks | appendix/A4 |
| App. I | drift replacement, sham and random-direction controls | appendix/A5, appendix/A3 |
| App. I, band table | band-limited drift replacement | appendix/A6 |
| App. J | does the fold's shape change sampling? (d = 784, exact score) | appendix/A3 |
| App. K | entropy-variance detectors | appendix/A7, appendix/A3 |
| App. M | per-pair tables | 03 (census), 06 (pretrained) |

All figures are regenerated by `scripts/make_figures.py` from `paper/data*.py`, which hold the per-pair results as
printed by the notebooks above. The committor curves in Fig. 3a were read off the markers of notebook 07's plot
(accurate to about 0.003).

## Notes on reproducibility

* Sampling uses fixed seeds, but GPU nondeterminism, library versions and checkpoint downloads mean reruns agree
  with the paper up to sampling noise, not to the last digit. Every number in the paper is reported with an interval.
* The census classifier is retrained by notebook 01 if no saved copy is mounted; a retrained classifier gives the same
  results up to the variation reported in the paper (two seeds: correlation 0.99 for lopsidedness and transitions).
* Kaggle sessions stop after 12 hours. Notebook 11 has switches (`RUN_A`, `RUN_B`, `RUN_BANDS`, `RUN_C`) so it can be
  split over two sessions.

## Third-party assets and licences

The code in this repository is released under the MIT licence (see `LICENSE`). It downloads, at run time:

* CIFAR-10 (Krizhevsky, 2009), MNIST and Fashion-MNIST (MIT licence) through torchvision;
* NVIDIA's EDM code and checkpoints, licensed **CC BY-NC-SA 4.0** (non-commercial use only);
* the DDPM CIFAR-10 checkpoint `google/ddpm-cifar10-32` and the ViT `aaraki/vit-base-patch16-224-in21k-finetuned-cifar10`
  from the Hugging Face Hub (see their model cards for licences);
* Inception weights for FID through `pytorch-fid`.

None of these are redistributed here.

## Citation

```bibtex
@misc{khilar2026decide,
  title  = {When Does a Diffusion Model Decide What to Draw?},
  author = {Khilar, Snigdha Chandan},
  year   = {2026},
  note   = {Code: https://github.com/nssprogrammer/diffusion-commit}
}
```

Contact: Snigdha Chandan Khilar, snkhilar@gmail.com
