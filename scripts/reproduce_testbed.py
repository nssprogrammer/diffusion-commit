"""Reproduce the exact test-bed results on CPU (Appendix "An Exactly Solvable Test Bed", band table of Section 7).

    python scripts/reproduce_testbed.py            # tilts: full-trajectory and band-limited (about 10 minutes)
    python scripts/reproduce_testbed.py --quick    # fewer samples, one seed (about 1 minute)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # run from a fresh clone

from commitment import testbed as tb

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
N, seeds = (4000, [1]) if args.quick else (20000, [1, 2, 3])

e0 = np.mean([tb.class_imbalance(tb.sample(N, s)) for s in seeds])
full = {m: np.mean([tb.class_imbalance(tb.sample(N, s, m)) for s in seeds]) for m in ['exact', 'first_order', 'shifted']}
print(f"class imbalance: no correction {e0:.4f}; full-trajectory correction: exact {full['exact']:.4f}, "
      f"first-order {full['first_order']:.4f}, shifted {full['shifted']:.4f}")
print(f"{'split':>6} {'high-noise only':>18} {'low-noise only':>18} {'sum':>6}")
for split in [0.5, 1.0, 2.0, 4.0]:
    share = {}
    for band in ['high', 'low']:
        e = np.array([tb.class_imbalance(tb.sample(N, s, 'exact', band, split)) for s in seeds])
        share[band] = (e0 - e) / (e0 - full['exact'])
    print(f"{split:6.1f} {share['high'].mean():+10.2f} +- {share['high'].std():.2f} {share['low'].mean():+10.2f} +- "
          f"{share['low'].std():.2f} {share['high'].mean() + share['low'].mean():+6.2f}")
