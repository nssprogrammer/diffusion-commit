# Notebooks

* `00` to `12`: the notebooks behind the main results, in run order. See the run-order table and the
  paper-to-notebook map in the top-level `README.md`.
* `appendix/`: notebooks behind the appendix results on digits, small networks, linear directions and detectors.
  Each is self-contained.
* `archive/`: earlier development steps (step 1 to step 15, and the unfinished step 21b). They are kept for
  transparency, are superseded by the notebooks above, and are not needed to reproduce the paper.

The suffix `__stepNN` is the step number under which each notebook was run. Output files keep these step numbers
(`step19_logits.npz`, ...) because later notebooks look them up by name. Stored outputs have been stripped.
