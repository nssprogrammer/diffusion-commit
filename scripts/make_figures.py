"""Regenerate every figure of the paper into paper/figs/ from the recorded results in paper/data*.py."""
import os
import runpy

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper')
os.chdir(here)
runpy.run_path('make_all_figs.py', run_name='__main__')
