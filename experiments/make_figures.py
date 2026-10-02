#!/usr/bin/env python3
"""Figures of the paper are built in ``paper/figures/`` (this file is only a pointer).

The v1.0 version of this script drew the figures from hard-coded numbers and is
obsolete. The current pipeline separates computation from plotting:

    python paper/figures/figures_data.py   # experiment outputs -> paper/figures/data/make_figures.npz
    python paper/figures/make_figures.py   # npz -> paper/figures/fig_{safety,budget,device,scale}.pdf

See ``paper/README.md`` for the full map of tables, figures and numbers. Running this
file runs the two steps above.
"""
import os
import subprocess
import sys

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")

if __name__ == "__main__":
    for step in ("figures_data.py", "make_figures.py"):
        subprocess.run([sys.executable, os.path.join(FIG, step)], check=True)
