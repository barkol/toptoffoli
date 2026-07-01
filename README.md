# toptoffoli

Companion code to

> K. Bartkiewicz and P. Tulewicz,
> **"Context-Verified, Error-Budget-Aware Decomposition Selection for Toffoli Networks"**,
> [arXiv:2606.31791](https://arxiv.org/abs/2606.31791) (2026), submitted to *Quantum*.

A compiler pass that selects, per Toffoli gate, the decomposition that
minimizes a **hardware two-qubit-infidelity error budget** — not gate count —
and admits every context-dependent (relative-phase or bounded-approximate)
decomposition **only when a per-instance equivalence check certifies it in
context**. Pattern-matched relative-phase substitution as deployed in current
tooling is silently incorrect; the verification gate makes aggressive
optimization sound while keeping essentially all of the savings.

The package ships the optimizer, the exact and scalable equivalence
verifiers, the reachable-subspace and phase-observability admissibility
checks, and — under `experiments/` — the complete drivers that regenerate
every empirical figure and table in the paper.

## Method in three sentences

1. **Objective.** Minimize the two-qubit-infidelity budget of the emitted
   circuit under the device's `p_{2q}` — not the CCX or two-qubit gate count.
2. **Admission.** For each Toffoli site, consider the exact 6-CX decomposition
   and every cheaper context-dependent alternative (relative-phase RCCX,
   bounded-approximate, ancilla-augmented). Reject an alternative unless
   either (a) reachable-subspace + phase-observability analysis certifies
   soundness, or (b) an exact equivalence check on the enclosing window
   confirms it.
3. **Verification.** Below ~10 qubits, exhaustive truth-table / unitary
   comparison; above, decision-diagram equivalence (MQT QCEC) certifies each
   accepted pair substitution up to global phase.

## Quickstart

```bash
git clone https://github.com/barkol/toptoffoli
cd toptoffoli
pip install -e ".[experiments,dev]"

pytest                                        # unit tests, ~1 min
python experiments/safety_experiment.py       # reproduces Fig. `safety`, ~30 s
python experiments/make_figures.py            # writes the four paper PDFs
```

## Reproducing every figure and table

Every empirical panel of the paper has a driver in `experiments/`. See
[`experiments/README.md`](experiments/README.md) for the full mapping
(paper artefact → script → runtime) and
[`docs/reproduction.md`](docs/reproduction.md) for step-by-step commands
and expected outputs.

Headline drivers:

| Paper artefact                       | Script                                 | Runtime |
|---                                   |---                                     |---      |
| Fig. `safety`                        | `experiments/safety_experiment.py`     | ~30 s   |
| Table (b) — verification-gate ablation | `experiments/ablation_b.py`          | ~10 s   |
| Fig. `budget`                        | `experiments/baselines.py`             | ~1-5 min (with/without tket) |
| Fig. `device`                        | `experiments/sensitivity_sweep.py`     | ~5-10 min |
| Fig. `scale` (12-24 q)               | `experiments/scale_eval.py`            | ~2 h    |
| Table / Fig. `reset` (application)   | `experiments/sync_benchmark.py`        | ~3-10 min |
| Density-matrix noise validation      | `experiments/noisy_sim.py`             | ~5 min  |
| All four figure PDFs                 | `experiments/make_figures.py`          | <5 s    |

## Repository layout

```
toptoffoli/
├── toffoli_optimizer/    # the package (core algorithms, CLI, utilities)
├── experiments/          # drivers that reproduce every paper figure/table
├── tests/                # pytest suite
├── examples/             # standalone demos + example input circuits
├── benchmarks/           # micro-benchmarks (not the paper's suite)
└── docs/                 # methodology + reproduction guide
```

## Requirements

- Python ≥ 3.9
- `qiskit` ≥ 1.2, `numpy`, `matplotlib` — always required
- Optional (for `experiments/`):
  - `qiskit-aer` — density-matrix noise validation
  - `pytket`, `pytket-qiskit` — the tket baseline in `baselines.py`
  - `mqt.qcec` — decision-diagram verification above the exhaustive limit

## Citation

If you use this software or reproduce results from the paper, please cite:

```bibtex
@article{bartkiewicz2026toptoffoli,
  title   = {Context-Verified, Error-Budget-Aware Decomposition Selection
             for Toffoli Networks},
  author  = {Bartkiewicz, Karol and Tulewicz, Patrycja},
  journal = {arXiv preprint arXiv:2606.31791},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.31791},
  note    = {Submitted to Quantum.}
}
```

Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff)
(rendered by GitHub as a "Cite this repository" widget in the sidebar).

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

We acknowledge support from the EuroHPC JU under Horizon Europe Grant
No. 101194322 (**QEC4QEA**), co-funded by the Polish National Centre for
Research and Development (NCBiR) under Decision No.
DWM/EuroHPC/2023/429/2025.

The authors are with the Institute of Spintronics and Quantum Information,
Faculty of Physics and Astronomy, Adam Mickiewicz University, Poznań, Poland.
