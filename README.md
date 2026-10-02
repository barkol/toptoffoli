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

## Maintained and deprecated parts

**Maintained (v1.2):** the certified decomposition pass of the paper —
`toffoli_optimizer.core` modules `decomposition_selector`, `subspace_check`,
`reach_local`, `reachable_subspace`, `window_pairs`, `orientation`,
`context_analysis`, `phase_observability`, `error_model`, `equivalence_verifier`,
`scalable_verification` and `toffoli_count_reducer`, plus the `experiments/` scripts.

**Deprecated (legacy v1.0, May 2025):** `core.compiler`, `core.toffoli_depth_optimizer`,
`core.circuit_gate_processor`, `core.pattern_library_module`, the `utils` package and
the `toffoli-optimizer` command-line tool. They emit a `DeprecationWarning` when used
and are kept only to reproduce the first release. An audit on 2026-10-02 found
confirmed correctness bugs in them (among others: the default Toffoli decomposition
is not a CCX; pattern rewrites not equivalent to CCX out of context; gates silently
dropped or moved to wrong qubits; command-line tools that write circuits not
equivalent to their input). Each bug is pinned as a strict `xfail` test in
`tests/unit/`. Do not use these modules for results that must be correct.

**Certificates (v1.2):** all decision-diagram certificates call MQT QCEC with
`run_zx_checker=False`, `elide_permutations=False` and `trace_threshold=1e-12`.
With the defaults used up to v1.1, QCEC could report `equivalent` for circuits that
differ on the input subspace when ancillas are declared, treated a SWAP as the
identity, and accepted differences of order 1e-4. Every QCEC-certified result of the
paper was re-certified with the corrected settings and, independently, by
state-vector simulation on random superposed inputs; all verdicts were unchanged.

## Tests

`pytest tests` runs about 450 tests: regression tests for the soundness of the pass
(including the editor's counterexample), property tests against dense-matrix ground
truth on random circuits, and strict `xfail` tests that document the known bugs of
the deprecated modules.


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

## Use of AI in code development

The code in this repository was developed with substantial help from a large
language model (Claude, Anthropic). It wrote and tested code for the optimizer
(including the subspace admissibility checks, certified windows, the mirrored
gadget, calibration-aware orientation and the error model) and the evaluation
and diagnostic scripts, and it ran and analysed the experiments. The authors
reviewed the code and the results and take full responsibility for them. The
regression tests in `tests/` (run with `pytest tests`) and the input-subspace
certificate, which every output of the pass must pass, are the safeguards we
rely on rather than review alone.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

We acknowledge support from the EuroHPC JU under Horizon Europe Grant
No. 101194322 (**QEC4QEA**), co-funded by the Polish National Centre for
Research and Development (NCBiR) under Decision No.
DWM/EuroHPC/2023/429/2025.

The authors are with the Institute of Spintronics and Quantum Information,
Faculty of Physics and Astronomy, Adam Mickiewicz University, Poznań, Poland.
