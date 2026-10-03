# toptoffoli

Companion code to

> K. Bartkiewicz and P. Tulewicz,
> **"Certified Context-Dependent Toffoli Decompositions with Error Accounting:
> Fewer Two-Qubit Gates and Subroutine Guarantees"**,
> [arXiv:2606.31791](https://arxiv.org/abs/2606.31791) (2026), submitted to *Quantum*.
> (Version 1 of the preprint appeared under the title "Context-Verified, Error-Budget-Aware
> Decomposition Selection for Toffoli Networks".)

A compiler pass that selects, per Toffoli gate, the decomposition that
minimizes a **hardware two-qubit-infidelity error budget** — not gate count —
and admits a context-dependent (relative-phase or control-drop) decomposition
**only under an operator condition on the reachable input subspace**, then certifies
the whole output against the exact decomposition. In the default subroutine semantics
the output equals the exact circuit on the input subspace up to one global phase, so
it can be used inside a larger algorithm. Pattern-matched relative-phase substitution
as deployed in current tooling is silently incorrect; the certified pass keeps most of
the savings without that risk.

The package ships the optimizer, the exact and scalable equivalence
verifiers, the reachable-subspace and phase-observability admissibility
checks, the experiment drivers under `experiments/`, and under `paper/` the scripts
and stored data that rebuild every table, figure and generated number of the paper.

## Method in three sentences

1. **Objective.** Minimize the two-qubit-infidelity budget of the emitted
   circuit under the device's `p_{2q}` — not the CCX or two-qubit gate count.
2. **Admission.** For each Toffoli, consider the exact 6-CX decomposition
   and the cheaper context-dependent alternatives (3-CX relative-phase gadgets,
   control drops). Admit one only under condition (C) (a compute/uncompute pair,
   checked exactly), (W) (a certified window between mirrored gates) or (R) (equality
   with CCX up to one phase on the span of the reachable basis states). Condition (U)
   (phase invisible to a terminal measurement) is available only in program semantics.
3. **Certification.** The whole output is certified against the exact decomposition on
   the input subspace: densely up to 12 qubits, by decision diagrams (MQT QCEC, pinned
   qubits as ancillas) above. If the certificate fails or is undecided, the pass
   returns the exact decomposition.

## Quickstart

```bash
git clone --branch v1.2.1 https://github.com/barkol/toptoffoli
cd toptoffoli
pip install -e ".[experiments,dev]"

pytest tests                                  # 502 passed, 87 xfailed (~6 min)
paper/reproduce.sh                            # rebuild tables, figures, numbers of the paper
git status paper/                             # empty: outputs identical to the stored ones
```

## Reproducing the paper

[`paper/README.md`](paper/README.md) maps every table, figure and generated number of the
paper to the script that builds it and the data it reads, and lists the order in which
to run the experiment drivers:

1. `experiments/` drivers (hours) write the raw results: 12-circuit suite
   (`safety_experiment.py`, `ablation_b.py`, `baselines.py`, `sensitivity_sweep.py`,
   `noisy_sim.py`), resetting circuits (`sync_benchmark.py`, `sync_scale.py`),
   12–24-qubit suite (`scale_eval_ckpt.py`, `scale_program.py`), RevLib
   (`revlib/fetch_revlib.py`, `revlib/revlib_eval.py`, `revlib/revlib_extra.py`),
   IBM mirror tests (`mirror_test_ibm.py`, `mirror_test_ibm_aachen.py`), error budget on hardware (`budget/`).
2. `paper/reproduce.sh` rebuilds from the stored results the tables
   (`paper/tables/make_tab_{semantics,mirror,mirror_aachen,revlib}.py`), the figure data and the
   figures (`paper/figures/figures_data.py`, `paper/figures/make_figures.py`) and the
   numbers of the text (`paper/liczby/liczby_v14.py` -> `liczby.json`). The outputs are
   byte-identical to the files of the manuscript (figure PDFs pixel-identical).
3. `paper/recert/` holds the recertification of every QCEC result with the corrected
   QCEC settings and a state-vector check; `paper/ablation/` the ablation of all
   verification; `paper/data_v12/` a rerun of the 12–24-qubit suite and RevLib with
   the release code.

## Repository layout

```
toptoffoli/
├── toffoli_optimizer/    # the package (core algorithms, CLI, utilities)
├── experiments/          # experiment drivers and their stored raw results
├── paper/                # tables, figures and numbers of the paper (scripts + data)
├── tests/                # pytest suite
├── examples/             # standalone demos + example input circuits
├── benchmarks/           # micro-benchmarks (not the paper's suite)
└── docs/                 # methodology + reproduction guide
```

## Requirements

- Python ≥ 3.10 (the paper used 3.12)
- `qiskit` ≥ 2.4, `numpy`, `matplotlib` — always required
- Optional (for `experiments/` and `paper/`):
  - `qiskit-aer` ≥ 0.17 — density-matrix noise validation, mirror-test plan
  - `pytket` ≥ 2.18, `pytket-qiskit` ≥ 0.77 — the tket baseline in `baselines.py`
  - `mqt.qcec` ≥ 3.6.1 — decision-diagram certificates above 12 qubits. The certificates
    pass `run_zx_checker`, `elide_permutations` and `trace_threshold` and fail closed
    if the installed QCEC does not accept them.
  - `scipy` — statistics of the hardware analysis
  - `qiskit-ibm-runtime` — only for submitting the IBM jobs
  - a LaTeX installation — the figures are typeset with `text.usetex`
- The minimum versions are the versions used for the paper; the full list of that
  environment is in [`paper/environment.txt`](paper/environment.txt).

## Maintained and deprecated parts

**Maintained (v1.2.x):** the certified decomposition pass of the paper —
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

**Certificates (since v1.2):** all decision-diagram certificates call MQT QCEC with
`run_zx_checker=False`, `elide_permutations=False` and `trace_threshold=1e-12`.
With the defaults used up to v1.1, QCEC could report `equivalent` for circuits that
differ on the input subspace when ancillas are declared, treated a SWAP as the
identity, and accepted differences of order 1e-4. Every QCEC-certified result of the
paper was re-certified with the corrected settings and, independently, by
state-vector simulation on random superposed inputs; all verdicts were unchanged.

## Tests

`pytest tests` runs 589 tests (502 pass, 87 are strict `xfail`): regression tests for the soundness of the pass
(including the editor's counterexample), property tests against dense-matrix ground
truth on random circuits, and strict `xfail` tests that document the known bugs of
the deprecated modules.

## Citation

If you use this software or reproduce results from the paper, please cite:

```bibtex
@article{bartkiewicz2026toptoffoli,
  title   = {Certified Context-Dependent {Toffoli} Decompositions beyond
             Compute--Uncompute Pairs: Fewer Two-Qubit Gates with
             Subroutine Guarantees},
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
