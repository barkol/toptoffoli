"""Tests for the CLI layer: toffoli_optimizer.scripts (main / optimize /
base_optimizer / reduce_count).

CLI runs happen inside tmp_path (cwd is moved there), either in-process via
``main()`` with a patched ``sys.argv`` or as a subprocess for exit codes.
Outputs are verified independently (qiskit's own QASM reader + Operators).
Confirmed bugs are ``xfail(strict=True)``.
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from toffoli_optimizer.core.optimizer import OptimizationStrategy, ToffoliDepthOptimizer
import importlib

main_mod = importlib.import_module("toffoli_optimizer.scripts.main")  # package attr `main` is the function
from toffoli_optimizer.scripts.base_optimizer import BaseOptimizer
from toffoli_optimizer.scripts.optimize import Optimize
from toffoli_optimizer.scripts.reduce_count import ReduceCount
from toffoli_optimizer.utils.circuit_utils import get_default_coupling_map

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # main() may flip this class attribute (--verify-equivalence); always restore.
    monkeypatch.setattr(ToffoliDepthOptimizer, "_verify_equivalence_default",
                        getattr(ToffoliDepthOptimizer, "_verify_equivalence_default", False),
                        raising=False)


def _write_net(tmp_path, name, instructions, num_qubits):
    with open(tmp_path / f"{name}.json", "w") as f:
        json.dump({"num_qubits": num_qubits, "instructions": instructions}, f)
    return str(tmp_path / name)


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["toffoli-optimizer", *argv])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main_mod.main()
    return rc, buf.getvalue()


def _read_qasm(path):
    """Independent reader: qiskit's legacy-compatible QASM 2 importer."""
    return QuantumCircuit.from_qasm_file(str(path))


def _subprocess_main(tmp_path, *argv):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, MPLBACKEND="Agg", OMP_NUM_THREADS="4")
    return subprocess.run([sys.executable, "-m", "toffoli_optimizer.scripts.main", *argv],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)


# =========================================================================== #
# argument parsing / exit codes (subprocess)
# =========================================================================== #

class TestCLIExitCodes:
    def test_no_command_prints_help(self, tmp_path):
        r = _subprocess_main(tmp_path)
        assert r.returncode == 0 and "optimize" in r.stdout and "reduce-count" in r.stdout

    def test_unknown_command(self, tmp_path):
        r = _subprocess_main(tmp_path, "bogus")
        assert r.returncode == 2

    def test_missing_required_input(self, tmp_path):
        r = _subprocess_main(tmp_path, "reduce-count")
        assert r.returncode == 2 and "--input" in r.stderr

    def test_invalid_choice(self, tmp_path):
        r = _subprocess_main(tmp_path, "optimize", "--input", "x", "--topology", "torus")
        assert r.returncode == 2

    def test_no_command_in_process(self, monkeypatch):
        rc, out = _run_main(monkeypatch)
        assert rc == 0 and "Toffoli Depth Optimizer" in out


# =========================================================================== #
# reduce-count
# =========================================================================== #

class TestReduceCount:
    def test_end_to_end_equivalence(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "net", [
            {"name": "ccx", "qubits": [0, 1, 2]},
            {"name": "ccx", "qubits": [0, 1, 2]},
            {"name": "ccx", "qubits": [1, 2, 3]},
            {"name": "mcx", "qubits": [0, 1, 2, 4]},
        ], 5)
        out = tmp_path / "red.qasm"
        rc, log = _run_main(monkeypatch, "reduce-count", "--input", inp, "--output", str(out))
        assert rc == 0
        ref = QuantumCircuit(5)
        ref.ccx(0, 1, 2)
        ref.ccx(0, 1, 2)
        ref.ccx(1, 2, 3)
        ref.mcx([0, 1, 2], 4)
        got = _read_qasm(out)
        assert Operator(got).equiv(Operator(ref))
        assert "count: 4 -> 2" in log
        assert sum(n for g, n in got.count_ops().items() if g in ("ccx", "mcx")) == 2

    def test_allow_permutation_flags(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "p", [{"name": "ccx", "qubits": [0, 1, 2]},
                                         {"name": "ccx", "qubits": [0, 1, 3]}], 4)
        out = tmp_path / "p.qasm"
        rc, log = _run_main(monkeypatch, "reduce-count", "--input", inp, "--output", str(out),
                            "--allow-permutation", "--no-expand-to-cancel", "--no-fanout-cse")
        assert rc == 0 and out.is_file()
        assert "output-wire permutation" in log.lower()

    def test_unwritable_output(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "w", [{"name": "ccx", "qubits": [0, 1, 2]}], 3)
        rc, _ = _run_main(monkeypatch, "reduce-count", "--input", inp,
                          "--output", str(tmp_path / "no" / "such" / "dir.qasm"))
        assert rc == 1

    def test_build_atomic_circuit(self):
        rc = ReduceCount(SimpleNamespace(num_qubits=7))
        qc = rc._build_atomic_circuit([([], 0), ([0], 1), ([0, 1], 2), ([0, 1, 2], 3)])
        assert qc.num_qubits == 7
        assert [i.operation.name for i in qc.data] == ["x", "cx", "ccx", "mcx"]
        assert [[qc.find_bit(q).index for q in i.qubits] for i in qc.data] == \
            [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        assert ReduceCount(SimpleNamespace(num_qubits=None))._build_atomic_circuit([([0, 1], 5)]).num_qubits == 6

    @pytest.mark.xfail(strict=True, reason="BUG (data corruption): X gates in the input network are "
                       "dropped by define_loaded_toffoli_network, so reduce-count writes a circuit that "
                       "is NOT equivalent to its input while reporting 'all rewrites verified "
                       "equivalent: True' (reduce_count.py:56 -> io_utils.py:922-944)")
    def test_x_gates_preserved(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "x", [{"name": "x", "qubits": [0]},
                                         {"name": "ccx", "qubits": [0, 1, 2]}], 3)
        out = tmp_path / "x.qasm"
        rc, _ = _run_main(monkeypatch, "reduce-count", "--input", inp, "--output", str(out))
        ref = QuantumCircuit(3)
        ref.x(0)
        ref.ccx(0, 1, 2)
        assert rc == 0 and Operator(_read_qasm(out)).equiv(Operator(ref))

    @pytest.mark.xfail(strict=True, reason="BUG: a non-existent --input silently falls back to a "
                       "hard-coded 6-gate default network and the command exits 0 "
                       "(io_utils.py:949-965; reduce_count.py:57 never sees a failure)")
    def test_missing_input_fails(self, tmp_path, monkeypatch):
        rc, _ = _run_main(monkeypatch, "reduce-count", "--input", str(tmp_path / "nope"),
                          "--output", str(tmp_path / "o.qasm"))
        assert rc != 0


# =========================================================================== #
# optimize (end-to-end, in-process)
# =========================================================================== #

NET4 = [{"name": "ccx", "qubits": [0, 1, 2]}, {"name": "ccx", "qubits": [1, 2, 3]}]


class TestOptimizeCLI:
    def test_standard_end_to_end(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "net", NET4, 4)
        out = tmp_path / "opt.qasm"
        rc, log = _run_main(monkeypatch, "optimize", "--input", inp, "--output", str(out),
                            "--visualize", "--output_dir", str(tmp_path / "res"))
        assert rc == 0
        circ = _read_qasm(out)
        assert circ.num_qubits >= 4
        rep = json.load(open(tmp_path / "opt_report.json"))
        for k in ("logical_depth", "naive_depth", "optimized_depth", "physical_depth",
                  "logical_to_optimized_reduction", "naive_to_physical_reduction"):
            assert k in rep
        if rep["logical_depth"]:
            assert rep["logical_to_optimized_reduction"] == pytest.approx(
                (rep["logical_depth"] - rep["optimized_depth"]) / rep["logical_depth"] * 100)
        assert rep["physical_depth"] == rep["mapped"]["depth"]
        assert not any(k.endswith("circuit") for k in rep["mapped"])
        # --visualize produces nothing (see xfail test_visualize_with_cli_results)
        assert "Optimization Results Table" in log
        # nothing escaped tmp_path: every new file lives below it
        assert all(str(p).startswith(str(tmp_path)) for p in tmp_path.rglob("*"))

    def test_transpiler_strategy(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "t", NET4, 4)
        out = tmp_path / "t.qasm"
        rc, log = _run_main(monkeypatch, "optimize", "--input", inp, "--output", str(out),
                            "--strategy", "TRANSPILER_L1", "--num_qubits", "4")
        assert rc == 0
        circ = _read_qasm(out)
        assert set(circ.count_ops()) <= {"id", "rz", "sx", "x", "cx"}
        rep = json.load(open(tmp_path / "t_report.json"))
        assert rep["mapped"]["name"] == "Transpiler L1"
        assert rep["mapped"]["cx_count"] == circ.count_ops().get("cx", 0)

    def test_verify_equivalence_flag_sets_default(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "v", NET4, 4)
        rc, log = _run_main(monkeypatch, "optimize", "--input", inp, "--output",
                            str(tmp_path / "v.qasm"), "--verify-equivalence", "--num_qubits", "4",
                            "--strategy", "TRANSPILER_L1")
        assert rc == 0 and "correctness gate ENABLED" in log
        # NOTE: process-global side effect (kept after main returns) - see report.
        assert ToffoliDepthOptimizer._verify_equivalence_default is True

    @pytest.mark.xfail(strict=True, reason="BUG (data corruption): --num_qubits defaults to 8 (not "
                       "None), so the 'use the larger value' guard in optimize.py:77-87 is dead; a "
                       "network touching qubit 9 is optimized on 8 qubits, the gate is skipped with a "
                       "warning and the command still exits 0")
    def test_network_wider_than_default(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "big", [{"name": "ccx", "qubits": [0, 1, 9]},
                                           {"name": "ccx", "qubits": [2, 3, 4]}], 10)
        out = tmp_path / "big.qasm"
        rc, _ = _run_main(monkeypatch, "optimize", "--input", inp, "--output", str(out),
                          "--strategy", "TRANSPILER_L1")
        assert rc != 0 or _read_qasm(out).num_qubits >= 10

    @pytest.mark.xfail(strict=True, reason="BUG (cosmetic): --output_dir is only created, nothing "
                       "is ever written into it (optimize.py:52); outputs go to --output / cwd")
    def test_output_dir_used(self, tmp_path, monkeypatch):
        inp = _write_net(tmp_path, "od", NET4, 4)
        od = tmp_path / "resdir"
        rc, _ = _run_main(monkeypatch, "optimize", "--input", inp, "--output", "od.qasm",
                          "--output_dir", str(od), "--strategy", "TRANSPILER_L1", "--num_qubits", "4")
        assert rc == 0 and any(od.iterdir())


# =========================================================================== #
# BaseOptimizer / Optimize helpers
# =========================================================================== #

def _sample_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.ccx(0, 1, 3)
    qc.cx(0, 3)
    qc.t(2)
    qc.ccx(3, 2, 0)
    return qc


CM4 = get_default_coupling_map("linear", 4)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class TestBaseOptimizer:
    def test_strategy_mapping(self):
        b = BaseOptimizer()
        for name in ["STANDARD", "DEPTH_REDUCTION", "GATE_REDUCTION", "FIDELITY", "HYBRID",
                     "ULTRA_DEPTH_REDUCTION", "DEPTH_FIDELITY_BALANCE"]:
            assert b.get_optimization_strategy(name) is getattr(OptimizationStrategy, name)
        assert b.get_optimization_strategy("TRANSPILER_L2") is OptimizationStrategy.STANDARD

    def test_make_serializable(self):
        b = BaseOptimizer()
        src = {"a": 1, "x_circuit": object(), "circuit": object(), "coupling_map": [[0, 1]],
               "nested": {"l": [1, 2.5, None, True, {"logical_circuit": 0}], "s": "t"}}
        out = b.make_serializable(src)
        assert out == {"a": 1, "nested": {"l": [1, 2.5, None, True, {}], "s": "t"}}
        json.dumps(out)
        assert "circuit" in src  # input untouched

    @pytest.mark.xfail(strict=True, reason="BUG (cosmetic): numpy integers are not int subclasses and "
                       "are stringified ('3') in the JSON report (base_optimizer.py:256-260)")
    def test_make_serializable_numpy_ints(self):
        assert BaseOptimizer().make_serializable({"d": np.int64(3)}) == {"d": 3}

    @pytest.mark.xfail(strict=True, reason="BUG: non-string dict keys crash make_serializable "
                       "(key.endswith, base_optimizer.py:250); save_optimization_report then silently "
                       "returns False and writes no report")
    def test_make_serializable_int_keys(self):
        assert BaseOptimizer().make_serializable({1: 2}) == {1: 2}

    def test_optimize_with_transpiler(self):
        qc = _sample_circuit()
        r = _quiet(BaseOptimizer().optimize_with_transpiler, qc, CM4, optimization_level=2)
        c = r["circuit"]
        assert r["physical_depth"] == c.depth()
        assert r["gate_count"] == c.size()
        assert r["cx_count"] == c.count_ops().get("cx", 0)
        assert set(c.count_ops()) <= {"id", "rz", "sx", "x", "cx"}
        assert Operator.from_circuit(c).equiv(Operator(qc))
        allowed = {tuple(e) for e in CM4}
        for inst in c.data:
            if len(inst.qubits) == 2:
                assert tuple(c.find_bit(q).index for q in inst.qubits) in allowed
        exp = (r["original_physical_depth"] - r["physical_depth"]) / r["original_physical_depth"] * 100
        assert r["naive_to_physical_reduction"] == pytest.approx(exp)

    def test_optimize_with_transpiler_failure_returns_none(self):
        big = QuantumCircuit(6)
        big.cx(0, 5)
        assert _quiet(BaseOptimizer().optimize_with_transpiler, big, get_default_coupling_map("linear", 3)) is None

    def test_create_physical_mappings_basic(self):
        r = _quiet(BaseOptimizer().create_physical_mappings, _sample_circuit(), CM4)
        assert set(r) == {"circuit", "depth", "name", "gate_count", "cx_count", "fidelity"}
        assert r["depth"] == r["circuit"].depth()
        assert r["gate_count"] == len(r["circuit"].data)
        assert 0.0 < r["fidelity"] <= 1.0

    @pytest.mark.xfail(strict=True, reason="BUG: the re-optimisation transpile (base_optimizer.py:152) "
                       "has no basis_gates, so the selected 'physical' circuit contains non-basis "
                       "'unitary'/'u2' blocks; cx_count/gate_count/fidelity then under-count the "
                       "2-qubit gates hidden inside 'unitary'")
    def test_create_physical_mappings_in_basis(self):
        r = _quiet(BaseOptimizer().create_physical_mappings, _sample_circuit(), CM4)
        assert set(r["circuit"].count_ops()) <= {"id", "rz", "sx", "x", "cx"}

    @pytest.mark.xfail(strict=True, reason="BUG: double transpile drops the first-pass layout; the "
                       "returned circuit is not equivalent to the input under its own layout "
                       "(Operator.from_circuit), i.e. the CLI's chosen 'mapped' circuit cannot be "
                       "related back to the logical circuit (base_optimizer.py:134-156)")
    def test_create_physical_mappings_equivalent(self):
        qc = _sample_circuit()
        r = _quiet(BaseOptimizer().create_physical_mappings, qc, CM4)
        assert Operator.from_circuit(r["circuit"]).equiv(Operator(qc))

    @pytest.mark.xfail(strict=True, reason="BUG: if the first (naive) transpile raises, "
                       "`physical_mappings` is unbound and UnboundLocalError escapes "
                       "(base_optimizer.py:100-189)")
    def test_create_physical_mappings_failure(self):
        big = QuantumCircuit(5)
        big.cx(0, 4)
        assert _quiet(BaseOptimizer().create_physical_mappings, big,
                      get_default_coupling_map("linear", 3)) is None

    @pytest.mark.xfail(strict=True, reason="BUG: apply_additional_optimizations calls "
                       "ToffoliPatternLibrary.optimize_circuit(..., require_verified=True) but that "
                       "method has no such keyword (it is a constructor argument); the TypeError is "
                       "swallowed, so --use_patterns is a silent no-op (base_optimizer.py:70-72)")
    def test_pattern_optimization_actually_runs(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 1, 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = BaseOptimizer().apply_additional_optimizations(qc, None)
        assert "Pattern optimization failed" not in buf.getvalue()
        assert Operator(out).equiv(Operator(qc))

    def test_apply_additional_optimizations_preserves_circuit(self):
        qc = _sample_circuit()
        ref = qc.copy()
        out = _quiet(BaseOptimizer().apply_additional_optimizations, qc, CM4)
        assert qc == ref and Operator(out).equiv(Operator(qc))

    def test_save_report_and_summary(self, tmp_path):
        b = BaseOptimizer()
        b.results = {"logical": {"depth": 20, "circuit": object()},
                     "naive_physical": {"depth": 50},
                     "optimized": {"depth": 15},
                     "mapped": {"depth": 40, "name": "Dense", "fidelity": 0.9, "circuit": object()},
                     "optimization_time": 1.5}
        f = tmp_path / "deep" / "r.json"
        assert _quiet(b.save_optimization_report, str(f)) is True
        rep = json.load(open(f))
        assert rep["logical_to_optimized_reduction"] == pytest.approx(25.0)
        assert rep["naive_to_physical_reduction"] == pytest.approx(20.0)
        assert rep["physical_mapping_method"] == "Dense"
        assert "circuit" not in rep["logical"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            b.print_optimization_summary(toffoli_gates=[1, 2], num_qubits=4, topology="linear",
                                         strategy="S", target_fidelity=0.9, min_fidelity=0.8,
                                         max_passes=2)
        s = buf.getvalue()
        assert "Logical-to-optimized depth reduction: 25.00%" in s
        assert "Naive-physical-to-final-physical depth reduction: 20.00%" in s
        with pytest.raises(NotImplementedError):
            b.display_summary_table()

    def test_report_unwritable_returns_false(self, tmp_path):
        b = BaseOptimizer()
        (tmp_path / "file").write_text("x")
        assert _quiet(b.save_optimization_report, str(tmp_path / "file" / "r.json")) is False

    def test_optimize_summary_table(self):
        o = Optimize(SimpleNamespace(debug=False))
        o.results = {"logical": {"depth": 10}, "naive_physical": {"depth": 0},
                     "optimized": {"depth": 5}, "mapped": {"depth": 7, "gate_count": 3,
                                                           "cx_count": 1, "fidelity": 0.5}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            o.display_summary_table()
        s = buf.getvalue()
        assert " 50.00%" in s and "Gate count: 3" in s and "Estimated fidelity: 0.5000" in s

    @pytest.mark.xfail(strict=True, reason="BUG: Optimize passes self.results (nested 'optimized': "
                       "{'depth': ..}) to visualize_optimization, which reads report['optimized_depth'] "
                       "-> KeyError swallowed inside visualize_optimization; --visualize writes no file "
                       "while generate_visualization returns True and prints 'Visualization saved' "
                       "(base_optimizer.py:467-470, visualization.py:559,595)")
    def test_visualize_with_cli_results(self, tmp_path):
        b = BaseOptimizer()
        b.results = {"logical": {"depth": 10}, "optimized": {"depth": 6}, "mapped": {"depth": 8}}
        base = str(tmp_path / "cli")
        assert _quiet(b.generate_visualization, _sample_circuit(), base) is True
        assert os.path.isfile(base + "_depth_reduction.png")

    def test_generate_visualization(self, tmp_path):
        b = BaseOptimizer()
        b.results = {"original_depth": 4, "optimized_depth": 2, "depth_reduction": 0.5}
        base = str(tmp_path / "vis")
        assert _quiet(b.generate_visualization, _sample_circuit(), base) is True
        assert os.path.getsize(base + "_depth_reduction.png") > 0
