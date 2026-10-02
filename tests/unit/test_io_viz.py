"""Unit tests for toffoli_optimizer.utils.io_utils and .visualization.

All files are written under pytest's tmp_path (cwd is also moved there).
Round-trips are checked with qiskit Operators on small circuits; plotting
functions are smoke-tested headless (Agg) with file-existence/size checks.
Confirmed bugs are ``xfail(strict=True)`` with minimal reproducers.
"""

import json
import os
import pickle
import subprocess
import sys
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from qiskit import QuantumCircuit, QuantumRegister  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402
from qiskit.visualization import circuit_drawer  # noqa: E402

from toffoli_optimizer.utils import io_utils as iou  # noqa: E402
from toffoli_optimizer.utils import visualization as viz  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
    plt.close("all")


def _qelib_circuit():
    """Circuit using only gates defined in the standard qelib1.inc."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.t(1)
    qc.tdg(3)
    qc.rz(0.37, 2)
    qc.x(0)
    qc.s(3)
    qc.sdg(1)
    return qc


def _nonempty(path):
    return path is not None and os.path.isfile(path) and os.path.getsize(path) > 0


def _write_json_net(path_base, instructions, num_qubits):
    with open(f"{path_base}.json", "w") as f:
        json.dump({"num_qubits": num_qubits, "instructions": instructions}, f)


# =========================================================================== #
# QASM I/O
# =========================================================================== #

class TestQasmIO:
    def test_roundtrip_qelib_gates_exact(self, tmp_path):
        qc = _qelib_circuit()
        f = tmp_path / "c.qasm"
        assert iou.save_circuit_to_qasm(qc, str(f)) is True
        back = iou.load_circuit_from_qasm(str(f))
        assert back is not None
        assert back.num_qubits == qc.num_qubits
        assert dict(back.count_ops()) == dict(qc.count_ops())
        assert Operator(back).equiv(Operator(qc))

    def test_roundtrip_with_measurement(self, tmp_path):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        f = tmp_path / "m.qasm"
        assert iou.save_circuit_to_qasm(qc, str(f))
        back = iou.load_circuit_from_qasm(str(f))
        assert back.num_clbits == 2 and back.count_ops()["measure"] == 2

    def test_input_not_mutated(self, tmp_path):
        qc = _qelib_circuit()
        ref = qc.copy()
        iou.save_circuit_to_qasm(qc, str(tmp_path / "x.qasm"))
        assert qc == ref

    @pytest.mark.parametrize("gate", ["sx", "p", "u", "swap", "mcx3"])
    @pytest.mark.xfail(strict=True, reason="BUG: load_circuit_from_qasm uses qasm2.loads without "
                       "LEGACY_CUSTOM_INSTRUCTIONS, so files written by save_circuit_to_qasm (qiskit's "
                       "dumps emits sx/p/u/swap/mcx definitions) cannot be read back -> returns None "
                       "(io_utils.py:101). Affects every transpiled circuit (basis contains sx).")
    def test_roundtrip_qiskit_gates(self, tmp_path, gate):
        qc = QuantumCircuit(4)
        {"sx": lambda: qc.sx(0), "p": lambda: qc.p(0.3, 1), "swap": lambda: qc.swap(0, 2),
         "mcx3": lambda: qc.mcx([0, 1, 2], 3), "u": lambda: qc.u(0.1, 0.2, 0.3, 1)}[gate]()
        f = tmp_path / "g.qasm"
        assert iou.save_circuit_to_qasm(qc, str(f))
        back = iou.load_circuit_from_qasm(str(f))
        assert back is not None and Operator(back).equiv(Operator(qc))

    def test_save_unexportable_returns_false(self, tmp_path):
        qc = QuantumCircuit(1)
        qc.rx(Parameter("a"), 0)
        f = tmp_path / "p.qasm"
        assert iou.save_circuit_to_qasm(qc, str(f)) is False

    def test_save_into_missing_dir_returns_false(self, tmp_path):
        assert iou.save_circuit_to_qasm(_qelib_circuit(), str(tmp_path / "no" / "dir" / "a.qasm")) is False

    def test_load_missing_returns_none(self, tmp_path):
        assert iou.load_circuit_from_qasm(str(tmp_path / "missing.qasm")) is None

    def test_load_malformed_returns_none_with_message(self, tmp_path, capsys):
        f = tmp_path / "bad.qasm"
        f.write_text("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nfoo q[0];\n")
        assert iou.load_circuit_from_qasm(str(f)) is None
        assert "Error loading circuit" in capsys.readouterr().out


class TestSaveCircuitSafely:
    def test_writes_loadable_qasm_inside_output_dir(self, tmp_path):
        out = tmp_path / "out"
        p = iou.save_circuit_safely(_qelib_circuit(), "net", str(out))
        assert p.endswith(".qasm") and os.path.dirname(p) == str(out)
        assert Operator(iou.load_circuit_from_qasm(p)).equiv(Operator(_qelib_circuit()))

    def test_none(self, tmp_path):
        assert iou.save_circuit_safely(None, "x", str(tmp_path)) is None

    def test_text_fallback_for_unexportable(self, tmp_path):
        qc = QuantumCircuit(2)
        qc.rx(Parameter("a"), 0)
        qc.cx(0, 1)
        p = iou.save_circuit_safely(qc, "par", str(tmp_path))
        assert p.endswith(".txt")
        txt = open(p).read()
        assert "Number of qubits: 2" in txt and "cx(q[0], q[1])" in txt and "rx(q[0], a)" in txt

    @pytest.mark.xfail(strict=True, reason="BUG: filenames use a 1-second timestamp; two saves of "
                       "different circuits under the same name within one second return the SAME "
                       "path and the first file is silently overwritten (io_utils.py:149-155; same "
                       "pattern in save_circuit_image/stats/text and visualization.py)")
    def test_two_saves_do_not_overwrite(self, tmp_path):
        a = QuantumCircuit(1)
        a.x(0)
        b = QuantumCircuit(1)
        b.h(0)
        # make both calls fall inside the same wall-clock second
        while time.time() % 1 > 0.5:
            time.sleep(0.05)
        pa = iou.save_circuit_safely(a, "same", str(tmp_path))
        pb = iou.save_circuit_safely(b, "same", str(tmp_path))
        assert pa != pb
        assert Operator(iou.load_circuit_from_qasm(pa)).equiv(Operator(a))


class TestTextAndStats:
    def test_text_image_matches_drawer(self, tmp_path):
        qc = _qelib_circuit()
        p = iou.save_circuit_image(qc, "t", output_dir=str(tmp_path / "img"), use_text_mode=True)
        assert _nonempty(p)
        assert open(p).read() == str(circuit_drawer(qc, output="text"))

    def test_png_image(self, tmp_path):
        p = iou.save_circuit_image(_qelib_circuit(), "i", output_dir=str(tmp_path), use_text_mode=False)
        assert _nonempty(p)
        assert open(p, "rb").read(8) == PNG_MAGIC

    @pytest.mark.parametrize("mod", [iou, viz], ids=["io_utils", "visualization"])
    @pytest.mark.xfail(strict=True, reason="BUG (cosmetic/leak): plt.figure() creates an empty figure, "
                       "circuit_drawer draws in a second one and plt.close() closes only the current "
                       "one -> one figure leaks per call (io_utils.py:310-319, visualization.py:100-109)")
    def test_png_image_no_figure_leak(self, tmp_path, mod):
        plt.close("all")
        mod.save_circuit_image(_qelib_circuit(), "i", output_dir=str(tmp_path), use_text_mode=False)
        assert plt.get_fignums() == []

    def test_none_circuit(self, tmp_path):
        assert iou.save_circuit_image(None, "n", output_dir=str(tmp_path)) is None
        assert iou.save_circuit_stats(None, "n", output_dir=str(tmp_path)) is None

    def test_stats_counts(self, tmp_path):
        qc = _qelib_circuit()
        p = iou.save_circuit_stats(qc, "s", output_dir=str(tmp_path))
        txt = open(p).read()
        assert f"Number of qubits: {qc.num_qubits}" in txt
        assert f"Depth: {qc.depth()}" in txt
        assert f"Total gates: {sum(qc.count_ops().values())}" in txt
        for g, n in qc.count_ops().items():
            assert f"  {g}: {n}\n" in txt

    @pytest.mark.parametrize("mod", [iou, viz], ids=["io_utils", "visualization"])
    @pytest.mark.xfail(strict=True, reason="BUG (cosmetic): instruction list uses Qubit.index, absent "
                       "in qiskit>=1, so every line falls to the bare-except and dumps the raw "
                       "CircuitInstruction repr (io_utils.py:384, visualization.py:232)")
    def test_stats_instruction_lines(self, tmp_path, mod):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        txt = open(mod.save_circuit_stats(qc, "s", output_dir=str(tmp_path))).read()
        assert "0: cx on qubits [0, 1]" in txt

    def test_benchmark_circuits_io(self, tmp_path):
        qc = _qelib_circuit()
        res = {"logical": {"circuit": qc}, "original": {"circuit": qc},
               "toffoli_optimizer": {"mapped": {"circuit": qc}},
               "qiskit_transpiler": {"circuit": qc}, "other": {}}
        paths = iou.save_benchmark_circuits(res, output_dir=str(tmp_path / "b"))
        assert set(paths) == {"logical", "original", "toffoli_optimizer", "qiskit_transpiler"}
        assert all(_nonempty(p) for p in paths.values())


# =========================================================================== #
# Toffoli network loading
# =========================================================================== #

class TestToffoliNetworkLoader:
    def test_extract_gates(self):
        a, b = QuantumRegister(3, "a"), QuantumRegister(3, "b")
        qc = QuantumCircuit(a, b)
        qc.ccx(a[0], a[1], b[2])          # global (0,1,5)
        qc.mcx([a[0], a[2], b[0]], b[1])  # global (0,2,3)->4
        qc.x(b[0])                        # global 3
        qc.cx(0, 1)                       # ignored by design (documented)
        qc.h(2)                           # ignored
        g = iou.ToffoliNetworkLoader.extract_toffoli_gates(qc)
        assert g == [([0, 1], 5), ([0, 2, 3], 4), ([], 3)]

    def test_extract_input_not_mutated(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        ref = qc.copy()
        iou.ToffoliNetworkLoader.extract_toffoli_gates(qc)
        assert qc == ref

    def test_json_all_formats(self, tmp_path):
        base = str(tmp_path / "n")
        _write_json_net(base, [
            {"name": "ccx", "qubits": [0, 1, 2]},
            {"name": "mcx", "control_qubits": [0, 1, 2], "target_qubit": 3},
            {"name": "mcx", "qubits": [1, 3]},
            {"name": "x", "qubits": [2]},
            {"name": "h", "qubits": [0]},
        ], 4)
        gates, n = iou.ToffoliNetworkLoader.load_json_toffoli_network(base)
        assert n == 4
        assert gates == [([0, 1], 2), ([0, 1, 2], 3), ([1], 3), ([], 2)]

    def test_json_missing_and_malformed(self, tmp_path):
        assert iou.ToffoliNetworkLoader.load_json_toffoli_network(str(tmp_path / "nope")) == (None, None)
        (tmp_path / "bad.json").write_text("{not json")
        assert iou.ToffoliNetworkLoader.load_json_toffoli_network(str(tmp_path / "bad")) == (None, None)
        (tmp_path / "nonq.json").write_text(json.dumps({"instructions": []}))
        assert iou.ToffoliNetworkLoader.load_json_toffoli_network(str(tmp_path / "nonq")) == (None, None)

    def test_load_from_json_with_metadata(self, tmp_path):
        base = str(tmp_path / "net")
        _write_json_net(base, [{"name": "ccx", "qubits": [0, 1, 7]}, {"name": "x", "qubits": [3]}], 8)
        with open(base + "_metadata.json", "w") as f:
            json.dump({"input_qubits": [0, 1], "output_qubits": [7]}, f)
        gates, outs, ins, n = iou.ToffoliNetworkLoader.load_toffoli_network(base)
        assert gates == [([0, 1], 7), ([], 3)]
        assert outs == [7] and ins == [0, 1] and n == 8

    def test_load_from_pickle(self, tmp_path):
        qc = QuantumCircuit(8)
        qc.ccx(0, 1, 2)
        qc.ccx(2, 3, 4)
        base = str(tmp_path / "pk")
        with open(base + ".pickle", "wb") as f:
            pickle.dump(qc, f)
        gates, outs, ins, n = iou.ToffoliNetworkLoader.load_toffoli_network(base)
        assert gates == [([0, 1], 2), ([2, 3], 4)] and n == 8
        assert all(0 <= q < 8 for q in outs + ins)

    def test_load_from_qasm(self, tmp_path):
        qc = QuantumCircuit(8)
        qc.ccx(0, 1, 2)
        qc.x(5)
        base = str(tmp_path / "qa")
        assert iou.save_circuit_to_qasm(qc, base + ".qasm")
        gates, _, _, n = iou.ToffoliNetworkLoader.load_toffoli_network(base)
        assert gates == [([0, 1], 2), ([], 5)] and n == 8

    def test_load_nothing(self, tmp_path):
        assert iou.ToffoliNetworkLoader.load_toffoli_network(str(tmp_path / "zzz")) is None

    def test_infer_io_qubits(self):
        gates = [([0, 1], 2), ([2, 3], 4), ([4, 5], 6)]
        outs, ins = iou.ToffoliNetworkLoader.infer_io_qubits(gates, 8)
        # 0,1,3,5 are pure controls; 6 is a pure target
        assert {0, 1, 3, 5} <= set(ins)
        assert 6 in outs
        assert all(0 <= q < 8 for q in outs + ins)

    @pytest.mark.xfail(strict=True, reason="BUG: returned num_qubits depends on the debug flag "
                       "(padded to 8 only when debug=True, io_utils.py:535-538)")
    def test_num_qubits_independent_of_debug(self, tmp_path):
        base = str(tmp_path / "small")
        _write_json_net(base, [{"name": "ccx", "qubits": [0, 1, 2]}], 3)
        a = iou.ToffoliNetworkLoader.load_toffoli_network(base, debug=False)
        b = iou.ToffoliNetworkLoader.load_toffoli_network(base, debug=True)
        assert a[3] == b[3]

    @pytest.mark.xfail(strict=True, reason="BUG: out-of-range control is 'validated' but the "
                       "filtered list is only adopted when a whole gate was dropped; the raw gate "
                       "is kept and infer_io_qubits crashes with IndexError (io_utils.py:541-563, 828)")
    def test_out_of_range_control_handled(self, tmp_path):
        base = str(tmp_path / "oor")
        _write_json_net(base, [{"name": "ccx", "qubits": [0, 5, 2]}], 3)
        res = iou.ToffoliNetworkLoader.load_toffoli_network(base)
        assert res is None or all(q < res[3] for cs, t in res[0] for q in list(cs) + [t])

    def test_define_loaded_network_ccx_mcx(self, tmp_path):
        base = str(tmp_path / "d")
        _write_json_net(base, [{"name": "ccx", "qubits": [0, 1, 2]},
                               {"name": "mcx", "qubits": [0, 1, 2, 3]}], 4)
        gates, outs, ins = iou.define_loaded_toffoli_network(base)
        assert gates == [([0, 1], 2), ([0, 1, 2], 3)]
        assert sorted(outs) == [2, 3] and ins == [0, 1, 2]

    @pytest.mark.xfail(strict=True, reason="BUG: define_loaded_toffoli_network (used by BOTH CLI "
                       "commands) ignores 'x' instructions although load_json_toffoli_network keeps "
                       "them -> X gates silently disappear from the network (io_utils.py:922-944)")
    def test_define_loaded_network_keeps_x(self, tmp_path):
        base = str(tmp_path / "dx")
        _write_json_net(base, [{"name": "x", "qubits": [0]}, {"name": "ccx", "qubits": [0, 1, 2]}], 3)
        gates, _, _ = iou.define_loaded_toffoli_network(base)
        assert ([], 0) in gates

    def test_define_loaded_network_missing_file_fallback(self, tmp_path, capsys):
        gates, outs, ins = iou.define_loaded_toffoli_network(str(tmp_path / "missing"))
        assert len(gates) == 6  # hard-coded default network (see report: CLI exit code issue)
        assert "Creating default Toffoli network" in capsys.readouterr().out

    def test_module_alias(self):
        assert iou.load_toffoli_network is iou.ToffoliNetworkLoader.load_toffoli_network


# =========================================================================== #
# visualization module
# =========================================================================== #

class TestVisualization:
    def test_save_circuit_text_small(self, tmp_path):
        qc = _qelib_circuit()
        p = viz.save_circuit_text(qc, "t", output_dir=str(tmp_path))
        assert _nonempty(p)
        assert open(p).read().strip() == str(circuit_drawer(qc, output="text")).strip()

    def test_save_circuit_text_large_uses_stats(self, tmp_path):
        qc = QuantumCircuit(60)
        for i in range(59):
            qc.cx(i, i + 1)
        p = viz.save_circuit_text(qc, "big", output_dir=str(tmp_path))
        txt = open(p).read()
        assert "Number of qubits: 60" in txt and "cx: 59" in txt
        assert "0: cx on qubits [0, 1]" in txt and "... and 39 more gates" in txt

    def test_save_circuit_image_text_and_png(self, tmp_path):
        qc = _qelib_circuit()
        pt = viz.save_circuit_image(qc, "a", output_dir=str(tmp_path), use_text_mode=True)
        pp = viz.save_circuit_image(qc, "b", output_dir=str(tmp_path), use_text_mode=False)
        assert _nonempty(pt) and _nonempty(pp)
        assert open(pp, "rb").read(8) == PNG_MAGIC
        assert viz.save_circuit_image(None, "c", output_dir=str(tmp_path)) is None

    def test_save_circuit_safely_qasm_and_fallback(self, tmp_path):
        p = viz.save_circuit_safely(_qelib_circuit(), "s", str(tmp_path))
        assert p.endswith(".qasm")
        assert Operator(iou.load_circuit_from_qasm(p)).equiv(Operator(_qelib_circuit()))
        qc = QuantumCircuit(1)
        qc.rx(Parameter("t"), 0)
        p2 = viz.save_circuit_safely(qc, "par", str(tmp_path / "fb"))
        assert p2.endswith(".txt") and "rx: 1" in open(p2).read()
        assert viz.save_circuit_safely(None, "n", str(tmp_path)) is None

    def test_save_benchmark_circuits(self, tmp_path):
        qc = _qelib_circuit()
        res = {"logical": {"circuit": qc}, "original": {"circuit": qc},
               "toffoli_optimizer": {"mapped": {"circuit": qc}, "optimized": {"circuit": qc},
                                     "circuit": qc},
               "qiskit_transpiler": {"circuit": qc}}
        paths = viz.save_benchmark_circuits(res, output_dir=str(tmp_path / "bc"))
        assert set(paths) == {"logical", "original", "toffoli_mapped", "toffoli_optimized",
                              "toffoli", "qiskit"}
        assert all(_nonempty(p) for p in paths.values())

    def test_plot_optimization_results(self, tmp_path):
        res = {"logical": {"depth": 20}, "naive_physical": {"depth": 40},
               "optimized": {"depth": 10}, "mapped": {"depth": 30}}
        plots = viz.plot_optimization_results(res, output_dir=str(tmp_path / "pl"))
        assert len(plots) == 2 and all(_nonempty(p) for p in plots)
        assert all(open(p, "rb").read(8) == PNG_MAGIC for p in plots)
        assert viz.plot_optimization_results({}, output_dir=str(tmp_path / "pl2")) == []
        assert viz.plot_optimization_results(None, output_dir=str(tmp_path / "pl3")) == []

    def test_visualize_optimization(self, tmp_path):
        qc = _qelib_circuit()
        base = str(tmp_path / "sub" / "rep")
        report = {"original_depth": 10, "optimized_depth": 6, "depth_reduction": 0.4,
                  "replacements": [object()], "note": "x"}
        viz.visualize_optimization(qc, report, filename=base)
        for suffix in ("_circuit.png", "_depth_reduction.png", "_report.json"):
            assert _nonempty(base + suffix)
        saved = json.load(open(base + "_report.json"))
        assert saved == {"original_depth": 10, "optimized_depth": 6, "depth_reduction": 0.4, "note": "x"}
        assert "replacements" in report  # input not mutated

    def test_memory_optimized_image_small_and_large(self, tmp_path):
        small = _qelib_circuit()
        p = viz.memory_optimized_save_circuit_image(small, "m", output_dir=str(tmp_path))
        assert p.endswith(".png") and _nonempty(p)
        big = QuantumCircuit(25)
        big.h(range(25))
        p2 = viz.memory_optimized_save_circuit_image(big, "big", output_dir=str(tmp_path))
        assert p2.endswith(".txt") and _nonempty(p2)
        assert viz.memory_optimized_save_circuit_image(None, "n", output_dir=str(tmp_path)) is None

    def test_stats_no_matplotlib(self, tmp_path):
        qc = QuantumCircuit(3)
        for _ in range(25):
            qc.cx(0, 2)
        p = viz.save_circuit_stats_no_matplotlib(qc, "nm", output_dir=str(tmp_path))
        txt = open(p).read()
        assert "Total gates: 25" in txt and "cx: 25" in txt and "Depth: 25" in txt
        assert "0: cx on qubits [0, 2]" in txt and "... and 5 more instructions" in txt

    def test_benchmark_with_memory_management(self, tmp_path):
        qc = _qelib_circuit()
        res = {"logical": {"circuit": qc}, "tof": {"mapped": {"circuit": qc}, "x": 1},
               "none": {"circuit": None}, "scalar": 3}
        paths = viz.save_benchmark_circuits_with_memory_management(res, output_dir=str(tmp_path / "mm"))
        assert set(paths) == {"logical_circuit", "tof_mapped_circuit"}
        assert all(_nonempty(p) for p in paths.values())

    def test_optimized_visualize_with_dir(self, tmp_path):
        base = str(tmp_path / "ovo" / "rep")
        out = viz.optimized_visualize_optimization(
            _qelib_circuit(), {"original_depth": 8, "optimized_depth": 4, "depth_reduction": 0.5,
                               "circuit": object()}, filename=base)
        assert set(out) == {"circuit", "report", "plot"}
        assert all(_nonempty(p) for p in out.values())
        assert "circuit" not in json.load(open(out["report"]))

    @pytest.mark.xfail(strict=True, reason="BUG: with a bare filename (no directory, also the "
                       "default 'optimization_report') output_dir is '' and os.makedirs('') fails, so "
                       "the circuit image is silently not saved (visualization.py:1093-1105, 662)")
    def test_optimized_visualize_bare_filename(self, tmp_path):
        out = viz.optimized_visualize_optimization(
            _qelib_circuit(), {"original_depth": 8, "optimized_depth": 4}, filename="rep")
        assert "circuit" in out

    @pytest.mark.xfail(strict=True, reason="BUG: memory_efficient_plot_comparison lowers the "
                       "PROCESS-WIDE RLIMIT_AS to 500 MB and never restores it; afterwards the caller "
                       "cannot allocate >500 MB (in a process already using more address space the "
                       "plot itself fails with std::bad_alloc) (visualization.py:932-937)")
    def test_memory_efficient_plot_comparison_isolated(self, tmp_path):
        # Run in a subprocess: the bug permanently cripples the interpreter it runs in.
        script = textwrap.dedent(f"""
            import resource, sys
            sys.path.insert(0, {REPO_ROOT!r})
            from toffoli_optimizer.utils import visualization as viz
            before = resource.getrlimit(resource.RLIMIT_AS)
            ok = viz.memory_efficient_plot_comparison(
                {{"a": {{"original_depth": 3, "optimized_depth": 2}}}}, {str(tmp_path / 'cmp.png')!r})
            after = resource.getrlimit(resource.RLIMIT_AS)
            print("RESULT", ok, before == after)
        """)
        env = dict(os.environ, MPLBACKEND="Agg")
        r = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                           capture_output=True, text=True, timeout=120)
        assert "RESULT True True" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
