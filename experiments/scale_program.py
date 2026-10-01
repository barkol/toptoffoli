"""Duzy zestaw w trybie programu (warunek U) i count-greedy, obok trybu podprocedury.
Wznawialny: kazdy obwod do scale_program_rows.jsonl.
- prog: pary zatwierdzone przez ScalableErrorBudgetSelector + gadzety na samotnych Toffolach z (U)
- greedy: gadzet wzgledno-fazowy na kazdym CCX, bez sprawdzania
Podprocedura: verify_scalable (wyczerpujaco <= 12 q, QCEC powyzej) -> rownowaznosc unitarna.
Program: certyfikat obserwacyjny na gestych unitarnych tylko dla n <= 12; powyzej 'analysis' (U)."""
import json, os, time
from _paths import EXPERIMENTS_DIR  # noqa: F401
import numpy as np
import scale_eval as SE
import large_benchmarks as LB
from qiskit.quantum_info import Operator
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.phase_observability import is_phase_unobservable, default_affected_qubits
from toffoli_optimizer.core.scalable_verification import verify_scalable
from toffoli_optimizer.core.subspace_check import certify_on_input_subspace
CK = os.path.join(os.path.dirname(__file__), "scale_program_rows.jsonl")
done = set()
if os.path.exists(CK):
    done = {json.loads(l)["name"] for l in open(CK)}
EM = SE.EM
for qc in LB.large_suite():
    if qc.name in done:
        print("pomijam", qc.name); continue
    sel = SE.ScalableErrorBudgetSelector(); sres = sel.select(qc)
    pair_idx = set()
    applied_sites = [s for s in find_relative_phase_safe_sites(qc)]
    # odtworz zbior indeksow par faktycznie zatwierdzonych (jak w select)
    rel = set()
    for site in applied_sites:
        cand = rel | {site.compute_idx, site.uncompute_idx}
        if verify_scalable(sres["exact"], sel._build(qc, cand), exhaustive_max_qubits=SE.EXHAUSTIVE_MAX_QUBITS).equivalent is True:
            rel = cand
    ccx_idx = [i for i, ins in enumerate(qc.data) if ins.operation.name.lower() in ("ccx", "mcx", "mcx_gray") and len(ins.qubits) == 3]
    U_idx = {i for i in ccx_idx if i not in rel and is_phase_unobservable(qc, i, default_affected_qubits(qc, i))}
    prog = sel._build(qc, rel | U_idx)
    greedy = sel._build(qc, set(ccx_idx))
    row = {"name": qc.name, "n": qc.num_qubits, "family": qc.family,
           "twoq_exact": EM.two_qubit_count(sres["exact"]), "infid_exact": EM.circuit_infidelity(sres["exact"]),
           "twoq_sub": sres["two_qubit_after"], "infid_sub": sres["infid_after"],
           "pairs": len(rel) // 2, "U_admitted": len(U_idx), "ccx": len(ccx_idx)}
    for tag, c in (("prog", prog), ("greedy", greedy)):
        row[f"twoq_{tag}"] = EM.two_qubit_count(c); row[f"infid_{tag}"] = EM.circuit_infidelity(c)
        t0 = time.time(); r = verify_scalable(sres["exact"], c, exhaustive_max_qubits=SE.EXHAUSTIVE_MAX_QUBITS)
        row[f"sub_ok_{tag}"] = r.equivalent; row[f"sub_method_{tag}"] = r.method; row[f"sub_t_{tag}"] = round(time.time() - t0, 3)
        if qc.num_qubits <= 12:
            Ue, Uc = Operator(sres["exact"]).data, Operator(c).data
            inputs = list(range(2 ** qc.num_qubits))   # jak dla zestawu 12 obwodow: wszystkie wejscia bazowe
            row[f"prog_ok_{tag}"] = bool(certify_on_input_subspace(Ue, Uc, inputs, "observational")[0])
        else:
            row[f"prog_ok_{tag}"] = None
    with open(CK, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str)[:300], flush=True)
print("koniec")
