#!/usr/bin/env python3
"""Warstwa obliczeniowa figur: czyta pliki wynikow eksperymentow i zapisuje
figures/data/make_figures.npz. Warstwa rysowania (make_figures.py) tylko wczytuje.

Zrodla (przebieg toptoffoli v1.1-revision, semantyka podprocedury):
  RES/safety_results.md, RES/ablation_b_results.md, RES/baseline_results.md,
  RES/fixtures/sensitivity_data.csv, RES/scale_rows.jsonl, RES/scale_results.md
Uruchomienie:  python figures_data.py RES   (domyslnie ../data/rerun_v14/experiments)
"""
import csv, json, os, re, sys
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rerun_v14", "experiments")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
d = {}

# --- Fig. safety: liczba cicho uszkodzonych obwodow (semantyka podprocedury) ----
s = open(os.path.join(RES, "safety_results.md")).read()
m = re.search(r"(?i)naive substitution silently corrupts (\d+) of (\d+) circuits; the verification gate yields (\d+) errors", s)
if not m:
    m2 = re.findall(r"\|\s*`?(\w+)`?\s*\|.*?\|\s*(ok|WRONG|\*\*NO\*\*|yes)\s*\|\s*(ok|WRONG)\s*\|", s)
    raise SystemExit("nie znaleziono naglowka w safety_results.md")
naive_bad, n_circ, gated_bad = int(m.group(1)), int(m.group(2)), int(m.group(3))
a = open(os.path.join(RES, "ablation_b_results.md")).read()
nogate_bad = int(re.search(r"introduces (\d+) of (\d+) silent errors", a).group(1))
d["safety_bad"] = np.array([naive_bad, nogate_bad, gated_bad])
d["safety_n"] = np.array(n_circ)

# --- Fig. budget: sumy po zestawie Q2 --------------------------------------------
b = open(os.path.join(RES, "baseline_results.md")).read()
names, infid, count = [], [], []
for label, key in (("Qiskit opt-3", "Qiskit opt-3"), ("tket", "tket (FullPeephole)"),
                   ("exact-only", "Exact-only (ours, all-exact)"), ("ours", "**Ours (gated)**")):
    row = re.search(re.escape("| " + key + " |") + r"\s*([\d.]+)\s*\|\s*(\d+)\s*\|", b)
    names.append(label); infid.append(float(row.group(1))); count.append(int(row.group(2)))
d["budget_names"] = np.array(names); d["budget_infid"] = np.array(infid); d["budget_count"] = np.array(count)

# --- Fig. device: agregat podzbioru (iloczyn wiernosci, jak sensitivity_sweep.py) --
rows = [r for r in csv.DictReader(open(os.path.join(RES, "fixtures", "sensitivity_data.csv")))
        if r["mode"] == "device"]
order = ["Quantinuum H2", "IBM Heron r2 (ibm_marrakesh)", "Google Willow (105q)", "IonQ Forte", "IonQ Aria"]
ex, ours, ratio, tq_b, tq_a = [], [], [], [], []
for dev in order:
    rr = [r for r in rows if r["device"] == dev]
    if not rr:
        raise SystemExit(f"brak urzadzenia {dev} w CSV")
    fb = np.prod([1 - float(r["infidelity_before"]) for r in rr])
    fa = np.prod([1 - float(r["infidelity_after"]) for r in rr])
    ex.append(1 - fb); ours.append(1 - fa)
    ratio.append(float(rr[0]["p2q"]) / float(rr[0]["p1q"]))
    tq_b.append(sum(int(r["two_qubit_before"]) for r in rr)); tq_a.append(sum(int(r["two_qubit_after"]) for r in rr))
d["dev_names"] = np.array(["Quantinuum H2", "IBM Heron r2", "Google Willow", "IonQ Forte", "IonQ Aria"])
d["dev_ex"] = np.array(ex); d["dev_ours"] = np.array(ours); d["dev_ratio"] = np.array(ratio)
d["dev_red"] = 100 * (np.array(ex) - np.array(ours)) / np.array(ex)
d["dev_2q"] = np.array([tq_b, tq_a])

# --- Fig. scale: rodziny i czasy certyfikacji (jesli przebieg skonczony) ---------
ck = os.path.join(RES, "scale_rows.jsonl")
if os.path.exists(ck):
    sr = [json.loads(l) for l in open(ck)]
    fam_order = ["grover/cu", "grover/mixed", "adder/cla", "adder/cu", "adder/live", "modular/cu", "multiplier/live"]
    fams = sorted({r["family"] for r in sr}, key=lambda f: fam_order.index(f) if f in fam_order else 99)
    d["fam_names"] = np.array(fams)
    d["fam_red"] = np.array([np.mean([r["red_2q"] for r in sr if r["family"] == f]) for f in fams])
    d["fam_n"] = np.array([sum(r["family"] == f for r in sr) for f in fams])
    q = [r for r in sr if r["cert_method"].startswith("qcec")]
    d["qcec_n"] = np.array([r["n"] for r in q]); d["qcec_t"] = np.array([r["cert_time"] for r in q])
    d["qcec_ok"] = np.array([bool(r["cert_equiv"]) for r in q])
    t = open(os.path.join(RES, "scale_results.md")).read()
    cr = re.findall(r"^\|\s*(\d+)\s*\|\s*([\d.]+|—)\s*\|\s*\S+\s*\|\s*([\d.]+)\s*\|", t, re.M)
    d["cross_n"] = np.array([int(a) for a, _, _ in cr])
    d["cross_exh"] = np.array([float(b) if b != "—" else np.nan for _, b, _ in cr])
    d["cross_qcec"] = np.array([float(c) for _, _, c in cr])
    d["scale_rows"] = np.array(len(sr))
    agg_b = sum(r["twoq_exact"] for r in sr); agg_a = sum(r["twoq_ours"] for r in sr)
    ib = sum(r["infid_exact"] for r in sr); ia = sum(r["infid_ours"] for r in sr)
    d["scale_agg"] = np.array([agg_b, agg_a, 100 * (agg_b - agg_a) / agg_b, 100 * (ib - ia) / ib])
np.savez(os.path.join(OUT, "make_figures.npz"), **d)
print("zapisano", os.path.join(OUT, "make_figures.npz"), sorted(d))
