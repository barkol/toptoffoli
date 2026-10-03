# Proofs of the lower bounds of Proposition 1 (cost of a Toffoli on its reachable support)

Problem statement: `PROBLEM.md`. By monotonicity of c* in L, each bound is proven on the smallest supports of its class.

| bound | proof | method | independent check |
|---|---|---|---|
| (B1) c* ≥ 1, (B2) c* ≥ 2, and c* ≥ 2 for classes 3, 4, 6 | `P1/DOWOD.md` | an untouched qubit splits U = V⊗W; injectivity of the bit maps fails on two strings | `P1/enumeracja.py` (exact enumeration of all 255 supports); referee `weryfikacja/V_P1` |
| (B3) c* ≥ 3 | `P2/DOWOD.md` | two-CNOT chain i–j–k: qubit i leaves after the first gate; ray relations "=" / "⊥" on qubit j; odd cycle in C² | `P2/check_b3.py` (integer logic, all 6 chains, 16 supports, positive controls); second proof `P3/DOWOD.md` (reduction to the middle qubit, `P3/sprawdz_B3.py`); referee `weryfikacja/V_P2` |
| (B4) c* ≥ 4 | `P4/DOWOD.md` | degree split: target of degree ≤ 1 (spectrum across the cut), control of degree ≤ 1 (unique candidate + Makhlin invariants), triangle topology (direct computation) | `P4/tozsamosci.py`, `P4/lemat_11.py` (exact sympy); referee `weryfikacja/V_P4a` |
| c* = 6 for L = {0,1}^3 | Shende & Markov, QIC 9, 461 (2009) | | |

The bounds 1–3 hold even with a separate phase for each string (only moduli of overlaps are used); (B3) also holds with arbitrary two-qubit gates in place of the two CNOTs.
All referees returned CORRECT (cosmetic fixes applied in this copy). `LITERATURA.md`: literature search; no earlier statement of the subspace-restricted costs was found.
Numerical corroboration: `../koszt.jsonl` (minimal loss ≥ 0.067 below c*, zero at c*).
