# Static per-Toffoli prediction of the 2q count of ErrorBudgetSelector.select (2026-10-02)

Code: ~/git/toptoffoli-fix @0dedc14 (read only). Drivers: `predykcja.py` (per-circuit worker, jsonl checkpoint, resumable), `analiza.py`. Data: `wyniki.jsonl` (per Toffoli: S(g), c_full, c_pass, C/W membership, rejected-site flag, chosen action), `analiza.json`.

Labels: string abt, x = a+2b+4t. S(g) is computed exactly as in select(): `local_reachable(..., pinned)`, else projected `reachable_overapprox` (491 Toffolis exact path; 37 over-approximation: phase_sensitive_oracle 1 (non-classical prefix), array_mult_6b 36 (24 free bits > 22)).

## What the pass actually tries (decomposition_selector.select, defaults epsilon=0, semantics=subroutine)

Gadget costs verified by building them: exact CCX 6 CX; relphase (Margolus) 3, phase -1 on abt=100; relphase_m 3, phase -1 on abt=101; control_drop keep=() 0; keep=(a,) or (b,) 1.

1. (C) structural sites from context_analysis, each verified by ExactEquivalenceVerifier.verify(exact, built). That verifier refuses non-classical circuits above 12 qubits, so for n>12 **every (C) site is rejected**.
2. (W) window pairs (mirror_pair_candidates, laminar, dense window check <=12 window qubits).
3. (R) relphase then relphase_m, default orientation only, on every CCX not in `locked` = applied actions + W pairs + **all** structural sites (also the rejected ones).
4. Control drop keep=(),(a,),(b,) with exact (R) check (epsilon=0) and budget margin>0, on every CCX not in paired_idx (all sites incl. rejected + W pairs); overrides an (R) relphase when cheaper. Drop-to-identity IS tried.
5. Final certificate: exact verifier if nothing admitted and no pins; dense subspace certificate <=12 qubits; QCEC above. Fail-closed to all-exact.

**Not tried by select():** the swapped orientation (phase on abt=010 / 011). orientation.py offers it only as a separate calibration-aware post-pass. Hence the pass realises c=3 only when S lacks 100 or 101 (`c_pass`), not when it lacks 010 or 011 (`c_full`).

## Match rate (2q count, per circuit)

| suite | circuits | pred(full table)==actual | pred(pass options)==actual | refined model==actual | skipped |
|---|---|---|---|---|---|
| primary | 12 | 11 | 12 | 12 | 0 |
| revlib | 50 | 40 | 50 | 50 | 5 |
| scale | 20 | 13 | 13 | 20 | 0 |

Refined model = cost 3 for C/W members; 6 for rejected (C) sites not in a W pair; min(6,c_pass) otherwise; whole circuit = all-exact if the final certificate fails. It agrees on 528/528 Toffolis and 82/82 circuits. Skipped (selection >300 s): urf1_149, urf2_152, urf3_155, urf4_187, urf5_158.

Sum of 2q: primary actual 194 / pred_full 191 / pred_pass 194; RevLib 965 / 914 / 965; scale 1536 / 1422 / 1422.

## Mismatches (Toffoli level)

### A. Orientation swap not tried (18 Toffolis, 11 circuits; full table predicts 3, pass gives 6)

S contains both 100 and 101 but lacks 010 or 011, so only the swapped gadget would be exact on S. For every such circuit I built the selection with the swapped gadget added and ran the dense input-subspace certificate: all 11 certify, with the 2q count equal to pred_full.

| circuit | idx | S (abt) | swapped gadget | 2q actual -> with swap |
|---|---|---|---|---|
| controlled_adder_2b | 3 | 000 100 001 101 011 111 | relphase+swap | 30 -> 27 (certified=True) |
| 4mod5-bdd_287.real | 8 | 000 100 010 110 001 101 | relphase_m+swap | 18 -> 15 (certified=True) |
| 4mod5-v0_19.real | 4 | 000 100 010 110 101 111 | relphase_m+swap | 11 -> 8 (certified=True) |
| rd32-v0_66.real | 2 | 000 100 010 110 001 101 | relphase_m+swap | 11 -> 8 (certified=True) |
| rd32-v1_68.real | 3 | 000 100 001 101 011 111 | relphase+swap | 11 -> 8 (certified=True) |
| rd32_272.real | 1 | 000 100 010 110 101 111 | relphase_m+swap | 18 -> 15 (certified=True) |
| rd53_138.real | 9 | 000 100 010 110 001 101 | relphase_m+swap | 37 -> 34 (certified=True) |
| mini_alu_305.real | 8 | 000 100 010 110 001 101 | relphase_m+swap | 49 -> 40 (certified=True) |
| mini_alu_305.real | 15 | 100 010 110 001 101 | relphase_m+swap | 49 -> 40 (certified=True) |
| mini_alu_305.real | 20 | 000 100 001 101 011 111 | relphase+swap | 49 -> 40 (certified=True) |
| sys6-v0_111.real | 10 | 000 100 010 110 001 101 | relphase_m+swap | 61 -> 55 (certified=True) |
| sys6-v0_111.real | 15 | 000 100 010 110 001 101 | relphase_m+swap | 61 -> 55 (certified=True) |
| rd73_140.real | 9 | 000 100 010 110 001 101 | relphase_m+swap | 69 -> 60 (certified=True) |
| rd73_140.real | 13 | 000 100 010 110 001 101 | relphase_m+swap | 69 -> 60 (certified=True) |
| rd73_140.real | 17 | 000 100 010 110 001 101 | relphase_m+swap | 69 -> 60 (certified=True) |
| sym9_146.real | 9 | 000 100 010 110 001 101 | relphase_m+swap | 98 -> 89 (certified=True) |
| sym9_146.real | 13 | 000 100 010 110 001 101 | relphase_m+swap | 98 -> 89 (certified=True) |
| sym9_146.real | 17 | 000 100 010 110 001 101 | relphase_m+swap | 98 -> 89 (certified=True) |

### B. Rejected structural (C) site locked as exact (26 Toffolis, 6 scale circuits, n=13-22)

The (C) verifier refuses n>12 ('non-classical circuit too large for exact unitary check'), the site is rejected, and the rejected pair stays in `locked`/`paired_idx`, so neither (R) nor control drop looks at it. S lacks 101 (c_pass=3, relphase_m would be exact on S). The stored QCEC-gated selector (paper/data_v12/scale_rows.jsonl) applies these sites and reaches exactly pred_pass in all 6 affected circuits.

| circuit | n | Toffolis | stock actual | pred | stored QCEC selector |
|---|---|---|---|---|---|
| cla_adder_4b | 13 | 2 | 49 | 43 | 43 |
| grover_oracle_8c | 16 | 4 | 55 | 43 | 43 |
| cla_adder_5b | 16 | 2 | 61 | 55 | 55 |
| grover_oracle_11c | 22 | 10 | 91 | 61 | 61 |
| cla_adder_7b | 22 | 4 | 91 | 79 | 79 |
| half_uncomputed_8c | 17 | 4 | 61 | 49 | 49 |

### C. Final certificate fallback (12 Toffolis, ripple_adder_10b, n=22)

Selection would give 125 (= prediction; = stored QCEC selector), but QCEC returned `probably_equivalent` (undecided); select() fails closed and returns all-exact (161). Five other scale circuits (ripple_live_carry_6b/8b, array_mult_4b/5b/6b) also fall back (exact verifier refuses >12 qubits), but nothing had been substituted there (all c=6), so their counts match.

No other mismatch types (no budget-margin rejections, no QCEC false negatives outside C).

## Histograms (c_full / c_pass over all Toffolis)

| suite | Toffolis | c_full 0/1/3/6 | c_pass 0/1/3/6 | in C | in W | c_full=6 in C/W | actions |
|---|---|---|---|---|---|---|---|
| primary | 38 | 0/0/26/12 | 0/0/22/16 | 20 | 8 | 4 | {"('relphase',)": 28, 'None': 9, "('relphase_m',)": 1} |
| revlib | 174 | 0/2/92/80 | 0/2/75/97 | 0 | 8 | 7 | {'None': 90, "('relphase',)": 24, "('relphase_m',)": 58, "('control_drop', (2,))": 1, "('control_drop', (0,))": 1} |
| scale | 316 | 0/0/176/140 | 0/0/176/140 | 12 | 102 | 28 | {'None': 150, "('relphase',)": 128, "('relphase_m',)": 38} |

No Toffoli has c=0 (drop-to-identity never applicable). c=1 occurs twice (RevLib), both taken as 1-CX control drops.

## c=6 check

c_full=6 Toffolis: primary 12, RevLib 80, scale 140 (232 total); 39 of them are C/W members and get 3 by phase cancellation, as the rule allows. **None of the 193 c=6 Toffolis outside C/W was substituted.** No contradiction of the criterion.

## Verdict

Outside C/W pairs, outside rejected (C) sites and when the final certificate passes, select() substitutes a Toffoli iff c_pass(S)<6, and then with cost exactly c_pass (528/528 Toffolis). With the full table (c_full) the 'if' direction fails only for the orientation case (A): the pass does not try the swapped gadget, although it would certify. Two implementation effects lie outside the static criterion: (B) rejected (C) sites above 12 qubits are locked out of (R), and (C) the fail-closed fallback on an undecided QCEC.

