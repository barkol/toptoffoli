# (B3): no circuit with at most 2 CNOTs implements CCX on a support with all four control values

Author id: P2. Date: 2026-10-03. Files: `DOWOD.md` (this), `check_b3.py` + `check_b3.out` (exact exhaustive cross-check, integer logic only).

## Statement

Let L = {00t0, 01t1, 10t2, 11t3} (strings written abt, any t0..t3 in {0,1}), and f = CCX, so
f(ab t) = (a, b, t xor ab). Let U be any 3-qubit circuit with k <= 2 CNOT gates (any pairs, any
directions) and arbitrary single-qubit gates. Then it is NOT true that U|x> = e^{iθ}|f(x)> for all x in L.

**Stronger form actually proved.** The proof uses only |<.|.>| (moduli of overlaps). Hence the
statement holds even if every string x is allowed its own phase, U|x> = e^{iθ_x}|f(x)>, and even if
the two-qubit parts are arbitrary two-qubit unitaries (not single CNOTs): no U of the form
"one 2-qubit unitary on a pair, then one 2-qubit unitary on a pair, plus single-qubit gates" works.
By monotonicity (L subset L' implies c*(L) <= c*(L')) this gives c*(L) >= 3 for every L with Π = {00,01,10,11}.

Throughout, "U|x> ∝ |y>" means equality up to a phase, and a *ray* is a unit vector up to phase.

## Structure of circuits with k <= 2 CNOTs

* k = 0 or 1, or k = 2 with both CNOTs on the same pair: some qubit q is touched by no CNOT.
  All single-qubit gates on q commute with everything acting on the other two qubits, so
  U = V_q ⊗ W (V a 1-qubit unitary on q, W a 2-qubit unitary on the rest).
* k = 2 on different pairs: the pairs share exactly one qubit j; call the qubit of the first CNOT
  pair other than j "i", the remaining qubit "k". Write the circuit as
  U = L3 C2 L2 C1 L1 with Lm = Lm_i ⊗ Lm_j ⊗ Lm_k products of single-qubit gates, C1 on {i,j}, C2 on {j,k}.
  Since L1_k, L2_k commute with C1, L1_i, L1_j, L2_i, L2_j and L3_i commutes with L3_j, L3_k, C2, L2_k:
  U = (I_i ⊗ V2_{jk}) (V1_{ij} ⊗ I_k),
  V1 = (L3_i L2_i ⊗ L2_j) C1 (L1_i ⊗ L1_j),  V2 = (L3_j ⊗ L3_k) C2 (I ⊗ L2_k L1_k).
  We call this the *chain (i, j, k)*. There are 6 chains = 6 orderings of (a, b, t).

## Lemma 1 (untouched qubit). If U = V_q ⊗ W, no L of type (B3) works.

Proof. For x in L, U|x> = V|x_q> ⊗ W|x_rest> ∝ |f_q(x)> ⊗ |f_rest(x)>, hence V|x_q> ∝ |f_q(x)> and
W|x_rest> ∝ |f_rest(x)> (a product vector equals a product of basis vectors only factor by factor, up to phases).
Unitarity: equal inputs give equal outputs, orthogonal inputs give orthogonal outputs.
* q = t. Strings 00t0 and 11t3: V|t0> ∝ |t0>, V|t3> ∝ |1 - t3>. If t0 = t3 then |t0> ∝ |1 - t0>, false.
  If t3 = 1 - t0 then V maps the orthogonal |t0>, |1 - t0> both to the ray |t0>, false.
* q = a. Strings 01t1, 11t3: W|1 t1> ∝ |1 t1>, W|1 t3> ∝ |1, 1 - t3>. If t1 = t3: |1 t1> ∝ |1, 1 - t1>, false.
  If t3 = 1 - t1: orthogonal inputs |1 t1>, |1, 1 - t1> both go to the ray |1 t1>, false.
* q = b. Same with strings 10t2, 11t3 (a <-> b). ∎

This settles k = 0, k = 1, and k = 2 on one pair. (It uses only the strings 00/11 or 01/11 or 10/11.)

## Lemma 2 (chain decomposition). Let U = (I_i ⊗ V2)(V1 ⊗ I_k) and U|x> ∝ |f(x)> for x in L. Then for every x in L:
(a) V1|x_i x_j> = |f_i(x)> ⊗ |η_x> for a unit vector η_x on qubit j;
(b) V2(|η_x> ⊗ |x_k>) ∝ |f_j(x) f_k(x)>.

Proof. (a) Qubit i is not acted on after V1, so the reduced state of qubit i in U|x> equals the reduced state
of qubit i in V1|x_i x_j>. U|x> ∝ |f(x)> has pure marginal |f_i(x)><f_i(x)| on i; a two-qubit pure state with a
pure marginal is a product, so V1|x_i x_j> = |f_i(x)> ⊗ |η_x>. (b) Substitute (a) into U|x>. ∎

## Lemma 3 (constraints from unitarity of V1, V2). For x ≠ y in L:
(V1-a) If (x_i,x_j) = (y_i,y_j): then f_i(x) = f_i(y) (else contradiction) and η_x = η_y, so η_x, η_y are the same ray. Label "=".
(V1-b) If (x_i,x_j) ≠ (y_i,y_j) and f_i(x) = f_i(y): 0 = <x_i x_j|y_i y_j> = <η_x|η_y>. Label "⊥".
(V2-a) If x_k ≠ y_k: the V2-inputs in (b) are orthogonal, so the outputs must be orthogonal:
       (f_j,f_k)(x) ≠ (f_j,f_k)(y), else contradiction.
(V2-b) If x_k = y_k: |<η_x|η_y>| = |<f_j f_k (x)|f_j f_k (y)>|, i.e. "=" if (f_j,f_k)(x) = (f_j,f_k)(y), "⊥" otherwise.
(No constraint arises from V1 when f_i(x) ≠ f_i(y) with different inputs.) Immediate from unitarity. ∎

## Lemma 4 (parity of a cycle of ray relations in C^2).
Let η_1, ..., η_m be unit vectors in C^2 and suppose consecutive pairs (cyclically) are each either "=" (|<η_p|η_q>| = 1)
or "⊥" (<η_p|η_q> = 0). Then the number of "⊥" relations is even.

Proof. In C^2 each ray r has exactly one orthogonal ray r^⊥, and (r^⊥)^⊥ = r. "=" means equal rays, "⊥" means η_q's ray = (η_p's ray)^⊥.
Going once around the cycle, the ray of η_1 is mapped to itself by an even number of ⊥'s and to its orthogonal ray
by an odd number; r = r^⊥ is impossible. ∎

So: build the signed graph on the 4 strings with the labels of Lemma 3; a direct contradiction (V1-a or V2-a) or an
odd cycle excludes the chain.

## Reduction of the 6 chains to 2

* **a <-> b symmetry.** Swapping qubits a and b commutes with CCX and maps a (B3)-support to a (B3)-support,
  and chains (a,t,b) <-> (b,t,a), (b,a,t) <-> (a,b,t), (t,a,b) <-> (t,b,a).
* **Inversion.** If U|x> = e^{iθ}|f(x)> on L then U†|y> = e^{-iθ}|f(y)> on L' = f(L) (f is an involution), and
  L' is again a (B3)-support (f preserves control values). If U is the chain (i,j,k), U† = (V1† ⊗ I)(I ⊗ V2†) is
  the chain (k,j,i). Thus (t,a,b) is excluded for all L once (b,a,t) is excluded for all L.
  (Per-string phases: e^{iθ_x} becomes e^{-iθ_x}; the stronger form survives.)

It remains to exclude chain (a,t,b) [middle qubit = target] and chain (b,a,t) [middle qubit = a control]
for all 16 target patterns. Strings are named by their control value: 00, 01, 10, 11.

## Case A: chain (a, t, b): i = a, j = t, k = b. All 16 patterns.

Here f_i = a, the V1 input is (a, t_x), the V2 input is (η_x, b), V2 output is (t_out, b) with t_out(ab) = t_ab xor ab.
* Pair 00-01 (same a = 0, so same f_i): if t0 = t1, same V1 input: "=" (V1-a); if t0 ≠ t1: "⊥" (V1-b). Label parity t0 ⊕ t1.
* Pair 10-11: likewise, parity t2 ⊕ t3.
* Pair 00-10 (same b = 0, V2-b): "=" iff t_out equal, outputs t0 and t2: parity t0 ⊕ t2.
* Pair 01-11 (same b = 1, V2-b): outputs t1 and 1 ⊕ t3: parity t1 ⊕ t3 ⊕ 1.
Cycle 00-01-11-10-00 has parity (t0⊕t1)⊕(t1⊕t3⊕1)⊕(t3⊕t2)⊕(t2⊕t0) = 1: odd. Contradiction by Lemma 4, for every t0..t3. ∎
(The odd parity is exactly the statement that ab is not of the form c_b ⊕ d_a: AND is not separable.)

## Case B: chain (b, a, t): i = b, j = a, k = t. All 16 patterns.

V1 input (b, a): the four strings have four different V1 inputs. f_i = b.
V2 input (η_x, t_x); V2 output (a, t_out), t_out(ab) = t_ab xor ab.
1. Pair 10-11: t-inputs t2, t3; outputs (1, t2), (1, 1 ⊕ t3). If t2 ≠ t3 then the outputs coincide while the
   k-inputs differ: contradiction (V2-a). Hence **t2 = t3 =: u**, and then (V2-b) gives 10-11 "⊥".
2. Pairs 00-10 and 01-11 have equal b (= f_i) and different V1 inputs: "⊥" (V1-b).
3. Pair 00-01: outputs (0,t0), (0,t1). If t0 = t1: same k-input, same output: "=" (V2-b).
4. Pair 00-11: outputs differ in a; if t0 = t3 (same k-input) then "⊥" (V2-b).
5. Pair 01-10: outputs differ in a; if t1 = t2 then "⊥" (V2-b).
Sub-cases (u from step 1):
* t0 = t1: cycle 00-10-11-01-00 with labels ⊥, ⊥, ⊥, = : three "⊥", odd.
* t0 ≠ t1 and t0 = u: then t0 = t3, triangle 00-10-11-00 with labels ⊥, ⊥, ⊥: odd.
* t0 ≠ t1 and t1 = u: then t1 = t2, triangle 01-10-11-01 with labels ⊥, ⊥, ⊥: odd.
These exhaust all patterns (t2 ≠ t3 excluded in step 1; with t2 = t3 = u and t0 ≠ t1 one of t0, t1 equals u).
Contradiction by Lemma 4 in every case. ∎

## Conclusion

Every circuit with at most 2 CNOTs is either of untouched-qubit type (excluded by Lemma 1) or one of the 6 chains
(excluded by Cases A, B plus a<->b symmetry and inversion). Hence c*(L) >= 3 for all L with Π = {00,01,10,11}.
Combined with the constructive 3-CNOT circuits (class 3), (B3) is proven. Also, since only moduli of overlaps were used,
no ≤2-CNOT circuit works even with string-dependent phases, and even with arbitrary 2-qubit unitaries in place of the CNOTs.

## Exact computer cross-check (not needed for the proof, but makes refereeing easy)

`check_b3.py` (pure integer/boolean logic: Z2-labelled union-find for Lemma 4, direct checks for Lemma 1 and V1-a/V2-a)
tests the necessary conditions of Lemmas 1-3 for **all 6 chains directly (no symmetry reduction) and all 3 choices of untouched qubit, on all 16 (B3) supports**. Output `check_b3.out`: every one of the 16 supports admits no structure (0/16).
Positive controls in the same run: for Π = {01,10,11} (c* = 2 claimed) all 8 supports admit chain (a,t,b), consistent
with the known circuit X_t CX(a,t) CX(b,t) (which is that chain), and no untouched-qubit structure (consistent with (B2)
for k <= 1); for Π = {10,11} the untouched-a structure is admitted, as it must be. So the checker is not vacuous.

## Relation to Song & Klappenecker (quant-ph/0312225)

Their Lemma 7 excludes 2-CNOT chains for the simplified Toffoli by noting that, with the top qubit set to |0>, the circuit
can still entangle the other two qubits — an argument about the action on the WHOLE space, which does not apply here
(our U is constrained only on span(L)). The argument above replaces it by Lemma 2 (pure marginal of the qubit that leaves
the circuit after the first two-qubit gate) and the C^2 parity Lemma 4, which need only the four strings of L.

## What is proven / open
* Proven: (B3) in full, all 16 target patterns, all circuits with 0, 1, 2 CNOTs; in the stronger per-string-phase form.
* Open within this task: nothing. (B1, B2, B4 not addressed here; Lemma 1 also yields (B1)-type statements for k = 0.)
