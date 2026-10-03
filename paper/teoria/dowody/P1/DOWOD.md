# P1: proof of (B1), (B2), and c* ≥ 2 on classes 3, 4, 6

Notation as in `../PROBLEM.md`. Strings are `abt`; f : {0,1}^3 → {0,1}^3 is the
classical Toffoli, f(a,b,t) = (a, b, t ⊕ ab), so CCX|x> = |f(x)>. Π = {ab : abt ∈ L}.
A circuit is *valid for L* if U|x> = e^{iθ}|f(x)> for all x ∈ L (one common θ).

## Statements proved

* **Theorem 1 (B1, and an exact characterisation of c* = 0).**
  c*(L) = 0  ⇔  11 ∉ Π or Π = {11}. In particular c*(L) ≥ 1 whenever 11 ∈ Π and Π ≠ {11}.
* **Theorem 2 (B2).** If Π ⊇ {01, 10, 11} then no circuit with 0 or 1 CNOT is valid for L,
  i.e. c*(L) ≥ 2. This covers class 2 (Π = {01,10,11}) and, directly, every L in classes 3, 4, 6
  (Π = {00,01,10,11}).
* **Corollary 3.** c*(L) ≥ 2 for every L of classes 3, 4, 6 (also via monotonicity from Theorem 2).

All bounds hold in a stronger form: the common phase is never used, so they hold even if every
x ∈ L is allowed its own phase. Since k = 0 and k = 1 are both excluded directly, the padding
remark (k → k+2) plays no role.

## Lemmas

**Lemma 0 (monotonicity).** L ⊆ L' ⇒ c*(L) ≤ c*(L').
*Proof.* A circuit valid for L' satisfies the defining equations for every x ∈ L' ⊇ L. ∎

**Lemma 1 (circuits with ≤ 1 CNOT factorise).** Let U be a circuit with at most one CNOT.
Then for some qubit k ∈ {a, b, t} (for k = 0 CNOTs: for *every* k) we have
U = V ⊗ W, where W is a unitary on qubit k and V a unitary on the other two qubits
(tensor factors ordered by qubit label, not by position).
*Proof.* With 0 CNOTs, U is a product of layers A_i ⊗ B_i ⊗ C_i, hence U = A ⊗ B ⊗ C,
which is of the form V ⊗ W for each choice of k. With one CNOT on the pair {i, j}, let k be the
third qubit. Every gate either acts only on k or only on {i, j}; gates on disjoint qubit sets
commute, so U = (product of the gates on {i,j}) ⊗ (product of the gates on k) = V ⊗ W. ∎

**Lemma 2 (product vectors).** Let u, r ∈ H_1 and w, q ∈ H_2 with u, w ≠ 0, r, q unit vectors,
and u ⊗ w = λ r ⊗ q with λ ≠ 0. Then u ∝ r and w ∝ q (nonzero multiples).
*Proof.* Apply (1 ⊗ <q|): u <q|w> = λ r. The right side is nonzero, so <q|w> ≠ 0 and
u = (λ/<q|w>) r. Symmetrically w ∝ q. ∎

**Lemma 3 (factorisation obstruction).** Suppose U = V ⊗ W (W on qubit k) is valid for L, even
with x-dependent phases. Write x = (x_k, x_r) with x_r the two remaining bits. Then both relations

  R_k = { (x_k, f(x)_k) : x ∈ L }  ⊆ {0,1} × {0,1},
  R_r = { (x_r, f(x)_r) : x ∈ L }  ⊆ {0,1}^2 × {0,1}^2

are graphs of *partial injections*: (functional) equal first entries force equal second
entries, and (injective) equal second entries force equal first entries.
*Proof.* For x ∈ L, V|x_r> ⊗ W|x_k> = U|x> = e^{iθ_x} |f(x)_r> ⊗ |f(x)_k>. Both V|x_r> and
W|x_k> are nonzero (unit vectors), so by Lemma 2

  (∗)  V|x_r> ∝ |f(x)_r>,   W|x_k> ∝ |f(x)_k>   for every x ∈ L.

Functional: if x_r = y_r for x, y ∈ L, then V|x_r> is a nonzero multiple of both basis vectors
|f(x)_r> and |f(y)_r>; distinct basis vectors are orthogonal, so f(x)_r = f(y)_r.
Injective: if x_r ≠ y_r then |x_r> ⟂ |y_r>, so V|x_r> ⟂ V|y_r> (V unitary); if f(x)_r = f(y)_r
these would be nonzero multiples of the same vector, not orthogonal, contradiction.
The same two arguments with W give the claim for R_k. ∎

Remark (warning (i) of the task). Lemma 2 makes the basis-to-basis property automatic: no
separate argument with two strings sharing the k-bit is needed — (∗) holds for each x on its own.
Remark (sufficiency, not needed for the lower bounds). If R_k and R_r are partial injections,
extend them to permutations σ_k of {0,1}, σ_r of {0,1}^2; the permutation matrices
W = P_{σ_k}, V = P_{σ_r} give U|x> = |f(x)> exactly (θ = 0). Whether such a V needs 0, 1, or
more CNOTs is a separate question; Lemma 3 is only used as an obstruction.

**Lemma 4 (which qubits cannot be split off).**
 (T) If 11 ∈ Π and Π ≠ {11}, R_t is not a partial injection (qubit t cannot be split off).
 (A) If {01, 11} ⊆ Π, R_r for k = a (remaining qubits b, t) is not a partial injection.
 (B) If {10, 11} ⊆ Π, R_r for k = b (remaining qubits a, t) is not a partial injection.
*Proof.*
 (T) Take 11t ∈ L and uvs ∈ L with uv ≠ 11. Then (t, t̄) ∈ R_t and (s, s) ∈ R_t.
   If s = t: t ↦ t̄ and t ↦ t, not functional. If s = t̄: t ↦ t̄ and t̄ ↦ t̄, not injective.
 (A) Take 01s ∈ L and 11s' ∈ L. Pairs in R_r (bits b t): (1s, 1s) and (1s', 1s̄').
   If s' = s: 1s ↦ 1s and 1s ↦ 1s̄, not functional. If s' = s̄: 1s ↦ 1s and 1s̄ ↦ 1s,
   not injective.
 (B) Identical to (A) with a and b exchanged: 10s, 11s' give (1s, 1s), (1s', 1s̄') in R_r on (a, t). ∎

## Proofs of the theorems

**Theorem 1.** (⇐ fails, i.e. c* ≥ 1:) If 11 ∈ Π and Π ≠ {11}, a 0-CNOT circuit is
U = A ⊗ B ⊗ C = (A ⊗ B) ⊗ C, of the form V ⊗ W with k = t (Lemma 1), contradicting Lemma 3 by
Lemma 4(T). This is (B1); it only uses the minimal pair {11t, uvs}.
(c* = 0 otherwise:) If 11 ∉ Π, f is the identity on L and U = 1 works. If Π = {11}, then
L ⊆ {110, 111} and U = 1 ⊗ 1 ⊗ X works (X|t> = |t̄>). ∎

**Theorem 2.** Let Π ⊇ {01, 10, 11}. A 0-CNOT circuit is excluded by Theorem 1 (11 ∈ Π,
Π ≠ {11}). A 1-CNOT circuit has U = V ⊗ W with W on the untouched qubit k (Lemma 1), and
k ∈ {a, b, t}:
 * k = t: excluded by Lemma 4(T) and Lemma 3 (whatever V is, including its CNOT);
 * k = a: excluded by Lemma 4(A) ({01,11} ⊆ Π) and Lemma 3;
 * k = b: excluded by Lemma 4(B) ({10,11} ⊆ Π) and Lemma 3.
The CNOT may be on any pair in either direction; that only fixes k, and all three k are
excluded. Hence c*(L) ≥ 2. ∎

Note on part (iii) of the task: with one CNOT some qubit is always untouched, and Lemma 3
already kills the *whole* candidate U = V ⊗ W for every V (not only V with one CNOT), so no
separate analysis of the two-qubit one-CNOT block is needed. The obstruction is purely
combinatorial: the required map on the remaining two qubits is not a partial bijection.

**Corollary 3.** Classes 3, 4, 6 have Π = {00,01,10,11} ⊇ {01,10,11}, so Theorem 2 applies
directly. (Alternatively: each such L contains a 3-element L' = {01s, 10s', 11s''} ⊆ L with
Π(L') = {01,10,11}; Lemma 0 and Theorem 2 give c*(L) ≥ c*(L') ≥ 2.) ∎

## Case list (minimal supports)

| bound | minimal L | excluded k-CNOT | qubit split off | obstruction |
|---|---|---|---|---|
| B1 | {11t, uvs}, uv ≠ 11 (2·3·2 = 12 sets) | 0 | t | Lemma 4(T) |
| B2 | {01s, 10s', 11s''} (8 sets) | 0 | t | Lemma 4(T) |
| B2 | same | 1, CNOT on {a,b} | t | Lemma 4(T) |
| B2 | same | 1, CNOT on {b,t} | a | Lemma 4(A): 01s, 11s'' |
| B2 | same | 1, CNOT on {a,t} | b | Lemma 4(B): 10s', 11s'' |

## Exact verification (exhaustive, no floating point)

`enumeracja.py` (pure Python, finite set operations) enumerates all 255 nonempty L ⊆ {0,1}^3 and checks:
 (i) t splittable ⇔ ¬(11 ∈ Π ∧ Π ≠ {11}) [Lemma 4(T) and its converse];
 (ii) a splittable ⇔ {01,11} ⊄ Π;  b splittable ⇔ {10,11} ⊄ Π [Lemma 4(A),(B) and converses];
 (iii) the 0-CNOT condition of Lemma 3 (all three single-bit relations partial injections)
   holds ⇔ claimed class 0;
 (iv) no qubit splittable ⇔ claimed class ≥ 2;
 (v) every L with Π = {00,01,10,11} contains a 3-subset with Π = {01,10,11};
 plus a cross-check against the numerical data `koszt.jsonl` (no support is numerically
 implementable with fewer CNOTs than the proven lower bound).
Output (`enumeracja_output.txt`):

```
supports enumerated: 255
class, splittable qubits -> #L
   (0, ('a', 'b', 't')) 66
   (1, ('a',)) 36
   (1, ('a', 'b')) 9
   (1, ('b',)) 36
   (2, ()) 27
   (3, ()) 72
   (4, ()) 8
   (6, ()) 1
FAILURES: none
full-Pi supports without a class-2 3-subset: 0
numeric cross-check: 119 supports with a feasible k; numeric c* below proven lower bound: 0
```

The table also shows consistency with the claimed class-1 upper bound: every class-1 support has
a or b splittable (the obstruction disappears exactly where a 1-CNOT circuit is claimed).

## Status

* Proven, fully: Lemma 0–4, Theorem 1 (B1, plus c* = 0 ⇔ class 0), Theorem 2 (B2),
  Corollary 3 (c* ≥ 2 for classes 3, 4, 6). The computer check is a redundant exact
  confirmation of the finite combinatorics; the proof does not depend on it.
* Not addressed here (out of scope of P1): (B3) c* ≥ 3 and (B4) c* ≥ 4, which require
  excluding 2- and 3-CNOT circuits where no qubit need be untouched, and the matching upper
  bounds (taken from the constructive circuits in PROBLEM.md).
