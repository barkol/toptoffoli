# Verdict on P1 (B1, B2, c* >= 2 for classes 3, 4, 6): CORRECT

Checked step by step, trying to break each one:
- Lemma 1 (<=1 CNOT => U = V (x) W). With 0 CNOTs U = A(x)B(x)C. With 1 CNOT on {i,j}, every gate acts either on {i,j} or on k, so the gates commute across the cut. The CNOT direction plays no role, and single-qubit gates on i and j are absorbed into V. Holds. Re-ordering the tensor factors by qubit label is a fixed isomorphism, which is harmless.
- Lemma 2 (u(x)w = lambda r(x)q => u ~ r, w ~ q). Apply 1(x)<q| to get u<q|w> = lambda r != 0, then <r|(x)1. Correct. Only u, w != 0 and r, q normalised are needed, and both hold because V, W are unitary and the targets are basis vectors.
- Lemma 3. Each x gives V|x_r> ~ |f(x)_r> and W|x_k> ~ |f(x)_k> separately, so the "basis state -> superposition" worry does not apply. Single-qubit gates that create superpositions on intermediate states are allowed, but the factorised end map is forced to be monomial on the inputs used. Functionality follows from orthogonality of distinct basis vectors. Injectivity follows from unitarity. Phases may depend on x, so the bound is stronger than needed and the global vs per-string phase question does not matter.
- Lemma 4 (T), (A), (B). Recomputed by hand, including both subcases s' = s and s' != s. Correct.
- Theorem 1. The converse (c* = 0 for the complementary class) holds via U = 1 or U = X_t. B1 uses only k = 0, so it is complete.
- Theorem 2. All 3 untouched qubits are covered, and each corresponds to 2 CNOT directions. k = 0 and k = 1 are both excluded directly, so the k -> k+2 padding issue does not arise.
- Corollary 3. Pi = {00,01,10,11} contains {01,10,11}, so it follows directly.
- enumeracja.py. Re-read the code. It does exact set logic and the output is consistent.

Numerical adversary (szukaj.py, BFGS, 25-40 random starts per case, loss = 1 - |sum_x <f(x)|U|x>|/|L|):
- 0 CNOT on all 12 B1 pairs, and 1 CNOT on all 8 class-2 triples x 6 ordered CNOT pairs (60 cases in total). Min loss = 0.195, bounded away from 0, so no counterexample was found.
- Positive controls reach loss ~1e-14: class-1 {110,010} with 1 CNOT, and class-2 {010,100,110} with CX(a,t)CX(b,t). So the optimiser can find solutions when they exist.

Optional cosmetic fixes (not required for correctness):
1. In Lemma 1, state explicitly that "V (x) W" means up to the qubit-reordering isomorphism.
2. In Lemma 2, note that only q needs to be normalised for the first step and r for the second.
