# P5 — literature for the lower bounds (B1)–(B4)

Date: 2026-10-03. Each entry was checked against the arXiv abstract page or the full text (PDFs and their `pdftotext` output are in `P5/pdf/`).
Verdict key: **direct** = gives a bound with no new work; **tool** = a method or criterion that a proof can use; **no** = does not apply, and why.

---

## A. Results that a proof can use directly

### A1. Žnidarič, Giraud, Georgeot — CNOT classes of three-qubit states  [tool, strongest]
M. Žnidarič, O. Giraud, B. Georgeot, "Optimal number of controlled-NOT gates to generate a three-qubit state",
Phys. Rev. A 77, 032320 (2008), doi:10.1103/PhysRevA.77.032320, arXiv:0711.4021.
Gate model: CNOTs plus arbitrary single-qubit unitaries, the same model as ours.
Exact statement, with d(ψ) = the minimal number of CNOTs that maps |000> to ψ:
- d = 0 iff ψ is a product state.
- d = 1 iff ψ is biseparable, |α>_i ⊗ |χ>_jk with χ entangled.
- d ≤ 2 iff, for some qubit i and some orthonormal pair {α, α⊥} on i, ψ = cos φ |α>|β>|γ> + sin φ |α⊥>|β'>|γ'>. In words: on one qubit there is an orthonormal basis such that both conditional two-qubit states are product.
- d = 3 for every other state (W-like states, and generic GHZ-class states).
- Also: any state is at most 2 CNOTs from GHZ, and any two states are at most 4 CNOTs apart (upper bound only; pairs at distance 4 are not classified).

**How this gives our bounds (my derivation, short and checkable).** Let ψ ∈ span(L). Then Uψ = e^{iθ} CCX ψ. Since d(·) is invariant under local unitaries and subadditive under composition,
  **c*(L) ≥ d(CCX ψ) − d(ψ)  for every ψ ∈ span(L).**   (*)
- **(B1), all cases.** Take L = {x, y} with x = 11t and y = uvs, uv ≠ 11, and ψ = |x> + |y>. A two-term state |x> + |y> has a local-unitary class fixed by the Hamming distance h(x, y): h = 1 gives a product state, h = 2 a biseparable one, h = 3 GHZ. CCX flips t on x only, so h changes by exactly 1, and the class changes. Single-qubit gates alone cannot change the class, so k ≥ 1.
- **(B3) with a constant target**, L = {00x, 01x, 10x, 11x} (the "AND into a clean |0> or |1> ancilla" case). Take ψ = |+>|+>|x>, so d(ψ) = 0. The output is |000> + |010> + |100> + |111> (up to X_t).
  - Every single-qubit reduced state is mixed, so the output is not biseparable.
  - Qubit t, conditioned on γ: the coefficient matrix [[γ0, γ0], [γ0, γ1]] has det γ0(γ1 − γ0). No orthonormal pair makes both determinants vanish.
  - Qubit a, conditioned on α: the matrix [[α0 + α1, 0], [α0, α1]] has det (α0 + α1)α1. Again no orthonormal pair works.
  - Qubit b follows from the a↔b symmetry.
  So d(output) = 3, and therefore **c* ≥ 3**.
- **(B2) with a constant target**, L = {01x, 10x, 11x}. Take ψ = (|01> + |10> + |11>)|x>, so d(ψ) = 1. The output is |010> + |100> + |111> (x = 0).
  - It is not biseparable: all three reduced states are mixed.
  - Conditioning on t gives det −γ0², so only γ = |1> works, and its orthogonal partner fails.
  - Conditioning on a gives det α1², so only α = |0> works, and its partner gives |00> + |11>, which is entangled.
  - Qubit b follows from symmetry.
  So d(output) = 3, and therefore **c* ≥ 3 − 1 = 2**.
- **Non-constant targets in (B2)/(B3).** Bound (*) still holds, but ψ must be chosen from span(L) with small d(ψ) and a class-3 image. Choosing it needs a case search over the 3- or 4-term superpositions. This is not done here, but it is a finite symbolic check for each support.
- **Limitation for (B4).** Since d ≤ 3 for every state, (*) can never certify more than 3. (B4) needs a pair (ψ, CCX ψ) at state-to-state distance 4, or an argument that uses several inputs at once (see A2 and A3). Žnidarič et al. do not characterise pairs at distance 4.

### A2. Song & Klappenecker — the simplified Toffoli needs 3 CNOTs  [tool; technique adapts to subspaces]
G. Song, A. Klappenecker, "Optimal realizations of simplified Toffoli gates", Quantum Inf. Comput. 4(5), 361–372 (2004), ACM DL 10.5555/2011586.2011588. The preprint is arXiv:quant-ph/0312225, titled "The simplified Toffoli gate implementation by Margolus is optimal".
- **Theorem M.** "Suppose that the simplified Toffoli gate M is realized by a sequence of controlled-not and single qubit gates. Any such sequence contains at least three controlled-not gates. If it contains three …, then at least four single qubit gates are needed." Here M is the Toffoli up to a diagonal of phases on the whole space (Margolus).
- **Proof ingredients, all entanglement-based:**
  - Lemma 1: a controlled-U acting on |ψ>|φ> produces entanglement iff φ is not an eigenvector of U and ψ has both amplitudes nonzero.
  - Lemma 9: entanglement between systems A and B survives gates that act on A and B separately.
  - Lemma 5 / Cor. 6: if only one CNOT touches the target, then the input |ϕ> ⊗ C1†|+> leaves the target unentangled for every ϕ. Meanwhile M maps |00>φ + |10>φ + |11>φ to |00>φ + |10>Zφ + |11>Xφ, which is entangled. So at least two CNOTs must touch the target.
  - Lemma 7: two CNOTs on the target from different controls. With a = 0 the circuit can still entangle b and t, while M acts locally there.
- **What it uses.** Lemma 5 uses inputs supported on control values {00, 10, 11} with an arbitrary target state, which needs 6 basis strings. Lemma 7 uses the whole a = 0 block. So the theorem does not imply (B2) or (B3) for 3- or 4-element L. It does fix the template: reduce by "some qubit is touched by ≤ 1 CNOT, so the circuit factorises", then exhibit an input in span(L) whose image has the wrong entanglement. That template is exactly (*) above.
- Relevance to class 3: c*(L) = 3 is attained by the Margolus gadget. When L is so large that the gate must equal Margolus on span(L), Theorem M is the matching lower bound. It is not needed for minimal L.

### A3. Shende & Markov 2009 — the CNOT cost of Toffoli  [direct for L = {0,1}^3; tool for (B4)]
V. V. Shende, I. L. Markov, "On the CNOT-cost of TOFFOLI gates", Quantum Inf. Comput. 9(5–6), 461–486 (2009), arXiv:0803.2316.
- **Theorem 1.** "A circuit consisting of CNOT gates and one-qubit gates which implements the n-qubit TOFFOLI gate without ancillae requires at least 2n CNOT gates. For n = 3, this bound holds even when ancillae are permitted." This gives c*({0,1}^3) = 6. Toffoli up to a global phase is the same requirement.
- **Corollary 23.** |PERES|_CZ = 5.
- **Theorem 30.** Complete CZ-cost classification of three-qubit diagonal operators by the invariants s(D) = (λ1, λ2, λ3; ξ). The cost is 4 CZ iff s(D) = (a, b, c; ab/c), and 6 always suffices. This is relevant if a (B4)-type support can be reduced to "Toffoli up to a diagonal on a subspace".
- **Tools** (they require full operator identities, not identities on a subspace):
  - Qubit-local cost |P|_{CZ;ℓ} ≥ … via the ℓ-mux-spectrum (Def. 16, Thm 17–18): 0 CZ on ℓ iff the spectrum of U1†U0 is congruent to {1, 1, …}; 1 CZ iff {1, −1, 1, −1, …}; ≤ 2 iff the eigenvalues come in conjugate pairs.
  - Total CZ count ≥ ½ · Σ_ℓ (local counts).
  - Lemma 8 / Cor. 9–13: matrix-entry constraints that force single-qubit gates to be diagonal or anti-diagonal.
- **Possible route to (B4).** Q ⊆ L forces U = e^{iθ}·I on span{01, 10} ⊗ C², and U must map 00x → 00x and 11y → 11ȳ. A circuit with ≤ 3 CNOTs has a qubit touched by ≤ 2 CNOTs. Corollary 13 (exactly two CZs on ℓ makes all gates on ℓ diagonal) and Lemma 8 are pure matrix-element arguments. They can be redone with the identity imposed only on span(L), because each one uses only specific entries ⟨i|·|j⟩. This adaptation has not been checked; it is the most promising route to (B4).

---

## B. Supporting criteria (exact, two-qubit or LU-invariant)

### B1'. Shende, Markov, Bullock — two-qubit CNOT counts
- V. V. Shende, I. L. Markov, S. S. Bullock, "Minimal universal two-qubit controlled-NOT-based circuits", Phys. Rev. A 69, 062321 (2004), arXiv:quant-ph/0308033. Prop. IV.3, with γ(u) = u σy⊗σy u^T σy⊗σy:
  - u is local iff γ(u) = I;
  - u and v are related by left local unitaries iff γ(u) = γ(v);
  - u and v are related by local unitaries on both sides iff χ[γ(u)] = χ[γ(v)].
- V. V. Shende, S. S. Bullock, I. L. Markov, "Recognizing small-circuit structure in two-qubit operators…", Phys. Rev. A 70, 012310 (2004), arXiv:quant-ph/0308045. Explicit tests for 0, 1, 2 CNOTs; 2 CNOTs suffice iff tr γ(u) is real.
- **Use.** In the 2-CNOT case where both CNOTs act on the same pair, U = V_{ij} ⊗ W_k. The restriction to span(L) then becomes a condition on V. That condition is again only on a subspace, so these tests apply only after V is shown to be determined on enough states. (*) is usually simpler.

### B2'. Shende & Markov — incompletely specified two-qubit operators
V. V. Shende, I. L. Markov, "Quantum circuits for incompletely specified two-qubit operators", Quantum Inf. Comput. 5(1), 49–57 (2005), arXiv:quant-ph/0401162.
- This is the earliest work found on "don't-care" synthesis (known input |0>, known final measurement).
- Prop. 1: two-qubit pure states are LU-equivalent iff |⟨φ*|σy⊗σy|φ⟩| is equal.
- Prop. 4: one CNOT is necessary and sufficient to prepare any entangled two-qubit state.
- Prop. 11: two CNOTs suffice up to a computational-basis measurement.
- Two qubits only. It gives no three-qubit lower bounds, but it is the right precedent to cite for the concept.

### B3'. Dür, Vidal, Cirac — three-qubit SLOCC classes
W. Dür, G. Vidal, J. I. Cirac, Phys. Rev. A 62, 062314 (2000), arXiv:quant-ph/0005115. Its GHZ/W/biseparable/product classes are invariant under local unitaries, which suffices for the k = 0 steps. A1 is the finer, CNOT-graded version.

### B4'. Operator Schmidt rank
M. A. Nielsen et al., "Quantum dynamics as a physical resource", Phys. Rev. A 67, 052301 (2003), arXiv:quant-ph/0208077.
- Schmidt number is submultiplicative, and a CNOT has Schmidt number 2 across its cut.
- This is weak here. CCX has operator Schmidt rank 2 across every single-qubit cut, so the method only shows that every cut is crossed, which means ≥ 2 CNOTs in total. It does not reach (B3) or (B4).

---

## C. Checked and found not to give (B1)–(B4)

- **Iten, Colbeck, Kukuljan, Home, Christandl**, "Quantum circuits for isometries", Phys. Rev. A 93, 032318 (2016), arXiv:1501.06911.
  - The lower bound is a parameter count: an m→n isometry has 2^{n+m+1} − 2^{2m} − 1 real parameters, which gives ⌈(2^{m+n+1} − 2^{2m} − 3m − n − 1)/4⌉ CNOTs for **generic** isometries.
  - It holds only outside a measure-zero set. The Toffoli-into-|0> isometry (and every CCX·P_L) is a specific, highly non-generic map, so the bound gives **nothing** for class 3. The same applies to Knill (1995) and Shende–Bullock–Markov (2006) counting bounds.
- **C. Gidney**, "Halving the cost of quantum addition", Quantum 2, 74 (2018), arXiv:1709.06648. The temporary logical-AND is optimised and analysed for **T-count** (4 T). There is **no CNOT lower bound**; the paper itself says the T lower bound is only "suspected". Jones, Phys. Rev. A 87, 022328 (2013) is the equivalent construction. **No paper found proves 3 CNOTs optimal for "AND into a clean |0> with relative phases allowed".** A1 (B3, constant target) appears to give this as a two-line corollary.
- **D. Maslov**, "Advantages of using relative-phase Toffoli gates with an application to multiple control Toffoli optimization", Phys. Rev. A 93, 022311 (2016), arXiv:1508.03273.
  - It states that the 3-CNOT relative-phase Toffoli (Margolus) is optimal "shown in [11]" (Song–Klappenecker).
  - For Toffoli-4 lower bounds it cites Shende–Markov (≥ 8 CNOT).
  - No new lower-bound technique.
- **J. Liu, L. Bello, H. Zhou**, "Relaxed peephole optimization: a novel compiler optimization for quantum circuits", CGO 2021, arXiv:2012.07711. This replaces Toffoli/CCX by cheaper gates when inputs are known basis states. Upper bounds and heuristics only; it is the compiler precedent for "reachable-support" savings.
- **W. Jang et al.**, "Initial-state dependent optimization of controlled gate operations with quantum computer", Quantum 6, 798 (2022), arXiv:2209.02322. Same kind of work: heuristic, no lower bounds.
- **H. Wang, D. B. Tan, J. Cong**, "Quantum state preparation circuit optimization exploiting don't cares", arXiv:2409.01418 (2024). Heuristic peephole optimisation, no lower bounds.
- **Two-qubit-gate counts, not CNOT counts:**
  - N. Yu, R. Duan, M. Ying, "Five two-qubit gates are necessary for implementing the Toffoli gate", Phys. Rev. A 88, 010304(R) (2013), arXiv:1301.3727.
  - K. Huang, J. Palsberg, "Toffoli requires six quantum neighbor gates" (UCLA preprint, 2025): arbitrary two-qubit gates, nearest-neighbour; also classifies 3-qubit diagonals.
  - Neither is directly useful.

---

## Bottom line for the provers

1. Inequality (*) together with the Žnidarič–Giraud–Georgeot classification proves (B1) completely. It also proves (B2) and (B3) for constant-target supports (Π-complete supports with all targets equal) in a few lines with exact algebra. For the other target patterns of (B2)/(B3), search span(L) for ψ with d(CCX ψ) − d(ψ) large enough. This is finite and symbolic: test product, biseparable and class-2 forms through determinant conditions.
2. (B4) cannot be reached by any single-input argument based on distance from |000>, because d ≤ 3. Use multiple inputs at once: adapt Song–Klappenecker Lemma 5/7 and Shende–Markov Lemma 8 / Cor. 13 / Thm 18 to identities that hold only on span(L) = span(Q ∪ {00x, 11y}). On that span, U acts as the identity on span{01, 10} ⊗ C².
3. Nothing in the literature states the subspace-restricted values c*(L) themselves. In particular, no paper proves CNOT optimality of the 3-CNOT "logical AND into |0>". Proposition 1 appears to be new.
