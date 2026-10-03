# Referee verdict on P4/DOWOD.md, (B4): c*(Q ∪ {00x,11y}) ≥ 4

Referee: V_P4a. Date: 2026-10-03.

**Verdict: CORRECT, with cosmetic fixes only.** I checked every step by hand and reran both scripts with identical output. An independent numerical cross-check (not part of the proof) agrees.

## Internal consistency (possible external edit)
I read DOWOD.md (mtime 08:49) against the summary: Prop A (diag(1,1,1,λ) vs c{1,1,−1,−1}), Prop B (non-product M, unique U = CCX·(I−2|00x̄><00x̄|), Pauli coefficients ½(IX+IZ−ZX+ZZ), G1 = q²/2, G2 = 1+q²−r²−s², Q = ±I, G1(G(ZQ)) = ½), and Prop C (triangle; (i) V_a diagonal/antidiagonal, (ii) not).
No passage contradicts the summary or another passage. The formulas in the text match the script outputs: K's Pauli coefficients for x = 0, the Makhlin invariants, N_A, N_B, V_0X_tV_0† = K.

## Attack points
1. **Exhaustive case split.** The degrees sum to 2k. For k ≤ 2 some qubit has degree ≤ 1. For k = 3, if every degree is ≥ 2 then the degrees are (2,2,2).
   If any pair were used twice, the third CNOT would give the remaining qubit degree 1. So (2,2,2) forces the triangle. k = 0 falls under Prop A (deg t = 0). The split is exhaustive. ✓
2. **Prop A and completion freedom.** No gap. The block form U = Σ P_k⊗G'_k is a property of the *circuit* (Lemma 0), not of L. The t-components of f_k are nonzero (forced by 11y), so each L-input constrains *both* blocks.
   Inputs 10t and 01t (both t), 11y and 00x (one t each) then fix G'_k on all four ab basis states. That gives G'_k = diag(1,1,1,g_k) exactly, and the spectrum argument uses only these forced entries. ✓
3. **Prop B uniqueness.** The free data are exactly the two unitarity phases h and s (the fourth columns of G'_0 and G'_1). The spectral match {1, h̄, ±√s} = c{1,1,−1,−1} forces h = −1 and s = 1. Again the block structure comes from the circuit. ✓
   B2: if V were local, VX_rV† would be a product, so V and W each need at least one CNOT and k ≥ 3. ✓
   B4: local conjugation preserves span{I⊗σ_j}, and CX(n·σ_b)CX has no component there, while K has I⊗X with coefficient ½. ✓
4. **Makhlin step.** The invariants are computed for G(Q) = V_0·C_Q and G(ZQ). Both are reached correctly:
   - C = V_0†V lies in the (anti)commutant of X_t.
   - The right local factors and the U(2) phase are removed without changing either cost condition.
   - N is Hermitian, so cost(W) = cost(V_0NC).
   - N·C_Q = C_{∓ZQ}, and H_t-conjugation gives G(·).
   "cost ≤ 1 ⇔ locally equivalent to I or CNOT ⇔ (G1,G2) ∈ {(1,3),(0,1)}" is correct, and G1 as written is invariant under global phase. Q = [[p+iq, −r+is],[r+is, p−iq]] is the general element of SU(2).
   Rerun of lemat_11.py: identical output. Independent float check (`niezalezne.py`): the closed forms match to 2e-15 over 2000 random Q. The Shende–Bullock–Markov criterion gives min_Q max(dist G(Q), dist G(ZQ)) ≈ 1.53, bounded away from 0. ✓
5. **Prop C by hand.**
   - The 6 orders reduce correctly (I checked all four reductions).
   - Case (i) gives a block form with a product G_0†G_1, which contradicts Lemma 1. Lemma 1 needs only (*), so it applies.
   - Case (ii): both u_j are non-computational (u_0 ⊥ u_1). S_0|c> is an X eigenvector, so σ_1 = −σ_0.
     - Order (ab,at,bt): 11y and 01ȳ ∈ L share the output bt = 1ȳ, while their t-factors are orthogonal. ✓
     - Order (at,ab,bt): (σ_1 − σ_0)ρ G_{0,1}|y> = 0 with ρ ≠ 0 (since u_{1,1} ≠ 0). ✓
6. **Symmetries.** S1: L and CCX are invariant under a↔b. ✓ S2: X_tCCX X_t = CCX and L(x,y) → L(x̄,ȳ), with a local gate only. ✓
   S3: U†CCX|z> = |z>, CCX is an involution, CCX(L(x,y)) = L(x,ȳ) (same family), and the phase becomes −θ. ✓

## Fixes (cosmetic)
- Prop C (at,ab,bt): write (σ_1 − σ_0)ρ G_{0,1}|y> = 0 and state ρ ≠ 0 (because u_{1,1} ≠ 0). The current "(σ_1/σ_0 − 1)G_{0,1}|y>" drops the factor ρ.
- Prop A: say explicitly that |e_k> = λ̄_k|f_k>, so U = Σ λ̄_k P_k⊗G_k.
- Conclusion, "by-product … and it costs 4": the lower bound follows from the theorem, but the upper bound (that this U has a 4-CNOT circuit) is not proven in the text. Either drop "costs 4" or cite a construction. See koszt_uboczny.out for the numerical status.

## Numerical status of the by-product (koszt_uboczny.out, floating point, not a proof)
Optimising the full unitary U* = CCX·(I−2|001><001|) (x = 0) over all pair sequences gives:
- k = 3: best loss 0.076, consistent with the theorem.
- k = 4: exact up to 8e-10 with the sequence (at, at, bt, at).
So "costs 4" is numerically correct. The doc should either state it as a numerical remark or give the explicit circuit; the theorem itself does not need it.
