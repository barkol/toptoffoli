# Referee verdict on P2/DOWOD.md, (B3): CORRECT (minor presentation fixes only)

## Step-by-step check
1. **Reduction to a chain.** k<=1, or two CNOTs on one pair (either direction): one qubit is never touched by a CNOT, so U = V_q (x) W. Two CNOTs on different pairs: with 3 qubits the pairs always share exactly one qubit j. Moving single-qubit gates is valid: L1_k and L2_k commute with C1 and with all gates on i and j, and L3_i commutes with C2 and with L2_k. This gives U = (I_i (x) V2_jk)(V1_ij (x) I_k). All placements and directions are covered, and CNOT direction is irrelevant because V1 and V2 are arbitrary.
2. **Lemma 2.** No gate acts on qubit i after V1, so the marginal of qubit i in U|x> equals its marginal in V1|x_i x_j>. This marginal is the pure state |f_i(x)>, so V1|x_i x_j> is a product state |f_i(x)> (x) eta_x, with the phase absorbed into eta_x. This holds for every x in L.
3. **Lemma 3.** Inputs are basis states, so they are either equal or orthogonal. V1-a, V1-b, V2-a and V2-b follow from unitarity. In V2-b the output overlap modulus is 0 or 1 because the outputs are basis states times phases, so "=" and "perp" are the only possible labels. Pairs that share nothing still get the V2-a check (orthogonal inputs must give distinct outputs); only pairs whose outputs differ get no constraint. Two labels on the same pair are handled consistently (the union-find in check_b3.py does this).
4. **Lemma 4.** Correct in C^2: the orthogonal complement r -> r^perp is a fixed-point-free involution on rays.
5. **Cases A and B.** I recomputed all labels by hand.
   - Case A, chain (a,t,b): the cycle 00-01-11-10 has parity (t0+t1)+(t1+t3+1)+(t3+t2)+(t2+t0) = 1 (mod 2) for all 16 patterns.
   - Case B, chain (b,a,t): step 1 forces t2=t3, and the three sub-cases give odd cycles (perp,perp,perp,=), (perp,perp,perp), (perp,perp,perp). The extra V2-b labels on 00-10 and 01-11 (when t0=t2 or t1=t3) are perp, consistent with the V1 labels, so no hidden conflict arises.
6. **Symmetry and inversion.** The a<->b swap commutes with CCX. U^dagger is the chain (k,j,i) and satisfies the condition on f(L), which is again a (B3) support (f fixes ab, and the phase becomes its conjugate). The orbits {atb,bta} and {bat,abt,tab,tba} cover all 6 chains. check_b3.py also tests all 6 chains directly without using symmetry.
7. **Stronger claim (arbitrary 2-qubit unitaries, per-string phases).** The proof never uses that the gates are CNOTs, so the stronger claim holds.

## Numerical adversarial search (search.py; floating point, used only to search for counterexamples)
- Set-up: two arbitrary U(4) gates in each of the 6 chains, plus the 3 untouched-qubit structures. Loss = sum_x (1 - |<f(x)|U|x>|^2), which allows per-string phases. 12 random starts x 500 Adam steps per structure.
- Coverage: patterns 0-7 (search_b3.out; the run was stopped by its time limit after pattern 7) plus pattern 000,011,101,111 (search_last.out). Flipping all target bits (X_t commutes with CCX) maps the other 8 patterns onto these.
- Result: the best loss is 0.5858 = 2 - sqrt2 everywhere, and exactly 1.0 for every structure that Lemma 1 or Cases A/B exclude.
- Positive control (search_pc.out), Pi={01,10,11}: loss 0 is reached for exactly the chains that check_b3.out admits.
- A direct 2-CNOT search is not needed, because it is a special case of the U(4) search.

## Fixes (cosmetic)
- (a) In the reduction, add a sentence saying that the CNOT direction plays no role.
- (b) In Lemma 3 (V2-b), say that the modulus is in {0,1} because the outputs are basis states.
- (c) In Case B, note that the extra V2-b labels on 00-10 and 01-11 agree with the V1 labels.
