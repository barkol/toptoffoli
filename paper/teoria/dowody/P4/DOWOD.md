# (B4): c*(L) >= 4 for L = Q ∪ {00x, 11y}

Author id: P4. Date: 2026-10-03.

## Statement

**Theorem.** Let Q = {100, 101, 010, 011} (strings abt), x, y ∈ {0,1}, and L = Q ∪ {00x, 11y}.
No circuit made of k ≤ 3 CNOT gates (any ordered pairs) and arbitrary single-qubit unitaries
satisfies U|z> = e^{iθ} CCX|z> for all z ∈ L (one common phase θ).

Together with the explicit 4-CNOT construction (class 4 upper bound) this gives c*(L) = 4, and by
monotonicity c*(L') >= 4 for every L' ⊇ Q ∪ {00x, 11y}.

**Dependencies.** None on (B1)–(B3): the cases k = 0, 1, 2 are excluded here directly
(Propositions A and B). The only external results used are
(i) the identity CX(r,q) = (H⊗H) CX(q,r) (H⊗H), and
(ii) Makhlin's theorem: two-qubit gates U, U' are locally equivalent iff their invariants
G1 = tr²(m)/(16 det U), G2 = (tr²(m) − tr(m²))/(4 det U), m = U_B^T U_B (U_B = U in the magic basis)
coincide [Makhlin, Quantum Inf. Process. 1, 243 (2002); Zhang et al., PRA 67, 042313 (2003)].
It is used once, in Step B5, to decide whether a two-qubit gate needs at most one CNOT.

**Status: everything is proven. No open cases.** Computer steps: two sympy scripts doing exact
symbolic algebra (`tozsamosci.py`, `lemat_11.py`, outputs in `*.out`). No floating point is used in the proof.

## 0. Conventions and elementary facts

* WLOG θ = 0 (absorb e^{-iθ} into any single-qubit gate). U is then required to satisfy
  U|z> = |CCX z> for z ∈ L.
* Direction of a CNOT is irrelevant: CX(r,q) = (H⊗H)CX(q,r)(H⊗H), and the Hadamards are absorbed
  into the neighbouring single-qubit layers. So when we look at a CNOT touching qubit q, we can take q as its control.
* The **degree** of a qubit is the number of CNOTs that touch it. The degrees sum to 2k.
  For k ≤ 2 some qubit has degree ≤ 1. For k = 3, either some qubit has degree ≤ 1 or the degrees
  are (2,2,2). In the second case the three CNOTs act on the three distinct pairs ab, at, bt ("triangle").
* **Data of CCX on L** (θ = 0). In the table the output a equals the input a:

  | input z | output CCX z | a | bt in → bt out |
  |---|---|---|---|
  | 100, 101 | same | 1 | 00→00, 01→01 |
  | 11y | 11ȳ | 1 | 1y→1ȳ |
  | 010, 011 | same | 0 | 10→10, 11→11 |
  | 00x | same | 0 | 0x→0x |

  For each value of a, both values of b and both values of t occur among the inputs.
* **Symmetries.** Each map below sends a valid circuit for L(x,y) to a valid circuit for L(x',y') with the same number of CNOTs:
  - S1 (swap a↔b, conjugate by SWAP_ab): (x,y) → (x,y). Exchanges the roles of a and b in the topology.
  - S2 (conjugate by X_t): X_t CCX X_t = CCX and X_t L(x,y) = L(x̄,ȳ). Same topology.
  - S3 (take U†, which reverses the circuit): U†|w> = CCX|w> for w ∈ CCX(L) = L(x,ȳ). Reverses the CNOT order.

### Lemma 0 (one CNOT across a cut)
Let q be a qubit of degree ≤ 1 and R the other two qubits. Then
U = Σ_{k=0,1} |e_k><f_k|_q ⊗ G_k, where {e_k}, {f_k} are orthonormal bases of C², G_k are unitaries on R, and
* if deg q = 1 (the CNOT acts on q and r ∈ R): G_k = V_R X_r^k W_R, with e_k = V_q|k>, f_k = W_q†|k>.
  Here V_R and W_R are the parts of the circuit on R after and before that CNOT. Their CNOT counts add up to k − 1.
* if deg q = 0: G_0 = G_1.

*Proof.* Write U = (V_q⊗V_R) CX(q,r) (W_q⊗W_R) (direction normalised, see §0). Then expand
CX(q,r) = Σ_k |k><k| ⊗ X_r^k. For degree 0 we have U = V_q ⊗ G. ∎

### Lemma 1 (block form forced by L)
Suppose U = Σ_k |e_k><f_k|_a ⊗ G_k as in Lemma 0 with q = a, and U agrees with CCX on L.
Then, after relabelling k and rephasing, U = |0><0|_a ⊗ G'_0 + |1><1|_a ⊗ G'_1 with G'_j ∝ G_{π(j)} for a permutation π.
In the basis |bt>:

* G'_0 = identity on |10>, |11>, |0x>, and G'_0|0x̄> = h|0x̄> with |h| = 1;
* G'_1 = identity on |00>, |01>, G'_1|1y> = |1ȳ>, and G'_1|1ȳ> = s|1y> with |s| = 1.

Consequently G'_0†G'_1 and G'_1G'_0† are block-diagonal in b. Their b=0 block is diagonal (diag(1, h̄) up to order). Their b=1 block is P_s = |ȳ><y| + s|y><ȳ|, which is anti-diagonal and nonzero.
**Neither of them is a product operator A_b ⊗ B_t.** A product that is block-diagonal in b has A diagonal, so its two blocks are proportional, while a diagonal and an anti-diagonal nonzero 2×2 matrix are not proportional.

*Proof.* Apply <e_k|_a ⊗ I to U|z> = |CCX z>. This gives (*) <f_k|z_a> G_k|r> = <e_k|z_a> |o_z>, where r is the input bt and o_z the output bt.
Suppose f_0 had both components nonzero. Then f_1 would too. For z = 010, 011 ∈ L (a = 0, inputs bt = 10, 11, see table), (*) gives G_k|10> ∝ |10> and G_k|11> ∝ |11>.
Both are nonzero because G_k is unitary. Now take z = 11y (z_a = 1, r = 1y ∈ {10,11}). The left side of (*) is a nonzero multiple of |1y>. The right side is a multiple of |1ȳ>. Contradiction.
So f_k ∝ |π(k)>. Then (*) with z_a ≠ π(k) gives <e_k|z_a> = 0, so e_k ∝ |π(k)> too.
Absorb the phases into G'_j. Now (*) reads G'_j|r> = |o_z> for all z ∈ L with z_a = j. This gives the listed columns. The remaining column is forced by unitarity, since it must be orthogonal to the three listed images. ∎

## Proposition A: t cannot have degree ≤ 1 (any k)

Apply Lemma 0 with q = t: U = Σ_k |e_k><f_k|_t ⊗ G_k, with G_k on ab.

* Inputs ab = 10, t = 0, 1. Apply <e_k| to U|t,10> = |t,10>: <f_k|t> G_k|10> = <e_k|t>|10>.
  So G_k|10> = λ_k|10> with |λ_k| = 1, and <e_k| = λ_k<f_k|. The same argument with ab = 01 gives G_k|01> = λ_k|01> (same λ_k, because e_k is fixed).
  So |e_k> = λ̄_k|f_k>. Put G'_k = λ̄_k G_k. Then U = Σ_k P_k ⊗ G'_k with P_k = |f_k><f_k|, and G'_k = 1 on |10>, |01>.
* Input 11y: <f_k|y> G'_k|11> = <f_k|ȳ> |11>. If <f_k|y> = 0, then <f_k|ȳ> = 0, so f_k = 0, which is impossible.
  So <f_k|y> ≠ 0 for both k, neither f_k is a computational vector, and G'_k|11> = g_k|11>.
  Then D := Σ_k g_k P_k satisfies D|y> = |ȳ> ⊥ |y>, so D is not ∝ I and g_0 ≠ g_1.
* Input 00x: <f_k|x> ≠ 0 (f_k is not computational), so G'_k|00> = |00>.

Hence G'_0†G'_1 = diag(1,1,1, ḡ_0 g_1) with ḡ_0 g_1 ≠ 1. The eigenvalue 1 has multiplicity exactly 3.
By Lemma 0, G'_0†G'_1 is a phase times W_R† X_r W_R (spectrum c·{1,1,−1,−1}) if deg t = 1, or a phase times I if deg t = 0.
Neither spectrum has a value of multiplicity exactly 3. Contradiction. ∎

## Proposition B: a (or b) cannot have degree ≤ 1 when k ≤ 3

By S1 it suffices to treat a. Lemmas 0 and 1 give U = Σ_j |j><j| ⊗ G'_j. Write M := G'_0†G'_1, which is not a product by Lemma 1.

**B1 (deg a = 0).** Then G_0 = G_1, so M ∝ I is a product. Contradiction.

**B2 (deg a = 1, counting).** Let a's CNOT act on (a, r). Lemma 0 gives {G'_0, G'_1} = {c VW, c' V X_r W}, with V = V_R and W = W_R.
In either assignment, G'_1G'_0† ∝ V X_r V† and G'_0†G'_1 ∝ W† X_r W. Both are non-products by Lemma 1.
So neither V nor W is a local gate, and each contains at least one CNOT. Since their counts add up to k − 1, we get k ≥ 3.
**This excludes k ≤ 2.** For k = 3, V and W each contain exactly one CNOT.

**B3 (spectrum).** M ∝ W†X_rW has spectrum c{1,1,−1,−1}. From Lemma 1, M has spectrum {1, h̄} ∪ {√s, −√s}.
The values ±√s are distinct, so they are c and −c. Then {1, h̄} = {c, −c}, which gives c = ±1, s = 1 and h = −1.
So U is uniquely determined:
G'_0 = I − 2|0x̄><0x̄|, G'_1 = CX(b,t), and U = CCX · (I − 2|00x̄><00x̄|).
By S2 (which preserves the topology) we may take x = 0. Then
K := G'_1G'_0† = G'_0†G'_1 = |0><0|_b ⊗ Z + |1><1|_b ⊗ X = ½(I⊗X + I⊗Z − Z⊗X + Z⊗Z).
K is Hermitian with K² = I, and its operator Schmidt rank across b|t is 2 (`tozsamosci.out`). Since K and V X_r V† are both Hermitian unitaries, V X_r V† = εK and W†X_rW = ε'K with ε, ε' = ±1.

**B4 (r = t).** Write V = A·CX(b→t)·B with A and B local. If r = b, then B X_b B† = n·σ on b, and
CX(n·σ_b)CX = (n_xX + n_yY)⊗X + n_z Z⊗I. This operator has no I⊗σ_j (j ≥ 1) component.
Conjugation by A = A_b⊗A_t maps span{I⊗σ_j} into itself, so V X_b V† has no such component either.
But K has the component I⊗X with coefficient ½. Contradiction. So r = t, i.e. a's CNOT is on the pair (a,t).

**B5 (the (1,1) split).** Let V_0 := |0><0|_b⊗H + |1><1|_b⊗I. Then V_0 X_t V_0† = K (exact, `tozsamosci.out`).
Let G_A ∈ {G'_0, G'_1} be the block with VW = λG_A, and set N := V_0†G_A V_0.
Exactly: N_A = |0><0|⊗X + |1><1|⊗I if G_A = G'_0, and N_B = CX(b,t) if G_A = G'_1. Both are Hermitian.

Put C := V_0†V. Then C X_t C† = εX_t, so C = C'Z_t^δ with C' in the commutant of X_t. The commutant is
{P⊗|+><+| + Q⊗|−><−| : P, Q ∈ U(2)}.
From W = λ̄ V†G_A we get W = λ̄ C† N V_0†, so cost(W) = cost(W†) = cost(V_0 N C).
(Here cost(·) is the minimal number of CNOTs, and it is invariant under multiplying by local gates on either side.)
The conditions cost(V_0C) ≤ 1 and cost(V_0NC) ≤ 1 are unchanged when C is multiplied on the right by a local gate in the commutant.
So we may take δ = 0 and C = C_Q := I⊗|+><+| + Q⊗|−><−| with Q ∈ SU(2).
In the ± basis N_A = I⊗|+><+| − Z⊗|−><−| and N_B = I⊗|+><+| + Z⊗|−><−|, so N C_Q = C_{∓ZQ} = C_{ZQ}·(I⊗X)^{0 or 1}.
Conjugating by the local H_t gives H_tV_0H_t = V_0 and H_tC_MH_t = I⊗|0><0| + M⊗|1><1|. Hence:

> k = 3 with deg a = 1 requires some Q ∈ SU(2) with cost(G(Q)) ≤ 1 and cost(G(ZQ)) ≤ 1, where
> G(M) := (|0><0|_b⊗H_t + |1><1|_b⊗I)(I_b⊗|0><0|_t + M_b⊗|1><1|_t).

cost ≤ 1 means local or locally equivalent to CNOT. By Makhlin's theorem this holds iff (G1, G2) ∈ {(1,3), (0,1)}.
Exact symbolic computation (`lemat_11.py`, with Q = [[p+iq, −r+is],[r+is, p−iq]] and p²+q²+r²+s² = 1) gives:
* G(Q): G1 = q²/2, G2 = 1 + q² − r² − s²;
* G(ZQ): G1 = p²/2, G2 = 2 − q² − 2r² − 2s².

For G(Q), G1 ≤ ½ forces G1 = 0, so q = 0. Then G2 = 1 forces r = s = 0, so Q = ±I and p² = 1.
But then G1(G(ZQ)) = ½ ∉ {0, 1}. So no Q works. ∎

## Proposition C: the triangle topology is impossible (k = 3)

**Reduction of orders.** The 6 orders of (ab, at, bt) reduce, via S1 and S3, to the two orders whose last CNOT is on bt.
For example, (bt,ab,at) = S3(at,ab,bt), (bt,at,ab) = S3(ab,at,bt), (ab,bt,at) = S1(ab,at,bt), and (at,bt,ab) = S1 S3(ab,at,bt).
These maps change (x,y), but the argument below holds for all four (x,y).
In both remaining orders, a is touched only by CNOTs 1 and 2 (partners r1 then r2, with {r1, r2} = {b, t}). So
U = (V_a⊗V_R) CX(a,r2) (M_a⊗M_b⊗M_t) CX(a,r1) (W_a⊗W_b⊗W_t), with V_R an arbitrary unitary on bt.
Expanding both CNOTs:
Y := (V_a⊗V_R)†U = Σ_{k,l} m_{kl} |k><w_l|_a ⊗ F_l ⊗ S_k.
Here m = M_a, <w_l| = <l|W_a, F_l acts on r1 with F_1 = F_0R_1 (R_1 = W†XW, traceless, so I and R_1 are linearly independent), and S_k acts on r2 with S_1 = X S_0.
Define G_{k,j} := Σ_l m_{kl}<w_l|j> F_l = F_0(m_{k0}<w_0|j> I + m_{k1}<w_1|j> R_1).
Validity means that for z ∈ L, Y|z> = u_{z_a} ⊗ Φ_z with u_j := V_a†|j> and Φ_z := V_R†|o_z>, while
Y|z> = Σ_k |k> ⊗ G_{k,z_a}|z_{r1}> ⊗ S_k|z_{r2}>.
So for each k: (**) G_{k,z_a}|z_{r1}> ⊗ S_k|z_{r2}> = u_{z_a,k} Φ_z.

**(i) u_j computational (V_a diagonal or anti-diagonal), u_j ∝ |π(j)>.** For k' ≠ π(j), (**) gives G_{k',j}|z_{r1}> = 0 for every z ∈ L with z_a = j.
Both values of z_{r1} occur (table), so G_{k',j} = 0. Then U = Σ_j |j><j| ⊗ G_j with G_j ∝ V_R(G_{π(j),j} ⊗ S_{π(j)}).
So G_0†G_1 ∝ (G_{π(0),0}⊗S_{π(0)})†(G_{π(1),1}⊗S_{π(1)}) is a product operator.
U has the form required by Lemma 1, whose computation gives G'_j ∝ G_j and says G'_0†G'_1 is not a product. Contradiction.

**(ii) u_j not computational (u_{j,0}, u_{j,1} ≠ 0).** By (**), S_0|z_{r2}> and S_1|z_{r2}> = XS_0|z_{r2}> are parallel, so S_0|c> is an X eigenvector.
This holds for c = 0, 1, since both values occur. Hence XS_0|c> = σ_cS_0|c> with σ_1 = −σ_0, and Φ_z ∝ (vector on r1) ⊗ S_0|z_{r2}>.
* Order (ab, at, bt), with r1 = b and r2 = t. The inputs 11y and 01ȳ have the same output bt = 1ȳ, so Φ_{11y} = Φ_{01ȳ}.
  Their t-factors are S_0|y> and S_0|ȳ>, which are orthogonal. Contradiction.
* Order (at, ab, bt), with r1 = t and r2 = b. (**) gives σ_{z_b}^k G_{k,z_a}|z_t> = u_{z_a,k} φ_z.
  The inputs 100 and 101 give G_{1,1} = ρG_{0,1} with ρ = u_{1,1}/(σ_0u_{1,0}).
  The input 11y then gives (σ_1 − σ_0) ρ G_{0,1}|y> = 0 with ρ ≠ 0 (because u_{1,1} ≠ 0) and σ_1 = −σ_0, so G_{0,1}|y> = G_{1,1}|y> = 0 and Y|11y> = 0.
  This is impossible for a unitary. ∎

## Conclusion
k ≤ 2: some qubit has degree ≤ 1, so Proposition A or B applies. k = 3: degree ≤ 1 (Propositions A, B) or triangle (Proposition C).
Hence no circuit with ≤ 3 CNOTs exists for any (x,y), and **c*(Q ∪ {00x,11y}) ≥ 4 is proven**.
A by-product: with deg a = 1, the only candidate U is CCX·(I − 2|00x̄><00x̄|). (It is not needed for the theorem; the referee V_P4a found numerically a 4-CNOT circuit for it with CNOT order (at,at,bt,at), loss 8e-10, see weryfikacja/V_P4a/koszt_uboczny.out.)
Numerical corroboration (not part of the proof): koszt.jsonl reports the best 3-CNOT loss 0.0674 > 0 for these supports.

## How to check
1. Read Lemmas 0–1 and Propositions A–C. Everything except B5 is hand linear algebra on 2×2 and 4×4 blocks.
2. Run `~/anaconda3/envs/ml/bin/python tozsamosci.py`. It checks U|_L = CCX|_L for all (x,y), K = G'_0†G'_1, K² = I, the Pauli coefficients of K, operator Schmidt rank 2, V_0X_tV_0† = K, and the forms of N_A and N_B.
3. Run `lemat_11.py`. It computes the Makhlin invariants of G(Q) and G(ZQ) symbolically, and as a sanity check gives CNOT → (0,1) and I → (1,3).
   The final step can also be checked by hand. The Makhlin invariants are standard, but a referee may prefer the Shende–Bullock–Markov criterion (cost ≤ 1 iff χ[γ(U)] = (x²+1)² or γ = ±I) as an independent check.
