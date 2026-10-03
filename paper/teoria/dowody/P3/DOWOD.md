# P3 — Proof of (B3): no circuit with ≤ 2 CNOTs on a minimal full-Π support

## Statement

Let L = {00t₀, 01t₁, 10t₂, 11t₃} (strings abt, any t₀..t₃ ∈ {0,1}; 16 supports).
There is no circuit U with k ∈ {0,1,2} CNOTs (any pairs, any directions) and arbitrary
single-qubit gates such that U|x⟩ = e^{iθ} CCX|x⟩ for all x ∈ L with one common θ.
Hence c*(L) ≥ 3, and by monotonicity c* ≥ 3 for every L with Π = {00,01,10,11}.

The proof is analytic; the only computer step is an exhaustive finite enumeration over
bits (no floating point): `sprawdz_B3.py` → `sprawdz_B3.out`. The analytic chain argument
of the other provers is not used. The common phase is never used either, so the proof
gives a slightly stronger result: no ≤2-CNOT circuit even with independent phases per string.

## Notation

y = CCX(x). For a set S of qubits and a subset M ⊆ L, the relation
Rel_S(M) = {(x_S, y_S) : x ∈ M} is a **partial bijection** (PB) if it is well defined
(x_S = x'_S ⇒ y_S = y'_S) and injective (y_S = y'_S ⇒ x_S = x'_S).
A single-qubit unitary is **monomial** (∈ D) if it is diagonal or antidiagonal. Equivalently, it maps
one computational basis vector to a phase times a basis vector. Then it does this for both, by orthogonality.
If g ∉ D, then g|c⟩ has two nonzero components for both c.

**Fact 0 (ν-PB).** If a unitary W on a set S of qubits satisfies W|x_S⟩ ∝ |y_S⟩ for all x ∈ M,
then Rel_S(M) is a PB. Proof: W is a function, so it is well defined. It is injective, and distinct basis
vectors are orthogonal while equal basis vectors are not.

**Fact T (target relation).** Rel_{t}(L) is never a PB. The string 11t₃ gives (t₃, ¬t₃). If another
string of L has target t₃, it gives (t₃, t₃), so the relation is not well defined. Otherwise all three others
give (¬t₃, ¬t₃), so it is not injective.

## Step 0: reduction to normal forms

Consecutive single-qubit gates merge. CNOT(c→t) = H_t CZ H_t, and CZ is symmetric, so the
direction is irrelevant and every CNOT becomes CZ on the same pair.
* k ≤ 1, or k = 2 on the same pair: some qubit r is untouched, so U = V_A ⊗ W_{r} with A the other two qubits.
* k = 2 on two different pairs: two distinct pairs of {a,b,t} share exactly one qubit j. Call i the
  qubit of the first CZ other than j, and k that of the second. Single-qubit gates between the two
  CZs act on i, j and k. The one on i commutes forward past CZ_jk, and the one on k commutes backward past CZ_ij.
  So the circuit has the form
  U = (α_i⊗μ_j⊗ν_k) · CZ_jk · ρ_j · CZ_ij · (β_i⊗γ_j⊗δ_k)      (★)
  with 7 arbitrary single-qubit unitaries. All 6 ordered triples (i,j,k) occur.

## Lemma 1 (product cut)
If U = V_A ⊗ W_B and U|x⟩ ∝ |y⟩ on L, then Rel_A(L) and Rel_B(L) are PBs.
*Proof.* V|x_A⟩⊗W|x_B⟩ ∝ |y_A⟩|y_B⟩ with nonzero factors. So V|x_A⟩ ∝ |y_A⟩ and W|x_B⟩ ∝ |y_B⟩, and Fact 0 applies. ∎

Application for B = {t}: Fact T. For B = {a} (A = {b,t}), take the strings 01t₁ and 11t₃. They give
((1,t₁),(1,t₁)) and ((1,t₃),(1,¬t₃)). This relation is not well defined if t₁ = t₃ and not injective if t₁ ≠ t₃.
B = {b} is the mirror case (strings 10t₂, 11t₃). **This excludes k = 0, k = 1, and k = 2 on the same pair.**
(The enumeration re-checks it: "product-cut survivors: []".)

## Lemma 2 (outer input gate non-monomial)
In (★), if β ∉ D then γ ∈ D, and the following hold:
 (a) for each c ∈ {0,1}, Rel_{i}({x∈L : x_j = c}) is a PB;
 (b) Rel_{j,k}(L) is a PB.
*Proof.* Fix x ∈ L. Let b = β|x_i⟩, g = γ|x_j⟩ and d = δ|x_k⟩. Then
CZ_jk ρ_j CZ_ij(b⊗g⊗d) = Σ_m b_m |m⟩_i ⊗ K(Z^m g), where K(w) = CZ_jk(ρw ⊗ d) is an injective linear map C²→C⁴.
This vector equals e^{iθ}(α†|y_i⟩)⊗(μ†|y_j⟩⊗ν†|y_k⟩), which is a product across i|jk.
Since β ∉ D, b₀ and b₁ are both nonzero. Rank one then forces K(g) ∥ K(Zg), so g ∥ Zg.
So g is a computational basis vector up to phase, and hence γ ∈ D.
Write γ|c⟩ = e^{iκ_c}|π(c)⟩. Then
U|x⟩ = e^{iκ}(αZ^{π(x_j)}β|x_i⟩) ⊗ G|x_j x_k⟩ with the fixed unitary G = (μ⊗ν)CZ_jk(ργ⊗δ).
Both factors are proportional to |y_i⟩ and |y_jk⟩ respectively. Fact 0 then applies with the fixed unitary
W_c = αZ^{π(c)}β on each class x_j = c, and with G on L. ∎

## Lemma 2′ (outer output gate non-monomial)
In (★), if ν ∉ D then μ ∈ D, and the following hold:
 (a) for each c, Rel_{k}({x∈L : y_j = c}) is a PB;
 (b) Rel_{i,j}(L) is a PB.
*Proof.* U† = (β†⊗γ†⊗δ†)·CZ_ij·ρ†·CZ_jk·(α†⊗μ†⊗ν†) maps |y⟩ to e^{-iθ}|x⟩ for y ∈ CCX(L). This has the
form (★) with the roles of i and k exchanged, and outer input gate ν†. Lemma 2's proof used only
"basis state ↦ phase·basis state", so it applies verbatim. PB is symmetric under inverting the relation. ∎

## Lemma 3 (core: β, ν ∈ D)
In (★), if β, ν ∈ D, then j = t and the circuit is impossible on L.
*Proof.* Write β|c⟩ = e^{i·}|π(c)⟩ and ν†|c⟩ = e^{i·}|π′(c)⟩. Then
U|x⟩ ∝ α|π(x_i)⟩ ⊗ (μ⊗ν)CZ_jk(w ⊗ d), with w = ρZ^{π(x_i)}γ|x_j⟩ and d = δ|x_k⟩.
(i) α|π(x_i)⟩ ∝ |y_i⟩ for all x, so Rel_{i}(L) is a PB (Fact 0).
(ii) CZ_jk(w⊗d) = w₀|0⟩d + w₁|1⟩Zd must be ∝ (μ†|y_j⟩)⊗|π′(y_k)⟩. Let e = |1−π′(y_k)⟩ and project
qubit k onto e. This gives w₀⟨e|d⟩ = 0 and w₁⟨e|Zd⟩ = ±w₁⟨e|d⟩ = 0. Since w ≠ 0, ⟨e|d⟩ = 0, so δ|x_k⟩ ∝ |π′(y_k)⟩.
Hence Rel_{k}(L) is a PB.
By Fact T, neither i nor k is t, so j = t and {i,k} = {a,b}. Then y_i = x_i and y_k = x_k, and
CZ_jk(w⊗|π′(x_k)⟩) = Z^{π′(x_k)}w ⊗ |π′(x_k)⟩. The t-factor condition becomes
  R^{n} V S^{m} |x_t⟩ ∝ |y_t⟩,   m = π(x_i), n = π′(x_k),
with V = μργ, R = μZμ† and S = γ†Zγ. R and S are traceless reflections (Hermitian, R² = I, eigenvalues ±1).
Fix c ∈ {0,1}. The strings with x_i = c are two strings x, x′, with x_k = 0 and x_k = 1, so n ≠ n′, while m is the same.
With W = VS^m we get W|x_t⟩ ∝ R^n|y_t⟩ and W|x′_t⟩ ∝ R^{n′}|y′_t⟩.
Unitarity and R^nR^{n′} = R give
  |⟨x_t|x′_t⟩| = |⟨y_t|R|y′_t⟩|.                         (♦)
Take i = a. The group a = 0 is {00t₀, 01t₁}, which CCX leaves unchanged, so (♦) reads δ_{t₀t₁} = |R_{t₀t₁}|.
If t₀ = t₁, then |R_diag| = 1. If t₀ ≠ t₁, then R_offdiag = 0. Either way R = ±Z.
The group a = 1 is {10t₂, 11t₃}, with outputs t₂ and ¬t₃. Here (♦) reads δ_{t₂t₃} = |R_{t₂,¬t₃}| = δ_{t₂,¬t₃} (because R = ±Z).
This is false for both values of t₂ and t₃.
For i = b, use the groups b = 0 {00t₀, 10t₂} and b = 1 {01t₁, 11t₃} with the same argument. ∎

## Lemma 4 (combinatorial exclusion of Lemmas 2 and 2′) — exhaustive enumeration
For all 16 choices of t₀..t₃ and all 6 ordered triples (i,j,k), conditions 2(a)∧2(b) fail and 2′(a)∧2′(b) fail.
So β ∈ D and ν ∈ D in every chain circuit, and Lemma 3 applies. The checks are finite operations on
bit tuples (`pbij`, `CB` and `Cp` in `sprawdz_B3.py`). The output reports "TOTAL surviving branches: 0".
For a referee, here are typical hand checks:
* j = t, i = a: 2(b) is Rel_{t,b}. It contains ((t₁,1),(t₁,1)) and ((t₃,1),(¬t₃,1)), which is not a PB.
* j = a, i = t: in 2(a), the class x_a = 1 gives (t₂,t₂) and (t₃,¬t₃), which is not a PB.
* j = a, i = b, k = t: 2(b) is Rel_{a,t}, with ((1,t₂),(1,t₂)) and ((1,t₃),(1,¬t₃)), which is not a PB.
In every case the obstruction comes from the pair of strings {10t₂, 11t₃}, the pair {01t₁, 11t₃}, or Fact T.

## Conclusion (case list)
| circuit class | excluded by |
|---|---|
| 0 CNOT | Lemma 1 (B = {t}) |
| 1 CNOT on {a,b} / {a,t} / {b,t} | Lemma 1 with B = {t} / {b} / {a} |
| 2 CNOT on the same pair (3 pairs, 4 directions each) | Lemma 1, as for 1 CNOT |
| 2 CNOT chain, 6 ordered (i,j,k), 4 directions each (absorbed into CZ) | Lemma 2/2′ + Lemma 4, so β, ν ∈ D, then Lemma 3 |

This holds for all 16 target choices. So c*(L) ≥ 3 for every minimal full-Π support, which is **(B3)**.

## Control (non-vacuity) and numeric cross-check (NOT part of the proof)
* `sprawdz_B3.py` also runs the 3-string class-2 supports {01,10,11} (where c* = 2).
  There, the chains a–t–b and b–t–a survive every necessary condition, as they must.
  This shows the conditions are not trivially false. In this case Lemma 3's group a = 0 has a single string, so it does not force R = ±Z.
* `kontrola_numeryczna.py` uses BFGS with 30 restarts on all 9 two-CNOT layouts.
  On 4-string supports, the minimum of the cost (4 − |Σ amplitudes|) is ≥ 0.3045, reached by chain a–t–b. All other layouts give ≥ 0.5858.
  On the control 3-string support the cost is 0.000000 for a–t–b and b–t–a only.
  This agrees exactly with the case structure of the proof.

## Status
Proven: (B3) in full, all 16 supports and all circuit classes. Open: nothing for B3.
Remark: the phase condition is never used. So even "U|x⟩ ∝ CCX|x⟩ with independent phases" needs 3 CNOTs.
This is a strengthening, and it is consistent with Song–Klappenecker.

Files: `sprawdz_B3.py`, `sprawdz_B3.out` (exact), `kontrola_numeryczna.py`, `kontrola_numeryczna.out` (guide only).
