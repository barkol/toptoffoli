"""Exact exhaustive check of the combinatorial content of the proof of (B1), (B2) and c*>=2 on classes 3,4,6.
Pure finite enumeration over all 255 nonempty L subset {0,1}^3 (strings 'abt'); no floating point.
Lemma F (factorization obstruction): if U = V (x) W, W on qubit k, satisfies U|x> ~ CCX|x> on L, then
  R_k   = {(x_k, f(x)_k)}       and   R_rest = {(x_rest, f(x)_rest)}   (x in L, f = CCX on bits)
are graphs of partial injections (functional + injective). We call qubit k 'splittable' if both are.
"""
import itertools, json, collections, sys

STR = [f"{a}{b}{t}" for a in '01' for b in '01' for t in '01']
def f(x):  # CCX on bit strings 'abt'
    a, b, t = x
    return x if (a, b) != ('1', '1') else a + b + ('1' if t == '0' else '0')
def partial_injection(pairs):
    fwd, bwd = {}, {}
    for u, v in pairs:
        if fwd.setdefault(u, v) != v or bwd.setdefault(v, u) != u: return False
    return True
POS = {'a': 0, 'b': 1, 't': 2}
def splittable(L, k):
    i = POS[k]; rest = [j for j in range(3) if j != i]
    Rk = [(x[i], f(x)[i]) for x in L]
    Rr = [(''.join(x[j] for j in rest), ''.join(f(x)[j] for j in rest)) for x in L]
    return partial_injection(Rk) and partial_injection(Rr)
def product_ok(L):  # 0 CNOT: U = A(x)B(x)C; necessary&sufficient: each single-bit relation a partial injection
    return all(partial_injection([(x[i], f(x)[i]) for x in L]) for i in range(3))
def Pi(L): return {x[:2] for x in L}
def claimed(L):
    P = Pi(L); Q = {'100', '101', '010', '011'}
    if '11' not in P or P == {'11'}: return 0
    if '01' not in P or '10' not in P: return 1
    if P == {'01', '10', '11'}: return 2
    if set(L) == set(STR): return 6
    if not Q <= set(L): return 3
    return 4

cnt = collections.Counter(); fails = []
allL = [L for r in range(1, 9) for L in itertools.combinations(STR, r)]
for L in allL:
    c = claimed(L); P = Pi(L)
    t_split = splittable(L, 't'); sp = {k: splittable(L, k) for k in 'abt'}
    # (i) Lemma T: 11 in Pi and Pi != {11}  <=>  t not splittable   (B1 obstruction, also for 0 CNOT)
    if (('11' in P and P != {'11'}) == t_split): fails.append(('T', L))
    # (ii) Lemma A/B: a not splittable <=> {01,11} subset Pi ; b not splittable <=> {10,11} subset Pi
    if (({'01', '11'} <= P) == sp['a']): fails.append(('A', L))
    if (({'10', '11'} <= P) == sp['b']): fails.append(('B', L))
    # (iii) product (0 CNOT) feasible  <=>  claimed class 0
    if product_ok(L) != (c == 0): fails.append(('P', L))
    # (iv) no qubit splittable  <=>  claimed class >= 2   (=> c* >= 2 on classes 2,3,4,6)
    if (not any(sp.values())) != (c >= 2): fails.append(('S', L))
    cnt[(c, tuple(k for k in 'abt' if sp[k]))] += 1
print("supports enumerated:", len(allL))
print("class, splittable qubits -> #L")
for key in sorted(cnt): print("  ", key, cnt[key])
print("FAILURES:", fails if fails else "none")

# monotonicity reduction: every L with Pi = {00,01,10,11} contains a 3-subset L' with Pi(L') = {01,10,11}
bad = [L for L in allL if Pi(L) == {'00', '01', '10', '11'}
       and not any(Pi(S) == {'01', '10', '11'} for S in itertools.combinations(L, 3))]
print("full-Pi supports without a class-2 3-subset:", len(bad))

# cross-check against the numerical data (min k with loss < 1e-8), where present
try:
    d = collections.defaultdict(dict)
    for l in open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'koszt.jsonl')):
        r = json.loads(l); d[tuple(r['labels'])][r['k']] = r['best_loss']
    mism = 0; n = 0
    for lab, ks in d.items():
        ok = [k for k, v in ks.items() if v < 1e-8]
        if not ok: continue
        n += 1; num = min(ok); c = claimed(lab)
        lb = 0 if c == 0 else (1 if c == 1 else 2)   # what this proof gives
        if num < lb: mism += 1; print("  numeric below proven bound:", lab, num, lb)
    print(f"numeric cross-check: {n} supports with a feasible k; numeric c* below proven lower bound: {mism}")
except FileNotFoundError:
    print("numeric data not found; skipped")
sys.exit(1 if fails or bad else 0)
