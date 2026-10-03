"""Exact exhaustive check of the necessary conditions (Lemmas 2-4 of DOWOD.md)
for <=2-CNOT circuits on supports with one string per control value.
Pure integer/boolean logic: no floating point anywhere.
Qubit indices: a=0, b=1, t=2. A string is a tuple (a,b,t)."""
import itertools

NAMES = "abt"
def ccx(x):
    a, b, t = x
    return (a, b, t ^ (a & b))

def untouched_ok(L, q):
    """U = V_q (x) W_rest. Necessary: x_q -> f_q(x) and x_rest -> f_rest(x)
    are well defined and injective on L (Lemma 1)."""
    rest = [r for r in range(3) if r != q]
    for part in ([q], rest):
        m = {}
        for x in L:
            k = tuple(x[r] for r in part); v = tuple(ccx(x)[r] for r in part)
            if k in m and m[k] != v: return False      # not well defined
            m[k] = v
        if len(set(m.values())) != len(m): return False # not injective
    return True

class ParityUF:
    """union-find with Z2 labels: rel 0 means 'same ray', 1 means 'orthogonal ray'"""
    def __init__(s, n): s.p = list(range(n)); s.r = [0]*n
    def find(s, x):
        if s.p[x] == x: return x, 0
        root, par = s.find(s.p[x]); s.p[x] = root; s.r[x] ^= par
        return root, s.r[x]
    def add(s, x, y, rel):
        rx, px = s.find(x); ry, py = s.find(y)
        if rx == ry: return (px ^ py) == rel   # False = odd cycle
        s.p[rx] = ry; s.r[rx] = px ^ py ^ rel; return True

def chain_ok(L, i, j, k):
    """U = (I_i (x) V2_jk)(V1_ij (x) I_k). Necessary conditions of Lemmas 2-3;
    satisfiability decided by Lemma 4 (balance of signed graph)."""
    n = len(L); F = [ccx(x) for x in L]; uf = ParityUF(n)
    for p, q in itertools.combinations(range(n), 2):
        x, y, fx, fy = L[p], L[q], F[p], F[q]
        # V1 conditions
        if (x[i], x[j]) == (y[i], y[j]):
            if fx[i] != fy[i]: return False, "V1 not well defined"
            if not uf.add(p, q, 0): return False, "odd cycle"
        elif fx[i] == fy[i]:
            if not uf.add(p, q, 1): return False, "odd cycle"
        # V2 conditions
        same_out = (fx[j], fx[k]) == (fy[j], fy[k])
        if x[k] != y[k]:
            if same_out: return False, "V2 maps orthogonal inputs to same output"
        else:
            if not uf.add(p, q, 0 if same_out else 1): return False, "odd cycle"
    return True, "consistent"

def report(supports, label):
    print(f"== {label}: {len(supports)} supports")
    bad = 0
    for L in supports:
        ok0 = [NAMES[q] for q in range(3) if untouched_ok(L, q)]
        okc = []
        for perm in itertools.permutations(range(3)):
            ok, why = chain_ok(L, *perm)
            if ok: okc.append("".join(NAMES[r] for r in perm))
        if ok0 or okc: bad += 1
        print(" L=", ["".join(map(str, x)) for x in L],
              "| untouched-qubit OK:", ok0 or "none", "| chains OK:", okc or "none")
    print(f"   supports admitting some <=2-CNOT structure: {bad}")

# (B3): one string per control value 00,01,10,11, all 16 target patterns
B3 = [[(0,0,t[0]), (0,1,t[1]), (1,0,t[2]), (1,1,t[3])]
      for t in itertools.product((0,1), repeat=4)]
report(B3, "B3 supports")
# positive control: Pi={01,10,11} (c*=2 claimed): should admit a chain, but no untouched qubit
PC = [[(0,1,t[0]), (1,0,t[1]), (1,1,t[2])] for t in itertools.product((0,1), repeat=3)]
report(PC, "positive control Pi={01,10,11}")
# positive control 2: Pi={10,11} (c*=1): should admit 1-CNOT structure? (untouched qubit)
PC2 = [[(1,0,t[0]), (1,1,t[1])] for t in itertools.product((0,1), repeat=2)]
report(PC2, "positive control Pi={10,11}")
