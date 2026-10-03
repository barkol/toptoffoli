"""Exact finite enumeration for the proof of (B3) in DOWOD.md.
Qubit indices: 0=a, 1=b, 2=t. A string is a tuple (a,b,t). CCX(a,b,t)=(a,b,t^(a&b)).
All checks are over finite sets of bits -- no floating point anywhere.
"""
from itertools import product, permutations
Q = {0: 'a', 1: 'b', 2: 't'}
ccx = lambda x: (x[0], x[1], x[2] ^ (x[0] & x[1]))

def pbij(pairs):
    """True iff the relation {(u,v)} is a partial bijection (well defined + injective)."""
    f, g = {}, {}
    for u, v in pairs:
        if f.setdefault(u, v) != v or g.setdefault(v, u) != u:
            return False
    return True

def rel(L, S, cond=lambda x, y: True):
    return [(tuple(x[q] for q in S), tuple(y[q] for q in S))
            for x in L for y in [ccx(x)] if cond(x, y)]

def lemma1(L, B):          # U = V_A (x) W_B possible only if both relations are partial bijections
    A = [q for q in range(3) if q not in B]
    return pbij(rel(L, A)) and pbij(rel(L, B))

def CB(L, i, j, k):        # Lemma 2: necessary if beta (input gate on first end i) is non-monomial
    a = all(pbij(rel(L, [i], lambda x, y, c=c: x[j] == c)) for c in (0, 1))
    return a and pbij(rel(L, [j, k]))

def Cp(L, i, j, k):        # Lemma 2': necessary if nu (output gate on second end k) is non-monomial
    a = all(pbij(rel(L, [k], lambda x, y, c=c: y[j] == c)) for c in (0, 1))
    return a and pbij(rel(L, [i, j]))

def core_ends(L, i, k):    # Lemma 3, first part: x_i<->y_i and x_k<->y_k partial bijections
    return pbij(rel(L, [i])) and pbij(rel(L, [k]))

def core_inner(L, i, k):   # Lemma 3, second part (t is the middle qubit): exact inner-product test
    # For each value c of x_i the two strings x (x_k=0), x' (x_k=1) give |<x_t|x'_t>| = |<y_t|R|y'_t>|.
    # R is a traceless reflection.  Returns the list of constraints; infeasible iff one group forces
    # R diagonal and the other demands |R_offdiag-or-diag| incompatible.
    cons = []
    for c in (0, 1):
        g = sorted([x for x in L if x[i] == c], key=lambda x: x[k])
        x0, x1 = g; y0, y1 = ccx(x0), ccx(x1)
        lhs = int(x0[2] == x1[2])                      # |<x0_t|x1_t>|
        entry = 'diag' if y0[2] == y1[2] else 'off'    # which entry of R is |<y0_t|R|y1_t>|
        cons.append((lhs, entry))
    # |R_diag|=1 <=> R=+-Z <=> |R_off|=0 ; |R_diag|=0 <=> |R_off|=1. Normalize to "R diagonal?"
    need = set()
    for lhs, entry in cons:
        need.add((lhs == 1) == (entry == 'diag'))      # True means R must be diagonal
    return len(need) == 1                              # feasible only if both groups agree

def supports4():
    for t in product((0, 1), repeat=4):
        yield t, [(0, 0, t[0]), (0, 1, t[1]), (1, 0, t[2]), (1, 1, t[3])]

def run(Lgen, label):
    print(f'=== {label} ===')
    surv_total = 0
    for t, L in Lgen():
        out = []
        # k<=1 CNOT and 2 CNOTs on the same pair: product across some single-qubit cut
        prod_ok = [Q[B[0]] for B in ([0], [1], [2]) if lemma1(L, B)]
        # 2-CNOT chain i-j then j-k (i first), all 6 ordered triples
        chain_ok = []
        for i, j, k in permutations(range(3)):
            nm = Q[i] + Q[j] + Q[k]
            if CB(L, i, j, k):
                chain_ok.append(nm + ':beta-nonmonomial')
            if Cp(L, i, j, k):
                chain_ok.append(nm + ':nu-nonmonomial')
            if core_ends(L, i, k):
                if len(L) == 4 and j == 2 and core_inner(L, i, k):
                    chain_ok.append(nm + ':core')
                elif len(L) != 4:
                    chain_ok.append(nm + ':core-ends-pass')
                elif j != 2:
                    raise AssertionError('core ends pass with t at an end?!')
        surv = prod_ok + chain_ok
        surv_total += len(surv)
        print(''.join(map(str, t)), 'product-cut survivors:', prod_ok, '| chain survivors:', chain_ok)
    print('TOTAL surviving branches:', surv_total)
    return surv_total

if __name__ == '__main__':
    n = run(supports4, 'B3: L={00t0,01t1,10t2,11t3}, all 16 target choices')
    assert n == 0
    print('B3 VERIFIED: every branch of every case is excluded.\n')
    # Non-vacuity control: class-2 supports (Pi={01,10,11}) are 2-CNOT feasible; the tests must NOT
    # exclude the chain with t in the middle there.
    def supports3():
        for t in product((0, 1), repeat=3):
            yield t, [(0, 1, t[0]), (1, 0, t[1]), (1, 1, t[2])]
    m = run(supports3, 'CONTROL: L={01t1,10t2,11t3} (c*=2, must have survivors)')
    assert m > 0
    print('Control OK: the necessary conditions are not vacuous.')
