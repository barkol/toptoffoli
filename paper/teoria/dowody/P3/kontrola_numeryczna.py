"""NOT part of the proof: floating-point cross-check that 2-CNOT circuits (all 6 chains + same-pair)
fail on 4-string full-Pi supports but succeed on a 3-string class-2 support."""
import numpy as np, itertools
from scipy.optimize import minimize
rng = np.random.default_rng(1)
I2 = np.eye(2); Z = np.diag([1, -1])
def u3(p):
    a, b, c = p
    return np.array([[np.cos(a/2), -np.exp(1j*c)*np.sin(a/2)],
                     [np.exp(1j*b)*np.sin(a/2), np.exp(1j*(b+c))*np.cos(a/2)]])
def kron3(A, B, C): return np.kron(np.kron(A, B), C)
def op1(q, M): l = [I2]*3; l[q] = M; return kron3(*l)
def cz(p, q):
    d = np.ones(8)
    for s in range(8):
        bits = [(s >> 2) & 1, (s >> 1) & 1, s & 1]
        if bits[p] and bits[q]: d[s] = -1
    return np.diag(d)
def idx(x): return 4*x[0] + 2*x[1] + x[2]
ccx = lambda x: (x[0], x[1], x[2] ^ (x[0] & x[1]))
def build(p, pairs):
    k = 0; U = np.eye(8)
    def layer():
        nonlocal k
        M = kron3(u3(p[k:k+3]), u3(p[k+3:k+6]), u3(p[k+6:k+9])); k += 9; return M
    U = layer()
    for (a, b) in pairs: U = layer() @ cz(a, b) @ U
    return U
def cost(p, pairs, L):
    U = build(p, pairs)
    amps = np.array([U[idx(ccx(x)), idx(x)] for x in L])
    return len(L) - abs(amps.sum())   # 0 iff all amplitudes equal a common unit phase
def best(pairs, L, n=30):
    m = 9*(len(pairs)+1); b = 9
    for _ in range(n):
        r = minimize(cost, rng.uniform(0, 2*np.pi, m), args=(pairs, L), method='BFGS')
        b = min(b, r.fun)
    return b
configs = {'same(a,b)': [(0,1),(0,1)], 'same(a,t)': [(0,2),(0,2)], 'same(b,t)': [(1,2),(1,2)]}
for i, j, k in itertools.permutations(range(3)):
    configs['chain %s-%s-%s' % ('abt'[i], 'abt'[j], 'abt'[k])] = [(i, j), (j, k)]
for t in [(0,0,0,0), (0,1,1,0), (1,0,0,1), (0,0,0,1)]:
    L = [(0,0,t[0]), (0,1,t[1]), (1,0,t[2]), (1,1,t[3])]
    print('L targets', t, {c: round(best(p, L), 4) for c, p in configs.items()})
L3 = [(0,1,0), (1,0,0), (1,1,0)]
print('control 3-string', {c: round(best(p, L3), 6) for c, p in configs.items()})
