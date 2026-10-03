# Exact (symbolic) check of the (1,1) case in the a-degree-1 topology.
# G(Q) = V0 * C_Q on qubits (b,t), basis |bt> = 00,01,10,11
#   V0  = |0><0|_b (x) H_t + |1><1|_b (x) I_t
#   C_Q = I_b (x) |0><0|_t + Q_b (x) |1><1|_t,   Q in SU(2)
# Claim: there is no Q in SU(2) with cost(G(Q)) <= 1 and cost(G(Z Q)) <= 1.
# cost <= 1  <=>  G locally equivalent to identity or CNOT  <=>  Makhlin invariants
#   (G1,G2) in {(1,3), (0,1)}   [Makhlin, QIP 1, 243 (2002); Zhang et al. PRA 67 042313]
import sympy as sp
p,q,r,s = sp.symbols('p q r s', real=True)   # Q = [[p+iq, -(r-is)],[r+is, p-iq]], p^2+q^2+r^2+s^2=1
I = sp.I
def kron(A,B): return sp.kronecker_product(A,B)
H = sp.Matrix([[1,1],[1,-1]])/sp.sqrt(2)
P0 = sp.Matrix([[1,0],[0,0]]); P1 = sp.Matrix([[0,0],[0,1]]); Id = sp.eye(2)
Zm = sp.Matrix([[1,0],[0,-1]])
Q = sp.Matrix([[p+I*q, -(r-I*s)],[r+I*s, p-I*q]])
V0 = kron(P0,H) + kron(P1,Id)
def G(Qm): return V0*(kron(Id,P0)+kron(Qm,P1))
# magic basis
B = sp.Matrix([[1,0,0,I],[0,I,1,0],[0,I,-1,0],[1,0,0,-I]])/sp.sqrt(2)
norm = {p**2: 1-q**2-r**2-s**2}
def inv(U):
    UB = B.H*U*B
    m = UB.T*UB
    d = sp.simplify(U.det())
    t1 = sp.expand(m.trace())
    t2 = sp.expand((m*m).trace())
    G1 = sp.simplify(sp.expand(t1**2/(16*d)).subs(norm))
    G2 = sp.simplify(sp.expand((t1**2-t2)/(4*d)).subs(norm))
    return sp.simplify(d.subs(norm)), G1, G2
for name,Qm in [("Q",Q),("ZQ",Zm*Q)]:
    d,G1,G2 = inv(G(Qm))
    print(name, " det =", d)
    print("   G1 =", sp.factor(G1))
    print("   G2 =", sp.factor(G2))
# sanity: CNOT and identity invariants
CX = kron(P0,Id)+kron(P1,sp.Matrix([[0,1],[1,0]]))
print("CNOT inv", inv(CX)[1:], " I inv", inv(sp.eye(4))[1:])
