# Exact identities used in DOWOD.md (x = 0; the case x = 1 follows from the X_t symmetry).
import sympy as sp
I=sp.I; kron=sp.kronecker_product
H=sp.Matrix([[1,1],[1,-1]])/sp.sqrt(2); X=sp.Matrix([[0,1],[1,0]]); Y=sp.Matrix([[0,-I],[I,0]]); Z=sp.diag(1,-1); E=sp.eye(2)
P0=sp.diag(1,0); P1=sp.diag(0,1)
# two-qubit operators on (b,t), basis 00,01,10,11
for x in (0,1):
    for y in (0,1):
        # G0: identity on 10,11,0x, sign h=-1 on 0xbar; G1: CX(b,t) (s=1)
        G0=sp.eye(4); G0[2*0+(1-x),2*0+(1-x)]=-1
        G1=kron(P0,E)+kron(P1,X)
        U=kron(P0,G0)+kron(P1,G1)        # qubits (a,b,t)
        CCX=sp.eye(8); CCX[6,6]=CCX[7,7]=0; CCX[6,7]=CCX[7,6]=1
        L=[0b100,0b101,0b010,0b011,0b000|x,0b110|y]
        ok=all(U[:,z]==CCX[:,z] for z in L)
        K=G1*G0.H
        print(f"x={x} y={y}: U agrees with CCX on L: {ok};  K==G0^dag G1: {K==G0.H*G1};  K Hermitian: {K==K.H};  K^2=I: {K*K==sp.eye(4)}")
        # Pauli coefficients of K
        P={'I':E,'X':X,'Y':Y,'Z':Z}
        coef={a+b:sp.nsimplify(sp.simplify((kron(P[a],P[b])*K).trace()/4)) for a in P for b in P}
        print("    K Pauli coefficients (nonzero):",{k:v for k,v in coef.items() if v!=0})
        # realignment rank (operator Schmidt rank across b|t)
        R=sp.Matrix(4,4,lambda i,j: K[2*(i//2)+(j//2), 2*(i%2)+(j%2)])
        print("    operator Schmidt rank of K across b|t:",R.rank())
G0=sp.diag(1,-1,1,1); G1=kron(P0,E)+kron(P1,X); K=G1*G0
V0=kron(P0,H)+kron(P1,E); Xt=kron(E,X)
print("x=0: V0 X_t V0^dag == K:", sp.simplify(V0*Xt*V0.H-K)==sp.zeros(4))
print("     V0^dag G0 V0 == |0><0| x X + |1><1| x I:", sp.simplify(V0.H*G0*V0-(kron(P0,X)+kron(P1,E)))==sp.zeros(4))
print("     V0^dag G1 V0 == CX(b,t):", sp.simplify(V0.H*G1*V0-G1)==sp.zeros(4))
Ht=kron(E,H)
print("     H_t V0 H_t == V0:", sp.simplify(Ht*V0*Ht-V0)==sp.zeros(4))
print("     H_t N_A H_t == diag(1,-1,1,1):", sp.simplify(Ht*(kron(P0,X)+kron(P1,E))*Ht-sp.diag(1,-1,1,1))==sp.zeros(4))
print("     H_t N_B H_t == CZ:", sp.simplify(Ht*G1*Ht-sp.diag(1,1,1,-1))==sp.zeros(4))
