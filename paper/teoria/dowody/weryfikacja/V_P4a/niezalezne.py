# Independent float sanity checks (not proof): Makhlin closed forms, SBM criterion, by-product cost.
import numpy as np, itertools
from scipy.optimize import minimize
rng=np.random.default_rng(1)
I2=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1]); H=np.array([[1,1],[1,-1]])/np.sqrt(2)
P0=np.diag([1,0]); P1=np.diag([0,1])
B=np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])/np.sqrt(2)
def mak(U):
    UB=B.conj().T@U@B; m=UB.T@UB; d=np.linalg.det(U)
    return np.trace(m)**2/(16*d), (np.trace(m)**2-np.trace(m@m))/(4*d)
def sbm_le1(U):  # Shende-Bullock-Markov: cost<=1 iff chi(gamma)=(x^2+1)^2 or gamma=+-I (U in SU(4))
    U=U/complex(np.linalg.det(U))**0.25; g=U@np.kron(Y,Y)@U.T@np.kron(Y,Y)
    return min(np.linalg.norm(g-np.eye(4)),np.linalg.norm(g+np.eye(4)),np.linalg.norm(np.poly(g)-np.array([1,0,2,0,1])))
V0=np.kron(P0,H)+np.kron(P1,I2)
G=lambda M: V0@(np.kron(I2,P0)+np.kron(M,P1))
err=0
for _ in range(2000):
    v=rng.normal(size=4); p,q,r,s=v/np.linalg.norm(v)
    Q=np.array([[p+1j*q,-r+1j*s],[r+1j*s,p-1j*q]])
    g1,g2=mak(G(Q)); h1,h2=mak(G(Z@Q))
    err=max(err,abs(g1-q*q/2),abs(g2-(1+q*q-r*r-s*s)),abs(h1-p*p/2),abs(h2-(2-q*q-2*r*r-2*s*s)))
print("max deviation of closed-form Makhlin invariants:",err)
# min over Q of max(SBM-distance of G(Q), of G(ZQ)) -- should be bounded away from 0
def f(v):
    p,q,r,s=v/np.linalg.norm(v); Q=np.array([[p+1j*q,-r+1j*s],[r+1j*s,p-1j*q]])
    return max(sbm_le1(G(Q)),sbm_le1(G(Z@Q)))
best=min(minimize(f,rng.normal(size=4),method='Nelder-Mead').fun for _ in range(200))
print("min_Q max(SBM dist G(Q), G(ZQ)):",best)
print("SBM sanity CNOT,I,SWAP:",sbm_le1(np.kron(P0,I2)+np.kron(P1,X)),sbm_le1(np.eye(4)),sbm_le1(np.eye(4)[[0,2,1,3]]))
