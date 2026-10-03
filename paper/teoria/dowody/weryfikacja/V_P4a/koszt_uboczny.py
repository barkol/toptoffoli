# Numerical (non-rigorous) check of the by-product claim: U* = CCX (I - 2|00xb><00xb|) has CNOT cost 4.
import numpy as np, itertools
from scipy.optimize import minimize
rng=np.random.default_rng(0)
def u3(t,f,l): return np.array([[np.cos(t/2),-np.exp(1j*l)*np.sin(t/2)],[np.exp(1j*f)*np.sin(t/2),np.exp(1j*(f+l))*np.cos(t/2)]])
def cx(c,t):
    M=np.zeros((8,8))
    for z in range(8):
        bits=[(z>>(2-i))&1 for i in range(3)]
        if bits[c]: bits[t]^=1
        M[bits[0]*4+bits[1]*2+bits[2],z]=1
    return M
CCX=np.eye(8); CCX[[6,7]]=CCX[[7,6]]
x=0; D=np.eye(8); D[0b000|(1-x),0b000|(1-x)]=-1; Ut=CCX@D
pairs=[(0,1),(0,2),(1,2)]
def circ(th,seq):
    th=th.reshape(len(seq)+1,3,3); U=np.eye(8)
    for i in range(len(seq)+1):
        L=np.kron(np.kron(u3(*th[i,0]),u3(*th[i,1])),u3(*th[i,2])); U=L@U
        if i<len(seq): U=cx(*seq[i])@U
    return U
def loss(th,seq): return 1-abs(np.trace(Ut.conj().T@circ(th,seq)))/8
for k in (3,4):
    best=1
    for seq in itertools.product(pairs,repeat=k):
        for _ in range(4):
            r=minimize(loss,rng.uniform(0,2*np.pi,9*(k+1)),args=(seq,),method='BFGS')
            best=min(best,r.fun)
            if best<1e-9: break
        if best<1e-9: print(k,"found",seq,best); break
    print("k=",k,"best loss",best)
