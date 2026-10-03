import numpy as np, itertools, sys
from scipy.optimize import minimize
rng=np.random.default_rng(1)
I2=np.eye(2); X=np.array([[0,1],[1,0]])
def u3(p):
    a,b,c=p; return np.array([[np.cos(a/2),-np.exp(1j*c)*np.sin(a/2)],[np.exp(1j*b)*np.sin(a/2),np.exp(1j*(b+c))*np.cos(a/2)]])
def kron3(A,B,C): return np.kron(np.kron(A,B),C)
def cx(c,t):
    P0=np.diag([1,0]);P1=np.diag([0,1]); ops0=[I2]*3; ops1=[I2]*3
    ops0[c]=P0; ops1[c]=P1; ops1[t]=X
    return kron3(*ops0)+kron3(*ops1)
PAIRS=[(i,j) for i in range(3) for j in range(3) if i!=j]
def circ(p,seq):
    p=p.reshape(-1,3,3); U=kron3(*[u3(q) for q in p[0]])
    for n,(c,t) in enumerate(seq):
        U=kron3(*[u3(q) for q in p[n+1]])@cx(c,t)@U
    return U
idx=lambda s:int(s,2)
def f(s): return s if s[:2]!='11' else s[:2]+('1' if s[2]=='0' else '0')
def loss(p,seq,L):
    U=circ(p,seq); z=sum(U[idx(f(x)),idx(x)] for x in L)
    return 1-abs(z)/len(L)
def best(L,seq,starts):
    b=9
    for _ in range(starts):
        p0=rng.uniform(0,2*np.pi,3*3*(len(seq)+1))
        r=minimize(loss,p0,args=(seq,L),method='BFGS',options={'gtol':1e-12})
        b=min(b,r.fun)
        if b<1e-9: break
    return b
STR=[''.join(s) for s in itertools.product('01',repeat=3)]
out=[]
# 0 CNOT on all class>=1 minimal pairs
for L in [('11'+t,uvs) for t in '01' for uvs in STR if uvs[:2]!='11']:
    out.append(('k=0',L,best(L,[],40)))
# 1 CNOT on all class-2 minimal triples, all 6 ordered pairs
for s in itertools.product('01',repeat=3):
    L=('01'+s[0],'10'+s[1],'11'+s[2])
    for pr in PAIRS: out.append(('k=1 '+str(pr),L,best(L,[pr],25)))
# positive controls
out.append(('CTRL k=1 cx(0,2) class1',('110','010'),best(('110','010'),[(0,2)],25)))
out.append(('CTRL k=2 class2',('010','100','110'),min(best(('010','100','110'),s,25) for s in [[(0,2),(1,2)]])))
mn=min(o[2] for o in out if not o[0].startswith('CTRL'))
for o in out:
    if o[0].startswith('CTRL') or o[2]<1e-3: print(o)
print('min loss over all obstruction cases:',mn, 'cases:',sum(1 for o in out if not o[0].startswith('CTRL')))
