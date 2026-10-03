# Numerical SEARCH (not proof): two arbitrary 2-qubit unitaries on a chain (i,j),(j,k), per-string phases allowed
# (loss = sum_x 1-|<f(x)|U|x>|^2). Also same-pair (V_q (x) W). Positive control Pi={01,10,11}.
import itertools, numpy as np, torch
torch.set_num_threads(4); torch.manual_seed(0)
def ccx(x): a,b,t=x; return (a,b,t^(a&b))
def idx(x): return x[0]*4+x[1]*2+x[2]
def herm(p,n):
    A=torch.complex(p[:n*n].view(n,n),p[n*n:].view(n,n)); return torch.matrix_exp(1j*(A+A.conj().T)/2)
def embed2(V,q1,q2):  # V on qubits (q1,q2), qubit order a=0,b=1,t=2
    V=V.reshape(2,2,2,2); r=[q for q in range(3) if q not in (q1,q2)][0]
    I=torch.eye(2,dtype=V.dtype); T=torch.einsum('ABab,Cc->ABCabc',V,I)  # axes (q1,q2,r)
    order=[q1,q2,r]; perm=[order.index(q) for q in range(3)]
    T=T.permute(*perm,*[p+3 for p in perm]); return T.reshape(8,8)
def loss_fn(U,L):
    s=0
    for x in L: s=s+1-U[idx(ccx(x)),idx(x)].abs()**2
    return s
def run(L,struct,starts=30,steps=600):
    best=9
    for s in range(starts):
        p1=torch.randn(32,dtype=torch.float64,requires_grad=True); p2=torch.randn(32,dtype=torch.float64,requires_grad=True)
        opt=torch.optim.Adam([p1,p2],lr=0.05)
        for it in range(steps):
            V1=herm(p1,4); V2=herm(p2,4)
            if struct[0]=='chain':
                i,j,k=struct[1]; U=embed2(V2,j,k)@embed2(V1,i,j)
            else:
                q=struct[1]; r=[x for x in range(3) if x!=q]; U=embed2(V2,*r)@embed2(V1,*r)
            l=loss_fn(U,L); opt.zero_grad(); l.backward(); opt.step()
        best=min(best,l.item())
    return best
B3=[[(0,0,t[0]),(0,1,t[1]),(1,0,t[2]),(1,1,t[3])] for t in itertools.product((0,1),repeat=4)]
PC=[[(0,1,t[0]),(1,0,t[1]),(1,1,t[2])] for t in itertools.product((0,1),repeat=3)]
structs=[('chain',p) for p in itertools.permutations(range(3))]+[('pair',q) for q in range(3)]
import sys
which=sys.argv[1]
if which=='pc':
    for L in PC[:3]:
        print('PC',[''.join(map(str,x)) for x in L],{str(s[1]):round(run(L,s,5,400),6) for s in structs[:6]},flush=True)
else:
    gmin=9
    for L in B3:
        r={str(s[1]):round(run(L,s,12,500),4) for s in structs}; m=min(r.values()); gmin=min(gmin,m)
        print('B3',[''.join(map(str,x)) for x in L],'min',m,r,flush=True)
    print('GLOBAL MIN over B3',gmin)
