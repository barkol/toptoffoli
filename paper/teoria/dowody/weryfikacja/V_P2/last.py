import sys; sys.argv=['x','none']
exec(open('search.py').read().split("import sys")[0])
L=[(0,0,0),(0,1,1),(1,0,1),(1,1,1)]
r={str(s[1]):round(run(L,s,12,500),4) for s in structs}; print('B3 000,011,101,111 min',min(r.values()),r)
