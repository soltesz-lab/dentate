from builtins import str
from builtins import range
import numpy as np

id_vec = np.arange(0,110)
t_vec  = np.arange(0,110) * 0.25

binlst  = []

typelst = ['AAC', 'BC', 'HC', 'GC', 'HCC', 'LPP', 'MC', 'MOPP', 'MPP', 'NGFC', 'IS']

for k in np.arange(0,11):
    binlst.append(k * 10)
        
binvect  = np.array(binlst)
sort_idx = np.argsort(binvect,axis=0)
bins     = binvect[sort_idx]

print('binvect = %s' % str(binvect))
print('bins = %s' % str(bins))

types    = [ typelst[i] for i in sort_idx ]
inds     = np.digitize(id_vec, bins)
print('inds = %s' % str(inds))

for i in range(0,10):
        spkdict  = {}
        sinds    = np.where(inds == i)
        print('sinds = %s' % str(sinds))
        if len(sinds) > 0:
            ids      = id_vec[sinds]
            ts       = t_vec[sinds]
            for j in range(0,len(ids)):
                id = ids[j]
                t  = ts[j]
                if id in spkdict:
                    spkdict[id]['t'].append(t)
                else:
                    spkdict[id]= {'t': [t]}
            for j in list(spkdict.keys()):
                spkdict[j]['t'] = np.array(spkdict[j]['t'])
        pop_name = types[i]

        print(spkdict)
