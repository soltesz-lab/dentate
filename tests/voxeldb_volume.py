from builtins import str
from collections import defaultdict

import h5py
import numpy as np

import rbf
from dentate.alphavol import alpha_shape
from dentate.rbf_volume import RBFVolume
from mayavi import mlab
from rbf.geometry import contains
from rbf.interpolate import RBFInterpolant
from rbf.nodes import disperse
from rbf.nodes import menodes
from rbf.nodes import snap_to_boundary
from tvtk.api import tvtk

#def make_volume(input_path, ns='Hippocampus', basis=rbf.basis.phs3, res=15):
input_path = './datasets/Hippocampus_voxeldb.h5'
ns  =  'Hippocampus'
res =  50
f   =  h5py.File(input_path)

pty = f['/voxeldb/%s/Type' % ns][:]
    
idxs = np.isin(pty, [1,2,3,4,5,6])
X = f['/voxeldb/%s/X' % ns][idxs]
Y = f['/voxeldb/%s/Y' % ns][idxs]
Z = f['/voxeldb/%s/Z' % ns][idxs]

uX=defaultdict(list)
for i,e in enumerate(X):
    uX[e].append(i)

uY=defaultdict(list)
for i,e in enumerate(Y):
    uY[e].append(i)

uZ=defaultdict(list)
for i,e in enumerate(Z):
    uZ[e].append(i)

Xkeys = sorted(uX.keys())
obs_u = np.random.choice(len(Xkeys), replace=False, size=res)

uX_indlst = []
for i in sorted(obs_u):
    k = Xkeys[i]
    uX_indlst.append(uX[k])

uX_indices = np.concatenate(uX_indlst)

Ykeys = sorted(uY.keys())
obs_v = np.random.choice(len(Ykeys), replace=False, size=res)

uY_indlst = []
for i in sorted(obs_v):
    k = Ykeys[i]
    uY_indlst.append(uY[k])
    
uY_indices = np.concatenate(uY_indlst)
    

Zkeys = sorted(uZ.keys())
obs_l = np.random.choice(len(Zkeys), replace=False, size=res)

uZ_indlst = []
for i in sorted(obs_l):
    k = Zkeys[i]
    uZ_indlst.append(uZ[k])
    
uZ_indices = np.concatenate(uZ_indlst)
    
#print 'uX: ', uX
#print 'uX_indices: ', uX_indices
obs_indices = np.concatenate([uX_indices, uY_indices, uZ_indices])
print('obs_indices.shape: %s' % str(obs_indices.shape))

#u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
#obs_uvl = np.array([obs_u.ravel(),obs_v.ravel(),obs_l.ravel()]).T

obs_x = X[obs_indices]
obs_y = Y[obs_indices]
obs_z = Z[obs_indices]


obs_xyz = np.column_stack([obs_x, obs_y, obs_z])
print('obs_xyz.shape: %s' % str(obs_xyz.shape))
print(obs_xyz)

#s = np.linspace(0., 1., obs_u.size)
#obs_uvl = np.array([s],dtype=np.float32).T

#print obs_uvl
#print X[obs_u]

#xvol = RBFInterpolant(obs_uvl,X[obs_u],basis='imq',penalty=0.1,extrapolate=False)

#mlab.mesh(X[obs_u], Y[obs_v], Z[obs_l], representation='wireframe', color=(0, 0, 0))
#mlab.points3d(X[obs_u], Y[obs_v], Z[obs_l], scale_factor=100.0, color=(1, 1, 0))


    
#vol = RBFVolume(np.asarray(obs_u.ravel(), dtype=np.float32), \
#                np.asarray(obs_v.ravel(), dtype=np.float32), \
#                np.asarray(obs_l.ravel(), dtype=np.float32), \
#                obs_xyz, basis='imq', order=1)


#xyz = make_volume()

#vol.mplot_surface(color=(0, 1, 0), ures=2, vres=2, opacity=0.33)

#xyz = np.array([X[obs_u].ravel(), Y[obs_v].ravel(), Z[obs_l].ravel()]).reshape(3, obs_u.size).T

#mlab.points3d(obs_xyz[:,0], obs_xyz[:,1], obs_xyz[:,2], scale_factor=20.0, color=(1, 1, 0))
print(obs_xyz.shape)
print(obs_xyz[:,0])

alpha =  alpha_shape(pts,radius,tri=None)

# Turn off perspective
fig = mlab.gcf()
fig.scene.camera.trait_set(parallel_projection=1)

mlab.show()
