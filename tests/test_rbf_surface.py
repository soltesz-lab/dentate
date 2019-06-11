from __future__ import division
from builtins import str
from past.utils import old_div
from mpi4py import MPI
import math
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
from mayavi import mlab
from matplotlib.colors import ColorConverter


def test_surface(u, v, l):
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])


obs_u_spatial_resolution = 500.  # um
obs_v_spatial_resolution = 150.  # um

max_u = 11690.
max_v = 2956.
    
du = old_div((1.01*np.pi-(-0.016*np.pi)),max_u*obs_u_spatial_resolution)
dv = old_div((1.425*np.pi-(-0.23*np.pi)),max_v*obs_v_spatial_resolution)
obs_u = np.arange(-0.016*np.pi, 1.01*np.pi, du)
obs_v = np.arange(-0.23*np.pi, 1.425*np.pi, dv)
print('obs_v size: %s' % str(obs_v.size))
obs_l = np.linspace(0., 1., num=10)

u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
xyz = test_surface (u, v, l).reshape(3, u.size)
uvl_obs = np.array([u.ravel(),v.ravel(),l.ravel()]).T

print('u shape: %s' % str(u.shape))
print('xyz shape: %s' % str(xyz.shape))
print('uvl_obs shape: %s' % str(uvl_obs.shape))

basis = rbf.basis.phs2
order = 1

print('creating xsrf...')
xsrf = RBFInterpolant(uvl_obs,xyz[0],penalty=0.1,basis=basis,order=order)
print('creating ysrf...')
ysrf = RBFInterpolant(uvl_obs,xyz[1],penalty=0.1,basis=basis,order=order)
print('creating zsrf...')
zsrf = RBFInterpolant(uvl_obs,xyz[2],penalty=0.1,basis=basis,order=order)

spatial_resolution = 25.  # um
du = old_div((1.01*np.pi-(-0.016*np.pi)),max_u*spatial_resolution)
dv = old_div((1.425*np.pi-(-0.23*np.pi)),max_v*spatial_resolution)
su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)
sl = np.linspace(0., 1., num=10)

u, v, l = np.meshgrid(su, sv, sl)
uvl_s = np.array([u.ravel(),v.ravel(),l.ravel()]).T

print('uvl_s shape: %s' % str(uvl_s.shape))

print('sampling xsrf...')
X = xsrf(uvl_s)
print('sampling ysrf...')
Y = ysrf(uvl_s)
print('sampling zsrf...')
Z = zsrf(uvl_s)

mpts = np.array([X,Y,Z]).reshape(3, len(u), -1)

mlab.figure(bgcolor=(1.,1.,1.), size=(1000, 1000))

mlab.mesh(*mpts, color=(0.2, 0.7, 0.9))
# Turn off perspective
fig = mlab.gcf()
fig.scene.camera.trait_set(parallel_projection=1)
mlab.show()
