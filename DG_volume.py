
import numpy as np
import rbf
from rbf.nodes import snap_to_boundary,disperse,menodes
from rbf.geometry import contains
import dentate
from dentate.alphavol import alpha_shape
from dentate.rbf_volume import RBFVolume
from rbf_volume import rotate3d

max_u = 11690.
max_v = 2956.


def DG_volume(u, v, l, rotate=None):
    u = np.array([u]).reshape(-1,)
    v = np.array([v]).reshape(-1,)
    l = np.array([l]).reshape(-1,)

    if rotate is not None:
        a = float(np.deg2rad(rotate))
        rot = rotate3d([1,0,0], a)
    else:
        rot = None

    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))

    pts = np.array([x, y, z]).reshape(3, u.size)

    if rot is not None:
        xyz = np.dot(rot, pts).T
    else:
        xyz = pts.T

    return xyz

def make_volume(lmin, lmax, basis=rbf.basis.phs3, rotate=None, ures=33, vres=30, lres=10):  
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, ures)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, vres)
    obs_l = np.linspace(lmin, lmax, num=lres)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = DG_volume (u, v, l, rotate=rotate)

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, basis=basis, order=2)

    return vol


def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a-b)**2,axis=1))


def make_uvl_distance(xyz_coords,rotate=None):
      f = lambda u, v, l: euclidean_distance(DG_volume(u,v,l,rotate=rotate), xyz_coords)
      return f
