
import numpy as np
import rbf
from rbf.nodes import snap_to_boundary,disperse,menodes
from rbf.geometry import contains
import dentate
from dentate.alphavol import alpha_shape
from dentate.rbf_volume import RBFVolume

max_u = 11690.
max_v = 2956.


def DG_volume(u, v, l):
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])


def make_volume(lmin, lmax, basis=rbf.basis.phs3):  
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 25)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 25)
    obs_l = np.linspace(lmin, lmax, num=10)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = DG_volume (u, v, l).reshape(3, u.size).T

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, basis=basis, order=2)

    return vol
