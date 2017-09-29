
import numpy as np
from bspline_surface import BSplineSurface

def DG_surface(u, v, l):
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])


def make_surface(l=-1.): # default l is for the middle of the granule cell layer:
    spatial_resolution = 50.  # um
    max_u = 11690.
    max_v = 2956.
    
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    u = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    v = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(u, v, indexing='ij')
    
    
    xyz = DG_surface (u, v, l)

    srf = BSplineSurface(np.linspace(0, 1, len(u)),
                         np.linspace(0, 1, xyz.shape[2]),
                         xyz)
