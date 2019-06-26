
import numpy as np
from bspline_surface import BSplineSurface

max_u = 11690.
max_v = 2956.


def DG_surface(u, v, l):
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])


def make_surface(l=-1., spatial_resolution=1.):  # default l is for the middle of the granule cell layer;

    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(su, sv, indexing='ij')
    
    xyz = DG_surface(u, v, l)

    srf = BSplineSurface(su, sv, xyz)

    return srf


def make_layer_surfaces(surface_config, l_fraction=0.5, spatial_resolution=1.):  

    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(su, sv, indexing='ij')

    srf_dict = {}

    min_ext = surface_config['Parametric Surface']['Minimum Extent']
    max_ext = surface_config['Parametric Surface']['Maximum Extent']
    
    for layer_label in min_ext.keys():

        layer_min_ext = min_ext[layer_label]
        layer_max_ext = max_ext[layer_label]

        l = layer_min_ext + ((layer_max_ext - layer_min_ext) * l_fraction)
        
        xyz = DG_surface(u, v, l)
        
        srf = BSplineSurface(su, sv, xyz)

        srf_dict[layer_label] = srf
        
    return srf_dict

