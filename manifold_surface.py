
import numpy as np
from sklearn import manifold
from scipy import interpolate

spatial_resolution = 25.  # um
max_u = 11690.
max_v = 2956.

def surface(u, v, l):
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return x,y,z

du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
u = np.arange(-0.016*np.pi, 1.01*np.pi, du)
v = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

u, v = np.meshgrid(u, v, indexing='ij')

# for the middle of the granule cell layer:
l = -1.

s = surface (u, v, l)
X = np.vstack((s[0].ravel(),s[1].ravel(),s[2].ravel())).T

del s, u, v, l

n_points = X.shape[0]
n_neighbors = 30
n_components = 2

Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method='modified').fit_transform(X)

Y1 = Y1 * 100.

ip_surface = interpolate.CloughTocher2DInterpolator(Y1, X)
