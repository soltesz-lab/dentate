import numpy as np
from scipy.spatial import Delaunay
from collections import namedtuple

AlphaShape = namedtuple('AlphaShape', ['points', 'simplices', 'bounds'], verbose=False)

## Determine circumcenters of polyhedra as described in the following page:
## http://mathworld.wolfram.com/Circumsphere.html
def circumcenters(simplices, points):

    n     = np.ones((simplices.shape[0],simplices.shape[1],1))
    spts  = points[simplices]
    a     = np.linalg.det(np.concatenate((spts,n),axis=2))
    
    xyz_sqsum = spts[:,:,0]**2 + spts[:,:,1]**2 + spts[:,:,2]**2
    Dx = np.linalg.det(np.stack((xyz_sqsum, spts[:,:,1], spts[:,:,2], np.ones((xyz_sqsum.shape[0],4))), axis=-1))
    Dy = np.linalg.det(np.stack((xyz_sqsum, spts[:,:,0], spts[:,:,2], np.ones((xyz_sqsum.shape[0],4))), axis=-1))
    Dz = np.linalg.det(np.stack((xyz_sqsum, spts[:,:,0], spts[:,:,1], np.ones((xyz_sqsum.shape[0],4))), axis=-1))

    c = np.linalg.det(np.stack((xyz_sqsum, spts[:,:,0], spts[:,:,1], spts[:,:,2]), axis=-1))
    del xyz_sqsum

    ## circumcenter of the sphere
    x0 = Dx / (2.0 * a)
    y0 = Dy / (2.0 * a)
    z0 = Dz / (2.0 * a)

    ## circumradius
    r = np.sqrt((Dx**2) + (Dy**2) + (Dz**2) - 4.0*a*c) / (2.0 * np.abs(a))
    
    return ((x0, y0, z0), r)
    
##
## Returns the facets that are reference only by simplex of the given
## triangulation.
##
def free_boundary(simplices):

    ## Sort the facet indices in the triangulation
    simplices = np.sort(simplices,axis=1)
    facets = np.vstack((simplices[:,[0, 1, 2]], \
                        simplices[:,[0, 1, 3]], \
                        simplices[:,[0, 2, 3]], \
                        simplices[:,[1, 2, 3]]))
    ## Find unique facets                    
    ufacets, counts = np.unique(facets,return_counts=True,axis=0)

    ## Determine which facets are part of only one simplex
    bidxs = np.where(counts == 1)[0]

    return ufacets[bidxs]
    

def alpha_shape(pts,radius,tri=None):
    
## Alpha shape of 2D or 3D point set.
##   V = ALPHAVOL(X,R) gives the area or volume V of the basic alpha shape
##    for a 2D or 3D point set. X is a coordinate matrix of size Nx2 or Nx3.
##
##   R is the probe radius with default value R = Inf. In the default case
##   the basic alpha shape (or alpha hull) is the convex hull.
##
##   [V,S] = ALPHAVOL(X,R) outputs a structure S with fields:
##    S.tri - Triangulation of the alpha shape (Mx3 or Mx4)
##   S.vol - Area or volume of simplices in triangulation (Mx1)
##   S.rcc - Circumradius of simplices in triangulation (Mx1)
##   S.bnd - Boundary facets (Px2 or Px3)

##   Based on MATLAB code by Jonas Lundgren <splinefit@gmail.com>


## Check coordinates
    if tri is None:
        dim = pts.shape[1]
    else:
        dim = tri.points.shape[1]

    if dim < 2 or dim > 3:
        raise ValueError('pts must have 2 or 3 columns.')

    ## Check probe radius
    if not (type(radius) == float):
        raise ValueError('radius must be a real number.')


    ## Delaunay triangulation
    if tri is None:
        tri = Delaunay(pts)

    ## Remove zero volume tetrahedra since
    ## these can be of arbitrary large circumradius
    #if dim == 3:
    #    n = tri.simplices.shape[0]
    #    vol = volumes(tri)
    #    epsvol = 1e-12*np.sum(vol)/n
    #    T = T(vol > epsvol,:)
    #    holes = size(T,1) < n

    ## Limit circumradius of simplices
    _,rcc   = circumcenters(tri.simplices, tri.points)
    rccidxs = np.where(rcc < radius)[0]
    T       = tri.simplices[rccidxs,:]
    rcc     = rcc[rccidxs]
    
    bnd = free_boundary(T)
    
    return AlphaShape(tri.points, T, bnd)
