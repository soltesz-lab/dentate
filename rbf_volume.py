"""Implements a parametric volume as a 3-tuple of RBF instances, one each for x, y and z.
Based on code from bspline_surface.py
"""

import math
import numpy as np
from collections import namedtuple
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis

def add_edge(edges, edge_points, coords, i, j):
    """
    Add a line between the i-th and j-th points,
    if not in the list already
    """
    if (i, j) in edges or (j, i) in edges:
        # already added
        return
    edges.add( (i, j) )
    edge_points.append(coords[ [i, j] ])

def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a-b)**2,axis=1))


def rotate3d(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

class RBFVolume(object):
    def __init__(self, u, v, l, xyz, order=1, basis=rbf.basis.phs2):
        """Parametric (u,v,l) 3D volume approximation.

        Parameters
        ----------
        u, v, l : array_like
            1-D arrays of coordinates.
        xyz : array_like
            3-D array of (x, y, z) data with shape (3, u.size, v.size).
        order : int, optional
            Order of interpolation. Default is 1.
        basis: RBF basis function
        """

        self._create_vol(u, v, l, xyz, order, basis)

        self.u  = u
        self.v  = v
        self.l  = l
        self.order = order

    def __call__(self, *args, **kwargs):
        """Convenience to allow evaluation of a RBFVolume
        instance via `foo(0, 0, 0)` instead of `foo.ev(0, 0, 0)`.
        """
        return self.ev(*args, **kwargs)

    def _create_vol(self, obs_u, obs_v, obs_l, xyz, order, basis):

        # Create volume definitions
        u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
        uvl_obs = np.array([u.ravel(),v.ravel(),l.ravel()]).T

        xvol = RBFInterpolant(uvl_obs,xyz[0],penalty=0.1,basis=basis,order=order)
        yvol = RBFInterpolant(uvl_obs,xyz[1],penalty=0.1,basis=basis,order=order)
        zvol = RBFInterpolant(uvl_obs,xyz[2],penalty=0.1,basis=basis,order=order)

        self._xvol = xvol
        self._yvol = yvol
        self._zvol = zvol

    def _resample_uv(self, ures, vres):
        """Helper function to re-sample to u and v parameters
        at the specified resolution
        """
        u, v = self.u, self.v
        lu, lv = len(u), len(v)
        nus = np.array(list(enumerate(u))).T
        nvs = np.array(list(enumerate(v))).T
        newundxs = np.linspace(0, lu - 1, ures * lu - (ures - 1))
        newvndxs = np.linspace(0, lv - 1, vres * lv - (vres - 1))
        hru = np.interp(newundxs, *nus)
        hrv = np.interp(newvndxs, *nvs)
        return hru, hrv

    def ev(self, su, sv, sl):
        """Get point(s) in volume at (su, sv, sl).

        Parameters
        ----------
        u, v, l : scalar or array-like
            u, v, l may be scalar or vector

        Returns
        -------
        Returns an array of shape 3 x len(u) x len(v) x len(l)
        """

        U, V, L = np.meshgrid(su, sv, sl)
        uvl_s = np.array([U.ravel(),V.ravel(),L.ravel()]).T

        X = self._xvol(uvl_s)
        Y = self._yvol(uvl_s)
        Z = self._zvol(uvl_s)

        arr = np.array([X,Y,Z]).reshape(3, len(U), -1)
        return arr

    
    def utan(self, su, sv, sl, normalize=True):

        u = np.array([su]).reshape(-1,)
        v = np.array([sv]).reshape(-1,)
        l = np.array([sl]).reshape(-1,)

        dxdu = self._xvol(u, v, l, diff=np.asarray([1,0,0]))
        dydu = self._yvol(u, v, l, diff=np.asarray([1,0,0]))
        dzdu = self._zvol(u, v, l, diff=np.asarray([1,0,0]))

        du = np.array([dxdu, dydu, dzdu]).T

        du = du.swapaxes(0, 1)

        if normalize:
            du /= np.sqrt((du**2).sum(axis=2))[:, :, np.newaxis]

        arr = du.transpose(2, 0, 1)
        return arr

    
    def vtan(self, su, sv, normalize=True):

        u = np.array([su]).reshape(-1,)
        v = np.array([sv]).reshape(-1,)
        l = np.array([sl]).reshape(-1,)
        
        dxdv = self._xvol(u, v, l, diff=np.asarray([0,1,0]))
        dydv = self._yvol(u, v, l, diff=np.asarray([0,1,0]))
        dzdv = self._zvol(u, v, l, diff=np.asarray([0,1,0]))
        dv = np.array([dxdv, dydv, dzdv]).T

        dv = dv.swapaxes(0, 1)

        if normalize:
            dv /= np.sqrt((dv**2).sum(axis=2))[:, :, np.newaxis]

        arr = dv.transpose(2, 0, 1)
        return arr

    

    def normal(self, su, sv, sl):
        """Get normal(s) at (u, v, l).

        Parameters
        ----------
        u, v, l : scalar or array-like
            u and v may be scalar or vector (see below)

        Returns
        -------
        Returns an array of shape 3 x len(u) x len(v) x len(l)
        """

        u = np.array([su]).reshape(-1,)
        v = np.array([sv]).reshape(-1,)
        l = np.array([sl]).reshape(-1,)

        dxdus = self._xvol(u, v, l, diff=np.asarray([1,0,0]))
        dydus = self._yvol(u, v, l, diff=np.asarray([1,0,0]))
        dzdus = self._zvol(u, v, l, diff=np.asarray([1,0,0]))
        dxdvs = self._xvol(u, v, l, diff=np.asarray([0,1,0]))
        dydvs = self._yvol(u, v, l, diff=np.asarray([0,1,0]))
        dzdvs = self._zvol(u, v, l, diff=np.asarray([0,1,0]))

        normals = np.cross([dxdus, dydus, dzdus],
                           [dxdvs, dydvs, dzdvs],
                           axisa=0, axisb=0)

        normals /= np.sqrt((normals**2).sum(axis=2))[:, :, np.newaxis]

        arr = normals.transpose(2, 0, 1)
        return arr

        
    def point_distance(self, su, sv, sl):
        """Cumulative distance between pairs of (u, v, l) coordinates.

        Parameters
        ----------
        u, v, l : array-like
 
        Returns
        -------
        If the lengths of u and v are at least 2, returns the total arc length
        between each u,v pair.
        """
        u = np.array([su]).reshape(-1,)
        v = np.array([sv]).reshape(-1,)
        l = np.array([sl]).reshape(-1,)

        npts   = max(u.shape[0], v.shape[0])
        pts    = self.ev(u, v, l).reshape(3, -1).T

        del u, v, l
        distance = 0
        
        if npts > 1:
            a = pts[1:,:]
            b = pts[0:npts-1,:]
            distance = np.sum(euclidean_distance(a, b))

                
        return distance

    def mplot_surface(self, ures=8, vres=8, **kwargs):
        """Plot the enclosing surfaces of the volume using Mayavi's `mesh()` function

        Parameters
        ----------
        ures, vres : int
            Specifies the oversampling of the original
            volume in u and v directions. For example:
            if `ures` = 2, and `self.u` = [0, 1, 2, 3],
            then the surface will be resampled at
            [0, 0.5, 1, 1.5, 2, 2.5, 3] prior to
            plotting.

        kwargs : dict
            See Mayavi docs for `mesh()`

        Returns
        -------
            None
        """
        from mayavi import mlab
        from matplotlib.colors import ColorConverter

        if not kwargs.has_key('color'):
            # Generate random color
            cvec = np.random.rand(3)
            cvec /= math.sqrt(cvec.dot(cvec))
            kwargs['color'] = tuple(cvec)
        else:
            # The following will convert text strings representing
            # colors into their (r, g, b) equivalents (which is
            # the only way Mayavi will accept them)
            from matplotlib.colors import ColorConverter
            cconv = ColorConverter()
            if kwargs['color'] is not None:
                kwargs['color'] = cconv.to_rgb(kwargs['color'])

        # Make new u and v values of (possibly) higher resolution
        # the original ones.
        hru, hrv = self._resample_uv(ures, vres)
        # Sample the surface at the new u, v values and plot
        meshpts1 = self.ev(hru, hrv, self.l[-1])
        meshpts2 = self.ev(hru, hrv, self.l[0])
        m1 = mlab.mesh(*meshpts1, **kwargs)
        m2 = mlab.mesh(*meshpts2, **kwargs)
        
        # Turn off perspective
        fig = mlab.gcf()
        fig.scene.camera.trait_set(parallel_projection=1)
        return fig

    
    def mplot_volume(self, ures=8, vres=8, **kwargs):
        """Plot the volume using Mayavi's `scalar_scatter()` function

        Parameters
        ----------
        ures, vres : int
            Specifies the oversampling of the original
            volume in u and v directions. For example:
            if `ures` = 2, and `self.u` = [0, 1, 2, 3],
            then the surface will be resampled at
            [0, 0.5, 1, 1.5, 2, 2.5, 3] prior to
            plotting.

        kwargs : dict
            See Mayavi docs for `mesh()`

        Returns
        -------
            None
        """
        from mayavi import mlab
        from matplotlib.colors import ColorConverter

        if not kwargs.has_key('color'):
            # Generate random color
            cvec = np.random.rand(3)
            cvec /= math.sqrt(cvec.dot(cvec))
            kwargs['color'] = tuple(cvec)
        else:
            # The following will convert text strings representing
            # colors into their (r, g, b) equivalents (which is
            # the only way Mayavi will accept them)
            from matplotlib.colors import ColorConverter
            cconv = ColorConverter()
            if kwargs['color'] is not None:
                kwargs['color'] = cconv.to_rgb(kwargs['color'])

        # Make new u and v values of (possibly) higher resolution
        # the original ones.
        hru, hrv = self._resample_uv(ures, vres)
        volpts = self.ev(hru, hrv, self.l)

        src =  mlab.pipeline.scalar_scatter(*volpts, **kwargs)
        mlab.pipeline.volume(src, **kwargs)
        
        # Turn off perspective
        fig = mlab.gcf()
        fig.scene.camera.trait_set(parallel_projection=1)
        return fig


    def compute_delaunay(self, ures=8, vres=8, **kwargs):
        """Compute the triangulation of the volume using scipy's
        `delaunay` function

        Parameters
        ----------
        ures, vres : int
        Specifies the oversampling of the original
        volume in u and v directions. For example:
        if `ures` = 2, and `self.u` = [0, 1, 2, 3],
        then the surface will be resampled at
        [0, 0.5, 1, 1.5, 2, 2.5, 3] prior to
        plotting.

        kwargs : dict
        See scipy docs for `scipy.spatial.Delaunay()`

        Returns
        -------
        None
        """
        from scipy.spatial import Delaunay

        # Make new u and v values of (possibly) higher resolution
        # the original ones.
        hru, hrv = self._resample_uv(ures, vres)
        volpts = self.ev(hru, hrv, self.l).reshape(3, -1).T

        tri = Delaunay(volpts)
        return tri



    
    def plot_srf(self, ures=8, vres=8, **kwargs):
        """Alias for mplot_surface()
        """
        self.mplot_surface(ures=ures, vres=vres, **kwargs)

        
    def copy(self):
        """Get a copy of the surface
        """
        from copy import deepcopy
        return deepcopy(self)



def test_surface(u, v, l):
    import numpy as np
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])

    import numpy as np


def test_nodes():
    from scipy.spatial import Delaunay
    from rbf.nodes import snap_to_boundary,disperse,menodes
    from rbf.geometry import contains

    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 10)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 10)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size)

    srf = RBFVolume(obs_u, obs_v, obs_l, xyz, order=1)

    tri = srf.compute_delaunay()
    
    # Define the problem domain with line segments.
    vert = np.asarray(tri.points, dtype=np.float64)
    smp  = srf.compute_boundary()
    print 'vert shape: ', vert.shape
    print 'smp shape: ', smp.shape

    N = 500 # total number of nodes
    
    # create N quasi-uniformly distributed nodes over the unit square
    nodes, smpid = menodes(N,vert,smp,itr=1)
    print 'nodes: ', nodes
    
    print 'nodes shape: ', nodes.shape
    # remove nodes outside of the domain
    ##in_nodes = nodes[contains(nodes,vert,smp)]
        
    from mayavi import mlab
    srf.mplot_surface(color=(0, 1, 0), opacity=0.33, ures=10, vres=10)

    mlab.points3d(*nodes.T, color=(1, 1, 0), scale_factor=100.0)
    
    mlab.show()


def test_uv_isospline():

    max_u = 11690.
    max_v = 2956.
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size)

    order = [1]
    for ii in xrange(len(order)):
        srf = RBFVolume(obs_u, obs_v, obs_l, xyz, order=order[ii])

        U, V = srf._resample_uv(5, 5)
        L = np.asarray([-1.0])

        nupts = U.shape[0]
        nvpts = V.shape[0]
            
        print srf.point_distance(U, V[0], L)
        print srf.point_distance(U[int(nupts/2)], V, L)


    from mayavi import mlab
        
    spatial_resolution = 5.  # um
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    U = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    V = np.arange(-0.23*np.pi, 1.425*np.pi, dv)
    L = 1.0
    
    U, V = srf._resample_uv(10, 10)
    L = np.asarray([1.0])
        
    nupts = U.shape[0]
    nvpts = V.shape[0]
    # Plot u,v-isosplines on the surface
    upts = srf(U, V[0], L)
    vpts = srf(U[int(nupts/2)], V, L)
    
    srf.mplot_scatter(color=(0, 1, 0), opacity=1.0, ures=10, vres=10)
    
    mlab.points3d(*upts, scale_factor=100.0, color=(1, 1, 0))
    mlab.points3d(*vpts, scale_factor=100.0, color=(1, 1, 0))
    
    mlab.show()

def test_tri():

    max_u = 11690.
    max_v = 2956.
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size)

    srf = RBFVolume(obs_u, obs_v, obs_l, xyz, order=1)

    
    


    
if __name__ == '__main__':
    #test_uv_isospline()
    test_nodes()

    
    
    
