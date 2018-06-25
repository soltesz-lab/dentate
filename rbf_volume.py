"""Implements a parametric volume as a 3-tuple of RBF instances, one each for u, v and l.
Based on code from bspline_surface.py
"""

import math
import numpy as np
from collections import namedtuple
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis

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


def cartesian_product(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian_product(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


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

        self._create_vol(u, v, l, xyz, order=order, basis=basis)

        self.u  = u
        self.v  = v
        self.l  = l
        self.order = order

        self.tri = None
        self.facets = None
        self.facet_counts = None
        
    def __call__(self, *args, **kwargs):
        """Convenience to allow evaluation of a RBFVolume
        instance via `foo(0, 0, 0)` instead of `foo.ev(0, 0, 0)`.
        """
        return self.ev(*args, **kwargs)

    def _create_vol(self, obs_u, obs_v, obs_l, xyz, **kwargs):

        # Create volume definitions
        u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
        uvl_obs = np.array([u.ravel(),v.ravel(),l.ravel()]).T

        xvol = RBFInterpolant(uvl_obs,xyz[:,0],**kwargs)
        yvol = RBFInterpolant(uvl_obs,xyz[:,1],**kwargs)
        zvol = RBFInterpolant(uvl_obs,xyz[:,2],**kwargs)

        uvol = RBFInterpolant(xyz,uvl_obs[:,0],**kwargs)
        vvol = RBFInterpolant(xyz,uvl_obs[:,1],**kwargs)
        lvol = RBFInterpolant(xyz,uvl_obs[:,2],**kwargs)

        self._xvol = xvol
        self._yvol = yvol
        self._zvol = zvol
        self._uvol = uvol
        self._vvol = vvol
        self._lvol = lvol

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

    def _resample_uvl(self, ures, vres, lres):
        """Helper function to re-sample u, v and l parameters
        at the specified resolution
        """
        u, v, l = self.u, self.v, self.l
        lu, lv, ll = len(u), len(v), len(l)
        nus = np.array(list(enumerate(u))).T
        nvs = np.array(list(enumerate(v))).T
        nls = np.array(list(enumerate(l))).T
        newundxs = np.linspace(0, lu - 1, ures * lu - (ures - 1))
        newvndxs = np.linspace(0, lv - 1, vres * lv - (vres - 1))
        newlndxs = np.linspace(0, ll - 1, lres * ll - (lres - 1))
        hru = np.interp(newundxs, *nus)
        hrv = np.interp(newvndxs, *nvs)
        hrl = np.interp(newlndxs, *nls)
        return hru, hrv, hrl

    def ev(self, su, sv, sl, mesh=True, chunk_size=1000, return_coords=False):
        """Get point(s) in volume at (su, sv, sl).

        Parameters
        ----------
        u, v, l : scalar or array-like
        return_coords : boolean, return the coordinates that were evaluated

        Returns
        -------
        if option mesh is True: Returns an array of shape len(u) x len(v) x len(l) x 3
        """

        if mesh:
            U, V, L = np.meshgrid(su, sv, sl)
        else:
            U = su
            V = sv
            L = sl
        
        uvl_coords = np.array([U.ravel(),V.ravel(),L.ravel()]).T
        X = self._xvol(uvl_coords, chunk_size=chunk_size)
        Y = self._yvol(uvl_coords, chunk_size=chunk_size)
        Z = self._zvol(uvl_coords, chunk_size=chunk_size)

        arr = np.array([X,Y,Z])

        if return_coords:
            return (arr.reshape(3, len(U), -1), uvl_coords)
        else:
            return arr.reshape(3, len(U), -1)

    def inverse(self, xyz):
        """Get parametric coordinates (u, v, l) that correspond to the given x, y, z.
        May return None if x, y, z are outside the interpolation domain.

        Parameters
        ----------
        xyz : array of coordinates

        Returns
        -------
        Returns an array of shape 3 x len(xyz)
        """

        U = self._uvol(xyz)
        V = self._vvol(xyz)
        L = self._lvol(xyz)

        
        arr = np.array([U,V,L])
        return arr.T

    
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

        
    def point_distance(self, su, sv, sl, axis=0, interp_chunk_size=1000, origin_coords=None, return_coords=True):
        """Cumulative distance along an axis between pairs of (u, v, l) coordinates

        Parameters
        ----------
        u, v, l : array-like

        axis: axis along which the distance should be computed

        origin_coords: the origin coordinates (the left-most coordinate of each axis if None)

        return_coords: if True, returns the coordinates for which computed distance (default: True)

        Returns
        -------
        If the lengths of u and v are at least 2, returns the total arc length
        between each u,v pair.
        """
        u = np.array([su]).reshape(-1,)
        v = np.array([sv]).reshape(-1,)
        l = np.array([sl]).reshape(-1,)

        
        assert(len(u) > 0)
        assert(len(v) > 0)
        assert(len(l) > 0)

        input_axes = [u, v, l]
        if origin_coords is None:
            origin_coords = np.asarray([ input_axes[i][0] for i in xrange(0,3) ])

        c = input_axes

        cl = [ (np.sort(c[i][np.where(c[i] <= origin_coords[i])[0]]))[::-1] if i == axis else c[i] for i in xrange(0,3) ]
        cr = [ (np.sort(c[i][np.where(c[i] > origin_coords[i])[0]])) if i == axis else c[i] for i in xrange(0,3) ]

        ordered_axes = [ (-1, cl), (1, cr) ]

        aidx = list(xrange(0,3))
        aidx.remove(axis)
        
        distances = []
        coords    = []

        origin_axes = [ origin_coords[i] if i == axis else c[i] for i in xrange(0,3) ]
        (origin_pts, origin_coords) = self.ev(*origin_axes, return_coords=True)
        origin_pts = origin_pts.reshape(3, -1).T
        oind = np.lexsort(tuple([ origin_coords[:,i] for i in aidx ]))
        origin_sorted = origin_pts[oind]

        
        for (sgn, axes) in ordered_axes:

            npts = axes[axis].shape[0]

            if npts > 0:
                (eval_pts, eval_coords) = self.ev(*axes, chunk_size=interp_chunk_size, return_coords=True)
                coord_idx = np.argsort(eval_coords[:,axis])
                if sgn < 0:
                    coord_idx = coord_idx[::-1]
                all_pts = (eval_pts.reshape(3, -1).T)[coord_idx,:]
                all_pts_coords = eval_coords[coord_idx,:]
                split_pts = np.split(all_pts, npts)
                split_pts_coords = np.split(all_pts_coords, npts)
                ## distance from axis_origin to first point
                fst_point  = split_pts[0]
                fst_coords = split_pts_coords[0]
                find = np.lexsort(tuple([ fst_coords[:,i] for i in aidx ]))
                fst_sorted = fst_point[find]
                cdist = euclidean_distance(origin_sorted, fst_sorted).reshape(-1,1)
                for i in xrange(0,len(cdist)):
                    if abs(cdist[i] < 1e-3):
                        cdist[i] = 0.
                distances.append(sgn * cdist)
                if return_coords:
                    coords.append(fst_coords[find].ravel())
                for i in xrange(0, npts-1):
                    a = split_pts[i+1]
                    b = split_pts[i]
                    a_coords = split_pts_coords[i+1]
                    b_coords = split_pts_coords[i]
                    aind = np.lexsort(tuple([ a_coords[:,i] for i in aidx ]))
                    bind = np.lexsort(tuple([ b_coords[:,i] for i in aidx ]))
                    a_sorted = a[aind]
                    b_sorted = b[bind]
                    dist = euclidean_distance(a_sorted, b_sorted).reshape(-1,1)
                    cdist = cdist + dist
                    distances.append(sgn * cdist)
                    if return_coords:
                        coords.append(a_coords[aind].ravel())

        if return_coords:
            return distances, coords
        else:
            return distances

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
        volpts = self.ev(hru, hrv, self.l).reshape(3,-1)

        src =  mlab.pipeline.scalar_scatter(volpts[0,:], volpts[1,:], volpts[2,:], **kwargs)
        mlab.pipeline.volume(src, **kwargs)
        
        # Turn off perspective
        fig = mlab.gcf()
        fig.scene.camera.trait_set(parallel_projection=1)
        return fig


    def create_triangulation(self, ures=2, vres=2, **kwargs):
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

        if self.tri is not None:
            return self.tri
        
        # Make new u and v values of (possibly) higher resolution
        # the original ones.
        hru, hrv = self._resample_uv(ures, vres)
        volpts = self.ev(hru, hrv, self.l).reshape(3, -1).T

        tri = Delaunay(volpts)
        self.tri = tri
        
        return tri


    
    def plot_srf(self, ures=8, vres=8, **kwargs):
        """Alias for mplot_surface()
        """
        self.mplot_surface(ures=ures, vres=vres, **kwargs)

        
    def copy(self):
        """Get a copy of the volume
        """
        from copy import deepcopy
        return deepcopy(self)



def test_surface(u, v, l):
    import numpy as np
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])

def test_nodes():
    from rbf.nodes import snap_to_boundary,disperse,menodes
    from rbf.geometry import contains
    from alphavol import alpha_shape
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size)

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=1)

    tri = vol.create_triangulation()
    alpha = alpha_shape([], 120., tri=tri)
    
    # Define the problem domain
    vert = alpha.points
    smp  = np.asarray(alpha.bounds, dtype=np.int64)

    N = 10000 # total number of nodes
    
    # create N quasi-uniformly distributed nodes
    nodes, smpid = menodes(N,vert,smp,itr=20)
    
    # remove nodes outside of the domain
    in_nodes = nodes[contains(nodes,vert,smp)]

    from mayavi import mlab
    vol.mplot_surface(color=(0, 1, 0), opacity=0.33, ures=10, vres=10)

    mlab.points3d(*in_nodes.T, color=(1, 1, 0), scale_factor=15.0)
    
    mlab.show()

    return in_nodes, vol.inverse(in_nodes)

def test_mplot_surface():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    order = 1
    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=order)

    from mayavi import mlab

    vol.mplot_surface(color=(0, 1, 0), opacity=1.0, ures=10, vres=10)
    
    mlab.show()


def test_mplot_volume():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    order = 1
    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=order)

    from mayavi import mlab

    vol.mplot_volume(color=(0, 1, 0), opacity=1.0, ures=10, vres=10)
    
    mlab.show()

def test_uv_isospline():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    order = [1]
    for ii in xrange(len(order)):
        vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=order[ii])

        U, V = vol._resample_uv(5, 5)
        L = np.asarray([-1.0])

        nupts = U.shape[0]
        nvpts = V.shape[0]

    from mayavi import mlab
        
    U, V = vol._resample_uv(10, 10)
    L = np.asarray([1.0])
        
    nupts = U.shape[0]
    nvpts = V.shape[0]
    # Plot u,v-isosplines on the surface
    upts = vol(U, V[0], L)
    vpts = vol(U[int(nupts/2)], V, L)
    
    vol.mplot_surface(color=(0, 1, 0), opacity=1.0, ures=10, vres=10)
    
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

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=1)

    tri = vol.create_triangulation()
    
    return vol, tri
    

def test_point_distance():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    U, V = vol._resample_uv(5, 5)
    L = np.asarray([1.0, 0.0, -1.0])
    
    #dist, coords = vol.point_distance(U, V, L)
    #print dist
    #print coords
    dist, coords = vol.point_distance(U, V[0], L)
    print dist
    print coords
    dist, coords = vol.point_distance(U, V[0], L, axis_origin=np.median(obs_u))
    print dist
    print coords

    

    
if __name__ == '__main__':
    test_mplot_surface()
#    test_mplot_volume()
#    test_uv_isospline()
#    test_nodes()
#    test_tri()
#     test_point_distance()
     

    
    
    
