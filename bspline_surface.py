"""Implements a b-spline surface as a 3-tuple of
scipy.interpolate.RectBivariateSpline instances, one
each for x, y and z.
Code from https://gist.github.com/subnivean/c622cc2b58e6376263b8.js
"""

import math
import numpy as np
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.spatial.distance import cdist
from collections import namedtuple

PointCloud = namedtuple('PointCloud', ['tree', 'U', 'V'], verbose=True)

def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a-b)**2,axis=1))


vdist = np.vectorize(euclidean_distance)


def normcoords(su, sv):
    nu = np.linspace(0, 1, len(su))
    nv = np.linspace(0, 1, len(sv))
    ip_u = InterpolatedUnivariateSpline(su, nu)
    ip_v = InterpolatedUnivariateSpline(sv, nv)

    return nu, ip_u, nv, ip_v

def rotate3d(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    print axis
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

class BSplineSurface(object):
    def __init__(self, u, v, xyz, ku=3, kv=3, bbox=[0, 1, 0, 1], s=0,
                 controlpts=False, U=None, V=None):
        """Parametric (u,v) surface approximation over a rectangular mesh.

        Parameters
        ----------
        u, v : array_like
            1-D arrays of coordinates.
        xyz : array_like
            3-D array of (x, y, z) data with shape (3, u.size, v.size).
        bbox : array_like, optional
            Sequence of length 4 specifying the boundary of the rectangular
            approximation domain. See scipy.interpolate.RectBivariateSpline
            for more info.
        ku, kv : ints, optional
            Degrees of the bivariate spline. Default is 3 for each.
        controlpts : boolean
            Indicates if the xyz points being passed are points to spline
            *through*, or are already control points as defined in some
            other format (e.g. the points from 'stepparser', which
            returns the control points as defined in the STEP file).
        U, V : array_like, optional
            Knot vectors in u and v direction, as parsed from a STEP
            file or similar
        """

        if controlpts is True:
            assert U is not None, \
                "Knot vector `U` must be passed when `controlpts` is True"
            assert V is not None, \
                "Knot vector `V` must be passed when `controlpts` is True"

        nu, ip_u, nv, ip_v = normcoords(u, v)

        self._create_srf(nu, nv, xyz, ku, kv, bbox, controlpts, s, U, V)

        self.ip_u = ip_u
        self.ip_v = ip_v
        self.bbox = bbox
        self.u = nu
        self.v = nv
        self.su = u
        self.sv = v
        self.ku = ku
        self.kv = kv

    def __call__(self, *args, **kwargs):
        """Convenience to allow evaluation of a BSplineSurface
        instance via `foosrf(0, 0)` instead of `foosrf.ev(0, 0)`,
        mostly to be consistent with the evaluation of
        BSpline objects (and other interpolators, such as
        `scipy.InterpolatedUnivariateSpline`).
        """
        return self.ev(*args, **kwargs)

    def _create_srf(self, u, v, xyz, ku, kv, bbox,
                    controlpts, s, U, V):

        # Create surface definitions
        xsrf = RectBivariateSpline(u, v, xyz[0], bbox=bbox, kx=ku, ky=kv, s=s)
        ysrf = RectBivariateSpline(u, v, xyz[1], bbox=bbox, kx=ku, ky=kv, s=s)
        zsrf = RectBivariateSpline(u, v, xyz[2], bbox=bbox, kx=ku, ky=kv, s=s)

        if controlpts is True:
            # A little back-dooring here - replace the calculated
            # control points with the *actual* control points, as
            # passed in.
            X = xyz[0].ravel()
            Y = xyz[1].ravel()
            Z = xyz[2].ravel()
            # Note that U and V must be passed to the constructor
            # if 'controlpts' is True - these were also explicitly
            # defined in something like a STEP file, for instance.
            xsrf.tck = (U, V, X)
            ysrf.tck = (U, V, Y)
            zsrf.tck = (U, V, Z)
        elif U is not None or V is not None:
            if U is not None:
                xsrf.tck = (U, xsrf.tck[1], xsrf.tck[2])
                ysrf.tck = (U, ysrf.tck[1], ysrf.tck[2])
                zsrf.tck = (U, zsrf.tck[1], zsrf.tck[2])
            if V is not None:
                xsrf.tck = (xsrf.tck[0], V, xsrf.tck[2])
                ysrf.tck = (ysrf.tck[0], V, ysrf.tck[2])
                zsrf.tck = (zsrf.tck[0], V, zsrf.tck[2])

        self._xsrf = xsrf
        self._ysrf = ysrf
        self._zsrf = zsrf

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

    def ev(self, su, sv, mesh=True, normalize_uv=False, return_uv=False):
        """Get point(s) on surface at (u, v).

        Parameters
        ----------
        u, v : scalar or array-like
            u and v may be scalar or vector
        normalize_uv: 
        return_uv: 

        mesh : boolean
            If True, will expand the u and v values into a mesh.
            For example, with u = [0, 1] and v = [0, 1]: if 'mesh'
            is True, the surface will be evaluated at [0, 0], [0, 1],
            [1, 0] and [1, 1], while if it is False, the evaluation
            will only be made at [0, 0] and [1, 1]

        Returns
        -------
        If scalar values are passed for *both* u and v, returns
        a 1-D 3-element array [x,y,z]. Otherwise, returns an array
        of shape 3 x len(u) x len(v), suitable for feeding to Mayavi's
        mlab.mesh() plotting function (as mlab.mesh(*arr)).
        """
        if normalize_uv:
            u = np.array([self.ip_u(su)]).reshape(-1,)
            v = np.array([self.ip_v(sv)]).reshape(-1,)
        else:
            u = np.array([su]).reshape(-1,)
            v = np.array([sv]).reshape(-1,)
        if mesh:
            # I'm still not sure why we're required to flip u and v
            # below, but trust me, it doesn't work otherwise.
            V, U = np.meshgrid(v, u)
            U = U.ravel()
            V = V.ravel()
        else:
            if len(u) != len(v): # *Need* to mesh this, like above!
                if len(v) < len(u):
                    V, U = np.meshgrid(v, u)
                else:
                    U, V = np.meshgrid(u, v)
                U = U.ravel()
                V = V.ravel()
            else:
                U, V = u, v
        x = self._xsrf.ev(U, V)
        y = self._ysrf.ev(U, V)
        z = self._zsrf.ev(U, V)

        if u.shape == (1,) and v.shape == (1,):
            # Scalar u and v values; return 1-D 3-element array.
            arr = np.array([x, y, z]).ravel()
        else:
            # u and/or v passed as lists; return 3 x m x n array,
            # where m is len(u) and n is len(v). This format
            # is compatible with mayavi's mlab.mesh()
            # function.
            arr = np.array([x, y, z]).reshape(3, len(u), -1)
        if return_uv:
            return (arr, U, V)
        else:
            return arr


    def utan(self, su, sv, normalize=True, mesh=True, normalize_uv=False):
        if normalize_uv:
            u = np.array([self.ip_u(su)]).reshape(-1,)
            v = np.array([self.ip_v(sv)]).reshape(-1,)
        else:
            u = np.array([su]).reshape(-1,)
            v = np.array([sv]).reshape(-1,)

        dxdu = self._xsrf(u, v, dx=1, dy=0, grid=mesh)
        dydu = self._ysrf(u, v, dx=1, dy=0, grid=mesh)
        dzdu = self._zsrf(u, v, dx=1, dy=0, grid=mesh)
        du = np.array([dxdu, dydu, dzdu]).T

        if mesh is True:
            du = du.swapaxes(0, 1)
        else:
            du = du[:, np.newaxis, :]

        if normalize:
            du /= np.sqrt((du**2).sum(axis=2))[:, :, np.newaxis]

        if u.shape == (1,) and v.shape == (1,):
            return du.reshape(3)
        else:
            arr = du.transpose(2, 0, 1)
            return arr

    def vtan(self, su, sv, normalize=True, mesh=True, normalize_uv=False):
        if normalize_uv:
            u = np.array([self.ip_u(su)]).reshape(-1,)
            v = np.array([self.ip_v(sv)]).reshape(-1,)
        else:
            u = np.array([su]).reshape(-1,)
            v = np.array([sv]).reshape(-1,)

        dxdv = self._xsrf(u, v, dx=0, dy=1, grid=mesh)
        dydv = self._ysrf(u, v, dx=0, dy=1, grid=mesh)
        dzdv = self._zsrf(u, v, dx=0, dy=1, grid=mesh)
        dv = np.array([dxdv, dydv, dzdv]).T

        if mesh is True:
            dv = dv.swapaxes(0, 1)
        else:
            dv = dv[:, np.newaxis, :]

        if normalize:
            dv /= np.sqrt((dv**2).sum(axis=2))[:, :, np.newaxis]

        if u.shape == (1,) and v.shape == (1,):
            return dv.reshape(3)
        else:
            arr = dv.transpose(2, 0, 1)
            return arr


    def normal(self, su, sv, mesh=True, normalize_uv=False):
        """Get normal(s) at (u, v).

        Parameters
        ----------
        u, v : scalar or array-like
            u and v may be scalar or vector (see below)
        normalize_uv: 

        Returns
        -------
        If scalar values are passed for *both* u and v, returns
        a 1-D 3-element array [x,y,z]. Otherwise, returns an array
        of shape 3 x len(u) x len(v), suitable for feeding to Mayavi's
        mlab.mesh() plotting function (as mlab.mesh(*arr)).
        """
        if normalize_uv:
            u = np.array([self.ip_u(su)]).reshape(-1,)
            v = np.array([self.ip_v(sv)]).reshape(-1,)
        else:
            u = np.array([su]).reshape(-1,)
            v = np.array([sv]).reshape(-1,)

        dxdus = self._xsrf(u, v, dx=1, grid=mesh)
        dydus = self._ysrf(u, v, dx=1, grid=mesh)
        dzdus = self._zsrf(u, v, dx=1, grid=mesh)
        dxdvs = self._xsrf(u, v, dy=1, grid=mesh)
        dydvs = self._ysrf(u, v, dy=1, grid=mesh)
        dzdvs = self._zsrf(u, v, dy=1, grid=mesh)

        normals = np.cross([dxdus, dydus, dzdus],
                           [dxdvs, dydvs, dzdvs],
                           axisa=0, axisb=0)
        if mesh is False:
            normals = normals[:, np.newaxis, :]

        normals /= np.sqrt((normals**2).sum(axis=2))[:, :, np.newaxis]

        if u.shape == (1,) and v.shape == (1,):
            return normals.reshape(3)
        else:
            arr = normals.transpose(2, 0, 1)
            return arr

        
    def point_distance(self, su, sv, normalize_uv=False):
        """Cumulative distance between pairs of (u, v) coordinates.

        Parameters
        ----------
        u, v : array-like
        normalize_uv: 
 
        Returns
        -------
        If the lengths of u and v are at least 2, returns the total arc length
        between each u,v pair.
        """
        if normalize_uv:
            u = np.array([self.ip_u(su)]).reshape(-1,)
            v = np.array([self.ip_v(sv)]).reshape(-1,)
        else:
            u = np.array([su]).reshape(-1,)
            v = np.array([sv]).reshape(-1,)

        npts   = max(u.shape[0], v.shape[0])
        pts    = self.ev(u, v, mesh=False).T.reshape(npts, 3)

        del u, v
        distance = 0
        
        if npts > 1:
            a = pts[1:,:]
            b = pts[0:npts-1,:]
            distance = np.sum(euclidean_distance(a, b))

                
        return distance

    def point_cloud(self, res = 5):
        """Creates a point cloud using `scipy.spatial.cKDTree`

        Parameters
        ----------
        res : int
            Specifies the oversampling of the original
            surface in u and v directions. For example:
            if `res` = 2, and `self.u` = [0, 1, 2, 3],
            then the surface will be resampled at
            [0, 0.5, 1, 1.5, 2, 2.5, 3] prior to
            plotting.

        kwargs : dict
            See scipy docs for `scipy.spatial.cKDTree`

        Returns
        -------
            Tuple (cKDTree, U coordinates, V coordinates, normalized U coordinates, normalized V coordinates)
        """
        from scipy.spatial import cKDTree
        # Make new u and v values of (possibly) higher resolution
        # the original ones.
        hru, hrv = self._resample_uv(res, res)
        # Sample the surface at the new u, v values
        meshpts, U, V = self.ev(hru, hrv, mesh=True, return_uv=True)
        # Create kdTree
        tree = cKDTree(meshpts.reshape(3, hru.shape[0] * hrv.shape[0]).T)
        return PointCloud(tree, U, V)
        

    def mplot(self, ures=8, vres=8, **kwargs):
        """Plot the surface using Mayavi's `mesh()` function

        Parameters
        ----------
        ures, vres : int
            Specifies the oversampling of the original
            surface in u and v directions. For example:
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
        meshpts = self.ev(hru, hrv, mesh=True)
        mlab.mesh(*meshpts, **kwargs)
        # Turn off perspective
        fig = mlab.gcf()
        fig.scene.camera.trait_set(parallel_projection=1)

        
    def plot(self, ures=8, vres=8, **kwargs):
        """Alias for mplot()
        """
        self.mplot(ures=ures, vres=vres, **kwargs)

        
    def flipu(self):
        """Flips the u-direction of the surface

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        xcoeffs = self._xsrf.get_coeffs()
        ycoeffs = self._ysrf.get_coeffs()
        zcoeffs = self._zsrf.get_coeffs()
        xuknots, xvknots = self._xsrf.get_knots()
        yuknots, yvknots = self._ysrf.get_knots()
        zuknots, zvknots = self._zsrf.get_knots()

        ulen = len(self.u)
        vlen = len(self.v)
        xcoeffs = xcoeffs.reshape(ulen, vlen)[-1::-1, :].ravel()
        ycoeffs = ycoeffs.reshape(ulen, vlen)[-1::-1, :].ravel()
        zcoeffs = zcoeffs.reshape(ulen, vlen)[-1::-1, :].ravel()

        xuknots = (1 - xuknots)[-1::-1]
        yuknots = (1 - yuknots)[-1::-1]
        zuknots = (1 - zuknots)[-1::-1]

        self.u = (1 - self.u)[-1::-1]
        bbox = self.bbox
        self.bbox = [1 - bbox[1], 1 - bbox[0], bbox[2], bbox[3]]

        self._xsrf.tck = (xuknots, xvknots, xcoeffs)
        self._ysrf.tck = (yuknots, yvknots, ycoeffs)
        self._zsrf.tck = (zuknots, zvknots, zcoeffs)

        
    def flipv(self):
        """Flips the v-direction of the surface.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        xcoeffs = self._xsrf.get_coeffs()
        ycoeffs = self._ysrf.get_coeffs()
        zcoeffs = self._zsrf.get_coeffs()
        xuknots, xvknots = self._xsrf.get_knots()
        yuknots, yvknots = self._ysrf.get_knots()
        zuknots, zvknots = self._zsrf.get_knots()

        ulen = len(self.u)
        vlen = len(self.v)
        xcoeffs = xcoeffs.reshape(ulen, vlen)[:, -1::-1].ravel()
        ycoeffs = ycoeffs.reshape(ulen, vlen)[:, -1::-1].ravel()
        zcoeffs = zcoeffs.reshape(ulen, vlen)[:, -1::-1].ravel()

        xvknots = (1 - xvknots)[-1::-1]
        yvknots = (1 - yvknots)[-1::-1]
        zvknots = (1 - zvknots)[-1::-1]

        self.v = (1 - self.v)[-1::-1]
        bbox = self.bbox
        self.bbox = [bbox[0], bbox[1], 1 - bbox[3], 1 - bbox[2]]

        self._xsrf.tck = (xuknots, xvknots, xcoeffs)
        self._ysrf.tck = (yuknots, yvknots, ycoeffs)
        self._zsrf.tck = (zuknots, zvknots, zcoeffs)

    def flipboth(self):
        self.flipu()
        self.flipv()

        
    def copy(self):
        """Get a copy of the surface
        """
        from copy import deepcopy
        return deepcopy(self)

    
    def swapuv(self, flipdir=None):
        """Swap u and v directions. In-place modification.

        Parameters
        ----------
        flipdir : Optional; one of ('u', 'v') if not `None`
            Direction to reverse to maintain surface normal direction

        Returns
        -------
        None
        """
        if flipdir is not None:
            flipdir = flipdir.lower()
            DIRS = ('u', 'v')
            assert flipdir in DIRS, \
               "Invalid value for `flipdir`; must be one of " + DIRS.__repr__()

        # Swap the bounding box numbers
        obbox = self.bbox
        swbbox = [obbox[2], obbox[3], obbox[0], obbox[1]]

        # Note that the method below gives *exactly* the same
        # surface as the original, judging by the amount of 'speckling'
        # seen when 'before' and 'after' surfaces are plotted in Mayavi
        # (i.e. there is *no* speckling - the 'after' surface
        # completely replaces the 'before' surface).
        U, V = self.u, self.v
        ssrf = BSplineSurface(V, U, self(U, V).swapaxes(1, 2),
                              ku=self.kv, kv=self.ku, bbox=swbbox)

        if flipdir is not None:
            if flipdir == 'u':
                ssrf.flipu()
            else:
                ssrf.flipv()

        # Re-assign all attributes
        self.__dict__ = ssrf.__dict__

    @property
    def uknots(self):
        """Return the knot vector in the u-parameter direction
        """
        return self._xsrf.tck[0]

    @property
    def vknots(self):
        """Return the knot vector in the v-parameter direction
        """
        return self._xsrf.tck[1]


def test_surface(u, v, l):
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])

def test_normal():

    spatial_resolution = 10.  # um
    max_u = 11690.
    max_v = 2956.
    
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(su, sv, indexing='ij')
    l = -1.
    xyz = test_surface (u, v, l)

    srf = BSplineSurface(su, sv, xyz)

    u = -0.020377
    v = 3.971938

    print 'Normal: ', srf.ev(u, v, normalize_uv=True), srf.normal(u, v, normalize_uv=True)
    
def test_point_distance():

    spatial_resolution = 10.  # um
    max_u = 11690.
    max_v = 2956.
    
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(su, sv, indexing='ij')
    l = -1.
    xyz = test_surface (u, v, l)

    srf = BSplineSurface(su, sv, xyz)

    destination_u = -0.020377
    destination_v = 3.971938
    source_u = 0.091337
    source_v = 1.932854

    npts=100
    
    U = np.linspace(destination_u, source_u, npts)
    V = np.linspace(destination_v, source_v, npts)

    source_distance_u = srf.point_distance(U, destination_v, normalize_uv=True)
    source_distance_v = srf.point_distance(destination_u, V, normalize_uv=True)

    print source_distance_u
    print source_distance_v

    
def test_uv_isospline():

    spatial_resolution = 1.  # um
    max_u = 11690.
    max_v = 2956.
    
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(su, sv, indexing='ij')
    # for the middle of the granule cell layer:
    l = -1.

    xyz = test_surface (u, v, l)

    s = [3e9, 2e9, 1e9, 1e8, 1e6, 0]
    for ii in xrange(len(s)):
        srf = BSplineSurface(np.linspace(0, 1, len(u)),
                             np.linspace(0, 1, xyz.shape[2]),
                             xyz, s=s[ii])

        for npts in [10, 100, 1000, ]:

            U = np.linspace(0, 1, npts)
            V = np.linspace(0, 1, npts)
            
            print 's = ', s[ii], ' npts = ', npts
            
            print srf.point_distance(U[0], V)
            print srf.point_distance(V, U[0])
            print srf.point_distance(U[int(npts/2)], V)
            print srf.point_distance(V, U[int(npts/2)])

    try:
        from mayavi import mlab

        npts = 400
        
        U = np.linspace(0, 1, npts)
        V = np.linspace(0, 1, npts)

        # Plot u,v-isosplines on the surface
        upts = srf(U, V[int(npts/2)]).reshape(3, npts)
        vpts = srf(U[int(npts/2)], V).reshape(3, npts)

        srf.mplot(color=(0, 1, 0), opacity=1.0, ures=1, vres=1)
        
        mlab.points3d(*upts,  scale_factor=100.0, color=(1, 1, 0))
        mlab.points3d(*vpts,  scale_factor=100.0, color=(1, 1, 0))
        
        mlab.show()
    except:
        pass

def pts_mplot():

    spatial_resolution = 50.  # um
    max_u = 11690.
    max_v = 2956.
    
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(su, sv, indexing='ij')

    ls = [-3.95, -1.95, 0., 1., 2., 3.]

    xyzs = [test_surface (u, v, l) for l in ls]

    s = 1e6
    srfs = [BSplineSurface(np.linspace(0, 1, len(u)),
                           np.linspace(0, 1, xyz.shape[2]),
                           xyz, s=s) for xyz in xyzs]
    print 'surfaces created'

    import h5py
    f = h5py.File("datasets/dentate_Full_Scale_Control_coords_distances_20171109.h5")
    x_coords = f['Populations']['GC']['Coordinates']['X Coordinate']['Attribute Value'][:]
    y_coords = f['Populations']['GC']['Coordinates']['Y Coordinate']['Attribute Value'][:]
    z_coords = f['Populations']['GC']['Coordinates']['Z Coordinate']['Attribute Value'][:]
    sample_inds = np.random.randint(0, x_coords.size-1, size=int(10000))
    pts = np.stack([x_coords[sample_inds], y_coords[sample_inds], z_coords[sample_inds]],axis=0)
    f.close()
    a = float(np.deg2rad(35.))
    r = rotate3d([1,0,0], a)
    pts = np.dot(r, pts)
    try:

        from mayavi import mlab

        mlab.figure(bgcolor=(1.,1.,1.), size=(1000, 1000))

         
        for srf in srfs:
            srf.mplot(color=(0.2, 0.7, 0.9), opacity=0.33, scale_factor=10.0, ures=1, vres=1)

        mlab.points3d(*pts, scale_factor=50.0, color=(1, 1, 0))

        
        mlab.show()
    except:
        pass

    
def test_point_cloud():

    spatial_resolution = 50.  # um
    max_u = 11690.
    max_v = 2956.
    
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    su = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    sv = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(su, sv, indexing='ij')

    xyz = test_surface (u, v, -1.)
    srf = BSplineSurface(su, sv, xyz1)

    return srf.point_cloud()
    
    
if __name__ == '__main__':
#    test_normal()
#    test_point_distance()
#    test_uv_isospline()
#    pts_mplot()
     test_point_cloud()
    
    
