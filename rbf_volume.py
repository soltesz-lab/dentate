"""Implements a parametric volume as a 3-tuple of RBF instances, one each for u, v and l.
Based on code from bspline_surface.py
"""

import math, pickle
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
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


class RBFVolume(object):
    def __init__(self, u, v, l, xyz, order=1, basis=rbf.basis.phs3, vol_coeffs=None):
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

        if vol_coeffs is None:
            self._create_vol(u, v, l, xyz, order=order, basis=basis)
        else:
            self._create_vol_from_coeffs(u, v, l, xyz, vol_coeffs, order=order, basis=basis)

        self.u  = u
        self.v  = v
        self.l  = l
        self.xyz = xyz
        self.order = order

        self.tri = None
        self.facets = None
        self.facet_counts = None

    @classmethod
    def load(cls, filename):
        
        f = open(filename, "rb")
        s = pickle.load(f)
        f.close()

        return cls(**s)

    def save(self, filename, basis_name):

        vol_coeffs = ( self._xvol._coeff, self._yvol._coeff, self._zvol._coeff, \
                       self._uvol._coeff, self._vvol._coeff, self._lvol._coeff )
        
        s = { 'u': self.u, 'v': self.v, 'l': self.l, 'xyz': self.xyz, 'order': self.order, \
              'basis': basis_name, 'vol_coeffs': vol_coeffs }
        
        f = open(filename, "wb")
        pickle.dump(s, f)
        f.close()

    
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


    def _create_vol_from_coeffs(self, obs_u, obs_v, obs_l, xyz, vol_coeffs, **kwargs):

        coeff_x, coeff_y, coeff_z, coeff_u, coeff_v, coeff_l = vol_coeffs

        u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
        uvl_obs = np.array([u.ravel(),v.ravel(),l.ravel()]).T

        xvol = RBFInterpolant(uvl_obs,coeff=coeff_x,**kwargs)
        yvol = RBFInterpolant(uvl_obs,coeff=coeff_y,**kwargs)
        zvol = RBFInterpolant(uvl_obs,coeff=coeff_z,**kwargs)

        uvol = RBFInterpolant(xyz,coeff=coeff_u,**kwargs)
        vvol = RBFInterpolant(xyz,coeff=coeff_v,**kwargs)
        lvol = RBFInterpolant(xyz,coeff=coeff_l,**kwargs)

        self._xvol = xvol
        self._yvol = yvol
        self._zvol = zvol
        self._uvol = uvol
        self._vvol = vvol
        self._lvol = lvol


  
    def _resample_distance_strategy(self):
        from scipy.spatial import cKDTree

        u, v, l = self.u, self.v, self.l
        um, vm, lm = np.meshgrid(u,v,l,indexing='ij')
        uvl = np.array([um.ravel(), vm.ravel(), lm.ravel()]).T
        x, y, z = self._xvol(uvl), self._yvol(uvl), self._zvol(uvl)
        xyz = np.asarray([x,y,z],dtype='float32').T
        tree = cKDTree(xyz)

        distance_dictionary = {}
        distances = []
        N = xyz.shape[0]
        min_distance, min_index = 1.0e9, -1
        for n in range(N):
            d, i = tree.query(xyz[n],k=2)
            d, i = d[-1], i[-1]
            distance_dictionary[tuple(xyz[n])] = [d,n,i]
            distances.append(d)
            if d < min_distance:
                min_distance, min_index = d, (n,i)

        f = open('distances_curv.txt','w')
        for distance in distances:
            f.write(str(distance) + '\n')
        f.close() 

        f = open('add_points.txt','w')
        for n in range(N):
            ndistance, self_index, neighbor_index = distance_dictionary[tuple(xyz[n])]
            nearest_neighbor = xyz[neighbor_index]
            rel = np.abs(ndistance - min_distance) / min_distance
            points_to_add = 2 ** (rel + 2)
            distance_dictionary[tuple(xyz[n])].append(points_to_add)
            f.write(str(int(points_to_add)) + '\n')
        f.close()

        def sample_from_sphere(R,xyz):
           phi = np.random.uniform(0, 2.*np.pi)
           costheta = np.random.uniform(-1,1)
           u = np.random.uniform(0,1)
           theta = np.arccos(costheta)
           r = sphere_radius * (u ** (1. / 3.))
           xnew, ynew, znew = xyz[0] + r * np.sin(theta) * np.cos(phi), xyz[1] + r * np.sin(theta) * np.sin(phi), xyz[2] + r * np.cos(theta)
 
           xyz_new = np.asarray([xnew, ynew, znew], dtype='float32').reshape(1,3)
           u, v, l = self._uvol(xyz_new), self._vvol(xyz_new), self._lvol(xyz_new)
           u, v, l = u[0], v[0], l[0]
           if (u < np.min(self.u) or u > np.max(self.u)):
               return None, None
           if (v < np.min(self.v) or v > np.max(self.v)):
               return None, None
           if (l < np.min(self.l) or l > np.max(self.l)):
               return None, None
           return xyz_new, np.array([u,v,l],dtype='float32').reshape(1,3)

        for xyz_key in list(distance_dictionary.keys()):
            ndistance, _, _, points_to_add = distance_dictionary[xyz_key]
            
            sphere_centroid, sphere_radius = xyz_key, ndistance / 2.
            for i in range(int(points_to_add)):
                xyz_new = None
                while xyz_new is None:
                    xyz_new, uvl_new = sample_from_sphere(sphere_radius, sphere_centroid)
                xyz = np.concatenate((xyz, xyz_new))
                uvl = np.concatenate((uvl, uvl_new))

        f = open('distance_curv_added.txt','w')
        tree_added = cKDTree(xyz)
        
        N2 = xyz.shape[0]
        for n in range(N2):
            d, i = tree_added.query(xyz[n],k=2)
            d, i = d[-1], i[-1] 
            f.write(str(d) + '\n')
        f.close()
        return xyz, uvl

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
            U = np.asarray(su)
            V = np.asarray(sv)
            L = np.asarray(sl)
            assert(len(U) == len(V))
            assert(len(U) == len(L))


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

        
    def point_distance(self, su, sv, sl, axis=0, interp_chunk_size=1000, return_coords=True, mesh=True):
        """Cumulative distance along an axis between arrays of (u, v, l) coordinates.

        Parameters
        ----------
        u, v, l : array-like

        axis: axis along which the distance should be computed

        mesh: calculate distances on a meshgrid, i.e. if axis=0, compute
        u-coordinate distances for all values of v and l (default: True)

        return_coords: if True, returns the coordinates for which distances were computed (default: True)

        Returns
        -------
        If the lengths of u and v are at least 2, returns the cumulative length
        between each u,v pair.
        """
        u = np.array([su]).reshape(-1,)
        v = np.array([sv]).reshape(-1,)
        l = np.array([sl]).reshape(-1,)

        assert(len(u) > 0)
        assert(len(v) > 0)
        assert(len(l) > 0)

        if not mesh:
            assert(len(u) == len(v))
            assert(len(u) == len(l))
        
        input_axes = [u, v, l]

        c = input_axes

        ordered_axes = [ np.sort(c[i]) if i == axis else c[i] for i in range(0,3) ]

        aidx = list(range(0,3))
        aidx.remove(axis)
        
        distances = []
        coords    = []

        npts = ordered_axes[axis].shape[0]
        if npts > 1:
            if mesh:
                (eval_pts, eval_coords) = self.ev(*ordered_axes, chunk_size=interp_chunk_size, return_coords=True)
                coord_idx = np.argsort(eval_coords[:,axis])
                all_pts = (eval_pts.reshape(3, -1).T)[coord_idx,:]
                all_pts_coords = eval_coords[coord_idx,:]
                split_pts = np.split(all_pts, npts)
                split_pts_coords = np.split(all_pts_coords, npts)
                cdist = np.zeros((split_pts[0].shape[0],1))
                distances.append(cdist)
                if return_coords:
                    cind = np.lexsort(tuple([ split_pts_coords[0][i] for i in aidx ]))
                    coords.append(split_pts_coords[0][cind])
                for i in range(0, npts-1):
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
                    distances.append(cdist)
                    if return_coords:
                        coords.append(a_coords[aind])
            else:
                (eval_pts, eval_coords) = self.ev(*ordered_axes, chunk_size=interp_chunk_size, mesh=False, return_coords=True)
                coord_idx = np.argsort(eval_coords[:,axis])
                all_pts   = (eval_pts.reshape(3, -1).T)[coord_idx,:]
                a  = all_pts[1:,:]
                b  = all_pts[:-1,:]
                a_coords = eval_coords[1:,:]
                b_coords = eval_coords[:-1,:]
                aind = np.lexsort(tuple([ a_coords[:,i] for i in aidx ]))
                bind = np.lexsort(tuple([ b_coords[:,i] for i in aidx ]))
                a_sorted = a[aind]
                b_sorted = b[bind]
                dist     = euclidean_distance(a_sorted, b_sorted).reshape(-1,1)
                distances = np.cumsum(dist)
                if return_coords:
                    coords = a_coords[aind]
                        
        if return_coords:
            return distances, coords
        else:
            return distances

        
    def boundary_distance(self, axis, b1, b2, coords, resolution=0.01):
        """Given U,V,L coordinates returns the distances of the points
        to the U, V boundaries in the corresponding L layer.

        Parameters
        ----------
        - axis - axis along which to compute distance
        - b1, b2 - boundary values
        - coords - U,V,L coordinates
        - resolution - discretization resolution in UVL space for distance calculation

        Returns
        -------
        - dist1, dist2 - distances to the b1 and b2 boundaries
        """
        ## Distance from b1 boundary to coordinate
        d1   = np.abs(b1 - coords[axis])
        ps1  = np.linspace(b1, coords[axis], int(d1/resolution))
        if len(ps1) > 1:
            p_grid1  = [ ps1 if i == axis else coords[i] for i in range(0,3) ]
            p_u, p_v, p_l = np.meshgrid(*p_grid1)
            p_dist1  = self.point_distance(p_u.ravel(), p_v.ravel(), p_l.ravel(),
                                           axis=axis, mesh=False, return_coords=False)[-1]
        else:
            p_dist1 = 0.

        ## Distance from coordinate to b2 boundary
        d2  = np.abs(b2 - coords[axis])
        ps2 = np.linspace(coords[axis], b2, int(d2/resolution))
        if len(ps2) > 1:
            p_grid2  = [ ps2 if i == axis else coords[i] for i in range(0,3) ]
            p_u, p_v, p_l = np.meshgrid(*p_grid2)
            p_dist2  = self.point_distance(p_u.ravel(), p_v.ravel(), p_l.ravel(),
                                           axis=axis, mesh=False, return_coords=False)[-1]
        else:
            p_dist2 = 0.

        return p_dist1, p_dist2

    
    def point_position(self, su, sv, sl, resolution=0.01, return_extent=True):
        """Given U,V,L coordinates returns the positions of the points
        relative to the U, V boundaries in the corresponding L layer.

        Parameters
        ----------
        u, v, l : array-like

        Returns
        -------
        - pos - relative position along U, V axes
        - extents - maximum extents along U and V for the given L
        """
        u = np.array([su]).reshape(-1,)
        v = np.array([sv]).reshape(-1,)
        l = np.array([sl]).reshape(-1,)

        assert(len(u) == len(v))
        assert(len(u) == len(l))

        
        uvl = np.array([u.ravel(),v.ravel(),l.ravel()]).T
        npts = uvl.shape[0]

        pos = []
        extents = []
        for i in range(0, npts):
            u_dist1, u_dist2 = self.boundary_distance(0, self.u[0], self.u[-1], uvl[i,:], resolution=resolution)

            u_extent = u_dist1 + u_dist2
            u_pos = u_dist1 / u_extent

            v_dist1, v_dist2 = self.boundary_distance(1, self.v[0], self.v[-1], uvl[i,:], resolution=resolution)
            
            v_extent = v_dist1 + v_dist2
            v_pos = v_dist1 / v_extent

            pos.append((u_pos, v_pos))
            extents.append((u_extent, v_extent))

        if return_extent:
            return (pos, extents)
        else:
            return pos
        
        
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

        if not 'color' in kwargs:
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
        meshpts1 = self.ev(hru, hrv, np.max(self.l))
        meshpts2 = self.ev(hru, hrv, np.min(self.l))
        
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

        if not 'color' in kwargs:
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


    def create_triangulation(self, ures=4, vres=4, lres=1, **kwargs):
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
        hru, hrv, hrl = self._resample_uvl(ures, vres, lres)
        
        volpts = self.ev(hru, hrv, hrl).reshape(3, -1).T
        tri = Delaunay(volpts, **kwargs)
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



def test_surface(u, v, l, rotate=None):
    import numpy as np

    if rotate is not None:
        for i in range(0, 3):
            if rotate[i] != 0.:
                a = float(np.deg2rad(rotate[i]))
                rot = rotate3d([ 1 if i == j else 0 for j in range(0,3) ], a)
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


def test_nodes():
    from rbf.nodes import snap_to_boundary,disperse,menodes
    from rbf.geometry import contains
    from .alphavol import alpha_shape
    
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
    for ii in range(len(order)):
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
    

def test_point_distance_mesh():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, num=3)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    U, V = vol._resample_uv(5, 5)
    L = np.asarray([1.0, 0.0, -1.0])
    
    dist, coords = vol.point_distance(U, V[0], L, axis=0)
    print(dist)
    print(coords)
    dist, coords = vol.point_distance(U[0], V, L, axis=1)
    print(dist)
    print(coords)


def test_point_distance():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    U, V = vol._resample_uv(5, 5)
    L = np.asarray([1.0, 0.0, -1.0])
    
    dist, coords = vol.point_distance(U, np.full((U.shape[0],1),V[10]), np.full((U.shape[0],1),L[1]), axis=0, mesh=False)
    print(dist)
    print(coords)
    dist, coords = vol.point_distance(np.full((V.shape[0],1),U[10]), V, np.full((V.shape[0],1),L[1]), axis=1, mesh=False)
    print(dist)
    print(coords)


def test_point_position():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-1.0, 1., num=3)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    U, V = vol._resample_uv(5, 5)
    L = np.asarray([1.0, 0.0, -1.0])
    
    print(vol.point_position(np.median(U), np.median(V), np.max(L)))
    print(vol.point_position(1.0, np.median(V), np.max(L)))

def test_precision():
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 25)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 25)
    obs_l = np.linspace(-1.0, 1., num=10)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    test_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 250)
    test_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 250)
    test_l = np.linspace(-1.0, 1., num=10)

    u, v, l = np.meshgrid(test_u, test_v, test_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size).T

    interp_xyz = vol(u, v, l, mesh=False).reshape(3, u.size).T

    error = xyz - interp_xyz
    print(('Min error: %f' % np.min(error)))
    print(('Max error: %f' % np.max(error)))
    print(('Mean error: %f' % np.mean(error)))
    
    
def test_alphavol():
    from .alphavol import alpha_shape
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 20)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 20)
    obs_l = np.linspace(-3.95, 3.2, num=10)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l, rotate=[-35., 0., 0.])

    print ('Constructing volume...')
    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    print ('Constructing volume triangulation...')
    tri = vol.create_triangulation()

    print ('Constructing alpha shape...')
    alpha = alpha_shape([], 120., tri=tri)

    vert = alpha.points
    smp  = np.asarray(alpha.bounds, dtype=np.int64)

    edges = np.vstack([np.column_stack([smp[:,0],smp[:,1]]), \
                       np.column_stack([smp[:,1],smp[:,2]])])

    x = vert[:,0]
    y = vert[:,1]
    z = vert[:,2]

    start_idx = edges[:,0]
    end_idx = edges[:,1]
    
    from mayavi import mlab
    vol.mplot_surface(color=(0, 1, 0), opacity=0.33, ures=10, vres=10)
    mlab.quiver3d(x[start_idx],
                  y[start_idx],
                  z[start_idx],
                  x[end_idx] - x[start_idx],
                  y[end_idx] - y[start_idx],
                  z[end_idx] - z[start_idx],
                  mode='2ddash',
                  scale_factor=1)
    
    
    mlab.show()
    

def test_tri():

    max_u = 11690.
    max_v = 2956.
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 30)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 30)
    obs_l = np.linspace(-1.0, 1., num=5)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l).reshape(3, u.size)

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    tri = vol.create_triangulation()
    
    return vol, tri

def test_load():

    max_u = 11690.
    max_v = 2956.
    
    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, 30)
    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, 30)
    obs_l = np.linspace(-1.0, 1., num=5)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = test_surface (u, v, l)

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, order=2)

    vol.save('vol.p', 'phs3')
    vol_from_file = RBFVolume.load('vol.p')

    print((vol(0.5, 0.5, 0.5)))
    print((vol_from_file(0.5, 0.5, 0.5)))
    
    
if __name__ == '__main__':
#    test_precision()
#    test_point_position()
#    test_point_distance_mesh()
#    test_point_distance()
#    test_mplot_surface()
#    test_mplot_volume()
#    test_uv_isospline()
#    test_nodes()
#    test_alphavol()
#    test_tri()
     test_load()
