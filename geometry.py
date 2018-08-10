
"""Classes and procedures related to neuronal geometry and distance calculation."""

import sys, time, gc, itertools
from collections import defaultdict
from mpi4py import MPI
import numpy as np
import rbf, rbf.basis
from rbf.interpolate import RBFInterpolant
from rbf.nodes import snap_to_boundary,disperse,menodes
from rbf.geometry import contains
from dentate.alphavol import alpha_shape
from dentate.rbf_volume import RBFVolume, rotate3d
from dentate.rbf_surface import RBFSurface
from dentate import utils
from neuroh5.io import NeuroH5CellAttrGen, bcast_cell_attributes, read_population_ranges, append_graph

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = utils.get_module_logger(__name__)

max_u = 11690.
max_v = 2956.

DG_u_extent = (-0.016*np.pi, 1.01*np.pi)
DG_v_extent = (-0.23*np.pi, 1.425*np.pi)
DG_l_extent = (-3.95, 3.1)

def DG_volume(u, v, l, rotate=None):
    """Parametric equations of the dentate gyrus volume."""
    
    u = np.array([u]).reshape(-1,)
    v = np.array([v]).reshape(-1,)
    l = np.array([l]).reshape(-1,)

    if rotate is not None:
        for i in xrange(0, 3):
            if rotate[i] != 0.:
                a = float(np.deg2rad(rotate[i]))
                rot = rotate3d([ 1 if i == j else 0 for j in xrange(0,3) ], a)
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


def DG_meshgrid(extent_u, extent_v, extent_l, resolution=[30, 30, 10], rotate=None, return_uvl=False):
    ures, vres, lres = resolution

    obs_u = np.linspace(extent_u[0], extent_u[1], num=ures)
    obs_v = np.linspace(extent_v[0], extent_v[1], num=vres)
    obs_l = np.linspace(extent_l[0], extent_l[1], num=lres)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = DG_volume(u, v, l, rotate=rotate)
    
    if return_uvl:
        return xyz, obs_u, obs_v, obs_l
    else:
        return xyz

def make_volume(extent_u, extent_v, extent_l, rotate=None, basis=rbf.basis.phs3, order=2, resolution=[30, 30, 10], return_xyz=False):  
    """Creates an RBF volume based on the parametric equations of the dentate volume."""
    
    xyz, obs_u, obs_v, obs_l = DG_meshgrid(extent_u, extent_v, extent_l,\
                                           rotate=rotate, resolution=resolution, return_uvl=True)
    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, basis=basis, order=order)

    if return_xyz:
        return vol, xyz
    else:
        return vol


def make_surface(extent_u, extent_v, obs_l, rotate=None, basis=rbf.basis.phs2, order=1, res=[33, 30]):  
    """Creates an RBF surface based on the parametric equations of the dentate volume.
    """
    ures = resolution[0]
    vres = resolution[1]

    obs_u = np.linspace(extent_u[0], extent_u[1], num=ures)
    obs_v = np.linspace(extent_v[0], extent_v[1], num=vres)

    u, v = np.meshgrid(obs_u, obs_v, indexing='ij')
    xyz = DG_volume (u, v, obs_l, rotate=rotate)

    srf = RBFSurface(obs_u, obs_v, xyz, basis=basis, order=order)

    return srf


def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a-b)**2,axis=1))


def make_uvl_distance(xyz_coords,rotate=None):
      f = lambda u, v, l: euclidean_distance(DG_volume(u,v,l,rotate=rotate), xyz_coords)
      return f


def get_volume_distances (ip_vol, origin_spec=None, rotate=None, nsample=250, alpha_radius=120., optiter=200, nodeitr=20):
    """Computes arc-distances along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    ip_vol : RBFVolume
        An interpolated volume instance of class RBFVolume.
    origin_coords : array(float)
        Origin point to use for distance computation.
    rotate : (float,float,float)
        Rotation angle (optional)
    nsample : int
        Number of points to sample inside the volume.
    alpha_radius : float
        Parameter for creation of alpha volume that encloses the
        RBFVolume. Smaller values improve the quality of alpha volume,
        but increase the time for sampling points inside the volume.
    optiter : int 
        Number of iterations for optimization that interpolates X,Y,Z coordinates into parameteric U,V,L coordinates.  
    nodeitr : int 
        Number of iterations for distributing sampled points inside the volume.
    Returns
    -------
    (Y1, X1, ... , YN, XN) where N is the number of dimensions of the volume.
    X : array of coordinates
        The sampled coordinates.
    Y : array of distances
        The arc-distance from the starting index of the coordinate space to the corresponding coordinates in X.

    """
    import dlib
    boundary_uvl_coords = np.array([[ip_vol.u[0],ip_vol.v[0],ip_vol.l[0]],
                                    [ip_vol.u[0],ip_vol.v[-1],ip_vol.l[0]],
                                    [ip_vol.u[-1],ip_vol.v[0],ip_vol.l[0]],
                                    [ip_vol.u[-1],ip_vol.v[-1],ip_vol.l[0]],
                                    [ip_vol.u[0],ip_vol.v[0],ip_vol.l[-1]],
                                    [ip_vol.u[0],ip_vol.v[-1],ip_vol.l[-1]],
                                    [ip_vol.u[-1],ip_vol.v[0],ip_vol.l[-1]],
                                    [ip_vol.u[-1],ip_vol.v[-1],ip_vol.l[-1]]])

    resample = 10
    span_U, span_V, span_L  = ip_vol._resample_uvl(resample, resample, resample)
    
    if origin_spec is None:
        origin_coords = np.asarray([np.median(span_U), np.median(span_V), np.max(span_L)])
    else:
        origin_coords = np.asarray([origin_spec['U'](span_U), origin_spec['V'](span_V), origin_spec['L'](span_L)])
        
    logger.info('Origin coordinates: %f %f %f' % (origin_coords[0], origin_coords[1], origin_coords[2]))

    pos, extents = ip_vol.point_position(origin_coords[0],origin_coords[1],origin_coords[2])

    origin_pos = pos[0]
    origin_extent = extents[0]

    logger.info('Origin position: %f %f extent: %f %f' % (origin_pos[0], origin_pos[1], origin_extent[0], origin_extent[1]))
    origin_pos_um = (origin_pos[0] * origin_extent[0], origin_pos[1] * origin_extent[1])
    
    logger.info("Constructing alpha shape...")
    tri = ip_vol.create_triangulation()
    alpha = alpha_shape([], alpha_radius, tri=tri)

    vert = alpha.points
    smp  = np.asarray(alpha.bounds, dtype=np.int64)

    N = nsample*2 # total number of nodes
    node_count = 0
    itr = nodeitr

    while node_count < nsample:
        logger.info("Generating %i nodes (%i iterations)..." % (N, itr))
        # create N quasi-uniformly distributed nodes
        nodes, smpid = menodes(N,vert,smp,itr=itr)
    
        # remove nodes outside of the domain
        in_nodes = nodes[contains(nodes,vert,smp)]
                              
        node_count = len(in_nodes)
        N = int(1.5*N)

    logger.info("%i interior nodes generated (%i iterations)" % (node_count, itr))

    
    logger.info('Inverse interpolation of UVL coordinates...')
    xyz_coords = in_nodes.reshape(-1,3)
    uvl_coords_interp = ip_vol.inverse(xyz_coords)
    xyz_coords_interp = ip_vol(uvl_coords_interp[:,0],uvl_coords_interp[:,1],uvl_coords_interp[:,2], mesh=False).reshape(3,-1).T

    xyz_error_interp  = np.abs(np.subtract(xyz_coords, xyz_coords_interp))

    min_extent = [np.min(ip_vol.u), np.min(ip_vol.v), np.min(ip_vol.l)]
    max_extent = [np.max(ip_vol.u), np.max(ip_vol.v), np.max(ip_vol.l)]
                      

    logger.info('Interpolation by optimization of UVL coordinates...')
    all_node_uvl_coords = []
    for i in xrange(0, xyz_coords.shape[0]):
        logger.info("coordinates %i" % i)
        this_xyz_coords = xyz_coords[i,:]
        f_uvl_distance = make_uvl_distance(this_xyz_coords,rotate=rotate)
        uvl_coords_opt,dist = dlib.find_min_global(f_uvl_distance, min_extent, max_extent, optiter)
        xyz_coords_opt = DG_volume(uvl_coords_opt[0], uvl_coords_opt[1], uvl_coords_opt[2], rotate=rotate)[0]
        xyz_error_opt   = np.abs(np.subtract(xyz_coords[i,:], xyz_coords_opt))
        logger.info('uvl_coords_opt: %s' % str(uvl_coords_opt))
        logger.info('uvl_coords_interp: %s' % str(uvl_coords_interp[i,:]))
        logger.info('xyz_error_opt: %s' % str(xyz_error_opt))
        logger.info('xyz_error_interp: %s' % str(xyz_error_interp[i,:]))
        if np.all (np.less (xyz_error_interp[i,:], xyz_error_opt)):
            coords = uvl_coords_interp[i,:]
        else:
            coords = np.asarray(uvl_coords_opt)
        all_node_uvl_coords.append(coords.ravel())

    node_uvl_coords = np.vstack(all_node_uvl_coords)
    uvl_coords = np.vstack([boundary_uvl_coords, node_uvl_coords])

    logger.info('Computing volume distances...')
    ldists_u = []
    ldists_v = []
    obs_uvls = []
    for uvl in uvl_coords:
        sample_U = uvl[0]
        sample_V = uvl[1]
        sample_L = uvl[2]
        pos, extent = ip_vol.point_position(sample_U, sample_V, sample_L)

        uvl_pos = pos[0]
        uvl_extent = extent[0]
        
        obs_uvls.append(uvl)
        ldists_u.append(uvl_pos[0] * origin_extent[0] - origin_pos_um[0])
        ldists_v.append(uvl_pos[1] * origin_extent[1] - origin_pos_um[1])
        
    distances_u = np.asarray(ldists_u, dtype=np.float32)
    distances_v = np.asarray(ldists_v, dtype=np.float32)

    obs_uv = np.vstack(obs_uvls)

    u_min_ind = np.argmin(distances_u)
    u_max_ind = np.argmax(distances_u)
    v_min_ind = np.argmin(distances_v)
    v_max_ind = np.argmax(distances_v)
    
    logger.info('U distance min: %f %s max: %f %s' % (distances_u[u_min_ind], str(obs_uv[u_min_ind]), distances_u[u_max_ind], str(obs_uv[u_max_ind])))
    logger.info('V distance min: %f %s max: %f %s' % (distances_v[v_min_ind], str(obs_uv[v_min_ind]), distances_v[v_max_ind], str(obs_uv[v_max_ind])))

    return (obs_uv, distances_u, distances_v)


        
def interp_soma_distances(comm, ip_dist_u, ip_dist_v, soma_coords, population_extents, interp_chunk_size=1000, populations=None, allgather=False):
    """Interpolates path lengths of cell coordinates along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    comm : MPIComm
        mpi4py MPI communicator
    ip_dist_u : RBFInterpolant
        Interpolation function for computing arc distances along the first dimension of the volume.
    ip_dist_v : RBFInterpolant
        Interpolation function for computing arc distances along the second dimension of the volume.
    soma_coords : { population_name : coords_dict }
        A dictionary that maps each cell population name to a dictionary of coordinates. The dictionary of coordinates must have the following type:
          coords_dict : { gid : (u, v, l) }
          where:
          - gid: cell identifier
          - u, v, l: floating point coordinates
    population_extents: { population_name : limits }
        A dictionary of maximum and minimum population coordinates in u,v,l space
        Argument limits has the following type:
         ((min_u, min_v, min_l), (max_u, max_v, max_l))
    allgather: boolean (default: False)
       if True, the results are gathered from all ranks and combined
    Returns
    -------
    A dictionary of the form:

      { population: { gid: (distance_U, distance_V } }

    """

    rank = comm.rank
    size = comm.size

    if populations is None:
        populations = soma_coords.keys()

    soma_distances = {}
    for pop in populations:
        coords_dict = soma_coords[pop]
        if rank == 0:
            logger.info('Computing soma distances for population %s...' % pop)
        count = 0
        local_dist_dict = {}
        limits = population_extents[pop]
        u_obs = []
        v_obs = []
        gids    = []
        for gid, coords in coords_dict.iteritems():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                try:
                    assert((limits[1][0] - soma_u + 0.001 >= 0.) and (soma_u - limits[0][0] + 0.001 >= 0.))
                    assert((limits[1][1] - soma_v + 0.001 >= 0.) and (soma_v - limits[0][1] + 0.001 >= 0.))
                    assert((limits[1][2] - soma_l + 0.001 >= 0.) and (soma_l - limits[0][2] + 0.001 >= 0.))
                except Exception as e:
                    logger.error("gid %i: out of limits error for coordinates: %f %f %f limits: %f:%f %f:%f %f:%f )" % \
                                     (gid, soma_u, soma_v, soma_l, limits[0][0], limits[1][0], limits[0][1], limits[1][1], limits[0][2], limits[1][2]))
                    raise e

                u_obs.append(np.array([soma_u,soma_v,soma_l]).ravel())
                v_obs.append(np.array([soma_u,soma_v,soma_l]).ravel())
                gids.append(gid)
        if len(u_obs) > 0:
            u_obs_array = np.vstack(u_obs)
            v_obs_array = np.vstack(v_obs)
            distance_u_obs = ip_dist_u(u_obs_array).reshape(-1,1)
            distance_v_obs = ip_dist_v(v_obs_array).reshape(-1,1)
            distance_u = np.mean(distance_u_obs, axis=1)
            distance_v = np.mean(distance_v_obs, axis=1)
            try:
                assert(np.all(np.isfinite(distance_u)))
                assert(np.all(np.isfinite(distance_v)))
            except Exception as e:
                u_nan_idxs = np.where(np.isnan(distance_u))[0]
                v_nan_idxs = np.where(np.isnan(distance_v))[0]
                logger.error('Invalid distances: u: %s; v: %s', str(u_obs_array[u_nan_idxs]), str(v_obs_array[v_nan_idxs]))
                raise e

            
        for (i,gid) in enumerate(gids):
            local_dist_dict[gid] = (distance_u[i], distance_v[i])
            if rank == 0:
                logger.info('gid %i: distances: %f %f' % (gid, distance_u[i], distance_v[i]))
        if allgather:
            dist_dicts = comm.allgather(local_dist_dict)
            combined_dist_dict = {}
            for dist_dict in dist_dicts:
                for k, v in dist_dict.iteritems():
                    combined_dist_dict[k] = v
            soma_distances[pop] = combined_dist_dict
        else:
            soma_distances[pop] = local_dist_dict

    return soma_distances


def measure_distances(env, comm, soma_coords, resolution=[30, 30, 10], interp_chunk_size=1000, allgather=False):

    rank = comm.rank

    min_u = float('inf')
    max_u = 0.0

    min_v = float('inf')
    max_v = 0.0

    min_l = float('inf')
    max_l = 0.0
    
    for layer, min_extent in env.geometry['Parametric Surface']['Minimum Extent'].iteritems():
        min_u = min(min_extent[0], min_u)
        min_v = min(min_extent[1], min_v)
        min_l = min(min_extent[2], min_l)
        
    for layer, max_extent in env.geometry['Parametric Surface']['Maximum Extent'].iteritems():
        max_u = max(max_extent[0], max_u)
        max_v = max(max_extent[1], max_v)
        max_l = max(max_extent[2], max_l)

    population_extents = {}
    for population in soma_coords.keys():
        min_extent = env.geometry['Cell Layers']['Minimum Extent'][population]
        max_extent = env.geometry['Cell Layers']['Maximum Extent'][population]
        population_extents[population] = (min_extent, max_extent)

    rotate = env.geometry['Parametric Surface']['Rotation']
    origin = env.geometry['Parametric Surface']['Origin']

    obs_uv = None
    coeff_dist_u = None
    coeff_dist_v = None
    
    interp_penalty = 0.01
    interp_basis = 'ga'
    interp_order = 1

    if rank == 0:
        logger.info('Computing soma distances...')

    ## This parameter is used to expand the range of L and avoid
    ## situations where the endpoints of L end up outside of the range
    ## of the distance interpolant
    safety = 0.01

    if rank == 0:
        logger.info('Creating volume: min_l = %f max_l = %f...' % (min_l, max_l))
        ip_volume = make_volume((min_u-safety, max_u+safety), \
                                (min_v-safety, max_v+safety), \
                                (min_l-safety, max_l+safety), \
                                resolution=resolution, rotate=rotate)

        logger.info('Computing volume distances...')
        vol_dist = get_volume_distances(ip_volume, origin_spec=origin)
        (obs_uv, dist_u, dist_v) = vol_dist
        logger.info('Computing U volume distance interpolants...')
        ip_dist_u = RBFInterpolant(obs_uv,dist_u,order=interp_order,basis=interp_basis,\
                                   penalty=interp_penalty)
        coeff_dist_u = ip_dist_u._coeff
        logger.info('Computing V volume distance interpolants...')
        ip_dist_v = RBFInterpolant(obs_uv,dist_v,order=interp_order,basis=interp_basis,\
                                   penalty=interp_penalty)
        coeff_dist_v = ip_dist_v._coeff
        logger.info('Broadcasting volume distance interpolants...')
        
    obs_uv = comm.bcast(obs_uv, root=0)
    coeff_dist_u = comm.bcast(coeff_dist_u, root=0)
    coeff_dist_v = comm.bcast(coeff_dist_v, root=0)
    
    ip_dist_u = RBFInterpolant(obs_uv,coeff=coeff_dist_u,order=interp_order,basis=interp_basis,\
                               penalty=interp_penalty)
    ip_dist_v = RBFInterpolant(obs_uv,coeff=coeff_dist_v,order=interp_order,basis=interp_basis,\
                               penalty=interp_penalty)

    
    soma_distances = interp_soma_distances(comm, ip_dist_u, ip_dist_v, soma_coords, population_extents, \
                                           interp_chunk_size=interp_chunk_size, allgather=allgather)

    return soma_distances



def icp_transform(comm, soma_coords, projection_ls, population_extents, rotate=None, populations=None, icp_iter=1000, opt_iter=100):
    """
    Uses the iterative closest point (ICP) algorithm of the PCL library to transform soma coordinates onto a surface for a particular L value.
    http://pointclouds.org/documentation/tutorials/iterative_closest_point.php#iterative-closest-point

    """
    
    import dlib, pcl
    
    rank = comm.rank
    size = comm.size

    if populations is None:
        populations = soma_coords.keys()

    srf_resample = 25
    
    projection_ptclouds = []
    for obs_l in projection_ls:
        srf = make_surface (obs_l, rotate=rotate)
        U, V = srf._resample_uv(srf_resample, srf_resample)
        meshpts = self.ev(U, V)
        projection_ptcloud = pcl.PointCloud()
        projection_ptcloud.from_array(meshpts)
        projection_ptclouds.append(projection_ptcloud)
        
    soma_coords_dict = {}
    for pop in populations:
        coords_dict = soma_coords[pop]
        if rank == 0:
            logger.info('Computing point transformation for population %s...' % pop)
        count = 0
        limits = population_extents[pop]
        xyz_coords = []
        gids = []
        for gid, coords in coords_dict.iteritems():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                xyz_coords.append(DG_volume(soma_u, soma_v, soma_l, rotate=rotate))
                gids.append(gid)
        xyz_pts = np.vstack(xyz_coords)
        
        cloud_in = pcl.PointCloud()
        cloud_in.from_array(xyz_pts)

        icp = cloud_in.make_IterativeClosestPoint()

        all_est_xyz_coords = []
        all_est_uvl_coords = []
        all_interp_err = []
        
        for (k,cloud_prj) in enumerate(projection_ls):
            k_est_xyz_coords = np.zeros((len(gids),3))
            k_est_uvl_coords = np.zeros((len(gids),3))
            interp_err = np.zeros((len(gids),))
            converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_prj, max_iter=icp_iter)
            logger.info('Transformation of population %s has converged: ' % (pop) + str(converged) + ' score: %f' % (fitness) )
            for i, gid in itertools.izip(xrange(0, estimate.size), gids):
                k_xyz_coords = estimate[i]
                k_est_xyz_coords[i,:] = est_xyz_coords
                f_uvl_distance = make_uvl_distance(est_xyz_coords,rotate=rotate)
                uvl_coords,err = dlib.find_min_global(f_uvl_distance, limits[0], limits[1], opt_iter)
                k_est_uvl_coords[i,:] = uvl_coords
                interp_err[i,] = err
                if rank == 0:
                    logger.info('gid %i: u: %f v: %f l: %f' % (gid, uvl_coords[0], uvl_coords[1], uvl_coords[2]))
            all_est_xyz_coords.append(k_est_xyz_coords)
            all_est_uvl_coords.append(k_est_uvl_coords)
            all_interp_err.append(interp_err)

        coords_dict = {}
        for (i, gid) in enumerate(gids):
            coords_dict[gid] = { 'X Coordinate': np.asarray([ col[i,0] for col in all_est_xyz_coords ], dtype='float32'),
                                 'Y Coordinate': np.asarray([ col[i,1] for col in all_est_xyz_coords ], dtype='float32'),
                                 'Z Coordinate': np.asarray([ col[i,2] for col in all_est_xyz_coords ], dtype='float32'),
                                 'U Coordinate': np.asarray([ col[i,0] for col in all_est_uvl_coords ], dtype='float32'),
                                 'V Coordinate': np.asarray([ col[i,1] for col in all_est_uvl_coords ], dtype='float32'),
                                 'L Coordinate': np.asarray([ col[i,2] for col in all_est_uvl_coords ], dtype='float32'),
                                 'Interpolation Error': np.asarray( [err[i] for err in all_interp_err ], dtype='float32') }

        soma_coords_dict[pop] = coords_dict
                                  
    return soma_coords_dict
