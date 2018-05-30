
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
from dentate.rbf_surface import RBFSurface, rotate3d
from dentate import utils
from neuroh5.io import NeuroH5CellAttrGen, bcast_cell_attributes, read_population_ranges, append_graph

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = utils.get_module_logger(__name__)

max_u = 11690.
max_v = 2956.


def DG_volume(u, v, l, rotate=None):
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

def make_volume(lmin, lmax, rotate=None, basis=rbf.basis.phs3, ures=33, vres=30, lres=10):  
    """Creates an RBF volume based on the parametric equations of the dentate volume.
    """
#    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, ures)
#    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, vres)
    obs_u = np.linspace(-0.02*np.pi, 1.01*np.pi, ures)
    obs_v = np.linspace(-0.26*np.pi, 1.455*np.pi, vres)
    obs_l = np.linspace(lmin, lmax, num=lres)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = DG_volume (u, v, l, rotate=rotate)

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, basis=basis, order=2)

    return vol


def make_surface(obs_l, rotate=None, basis=rbf.basis.phs3, ures=33, vres=30, lres=10):  
    """Creates an RBF surface based on the parametric equations of the dentate volume.
    """
#    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, ures)
#    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, vres)
    obs_u = np.linspace(-0.02*np.pi, 1.01*np.pi, ures)
    obs_v = np.linspace(-0.26*np.pi, 1.455*np.pi, vres)

    u, v = np.meshgrid(obs_u, obs_v, indexing='ij')
    xyz = DG_volume (u, v, obs_l, rotate=rotate)

    srf = RBFSurface(obs_u, obs_v, xyz, basis=basis, order=2)

    return srf


def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a-b)**2,axis=1))


def make_uvl_distance(xyz_coords,rotate=None):
      f = lambda u, v, l: euclidean_distance(DG_volume(u,v,l,rotate=rotate), xyz_coords)
      return f


def get_volume_distances (ip_vol, nsample=250, res=3, alpha_radius=120., interp_chunk_size=1000):
    """Computes arc-distances along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    ip_vol : RBFVolume
        An interpolated volume instance of class RBFVolume.
    nsample : int
        Number of points to sample inside the volume.
    Returns
    -------
    (Y1, X1, ... , YN, XN) where N is the number of dimensions of the volume.
    X : array of coordinates
        The sampled coordinates.
    Y : array of distances
        The arc-distance from the starting index of the coordinate space to the corresponding coordinates in X.
    """

    span_U, span_V, span_L  = ip_vol._resample_uvl(res, res, res)

    origin_coords = np.asarray([np.median(span_U), np.median(span_V), np.max(span_L)])
    logger.info('Origin coordinates: %f %f %f' % (origin_coords[0], origin_coords[1], origin_coords[2]))

    logger.info("Constructing alpha shape...")
    tri = ip_vol.create_triangulation()
    alpha = alpha_shape([], alpha_radius, tri=tri)

    vert = alpha.points
    smp  = np.asarray(alpha.bounds, dtype=np.int64)

    N = nsample*2 # total number of nodes
    node_count = 0
    itr = 1

    logger.info("Generating %i nodes..." % N)
    while node_count < nsample:
        # create N quasi-uniformly distributed nodes
        nodes, smpid = menodes(N,vert,smp,itr=itr)
    
        # remove nodes outside of the domain
        in_nodes = nodes[contains(nodes,vert,smp)]
                              
        node_count = len(in_nodes)
        itr = int(itr / 2)

    xyz_coords = in_nodes.reshape(-1,3)
    uvl_coords_interp = ip_vol.inverse(xyz_coords)

    logger.info("%i nodes generated" % node_count)
        
    logger.info('Computing distances...')
    ldists_u = []
    ldists_v = []
    obss_u = []
    obss_v = []
    for uvl in uvl_coords_interp:
        sample_U = uvl[0]
        sample_V = uvl[1]
        sample_L = uvl[2]
        ldist_u, obs_dist_u = ip_vol.point_distance(span_U, sample_V, sample_L, axis=0, \
                                                    origin_coords=origin_coords, \
                                                    interp_chunk_size=interp_chunk_size)
        ldist_v, obs_dist_v = ip_vol.point_distance(sample_U, span_V, sample_L, axis=1, \
                                                    origin_coords=origin_coords, \
                                                    interp_chunk_size=interp_chunk_size)
        obs_u = np.array([np.concatenate(obs_dist_u[0]), \
                          np.concatenate(obs_dist_u[1]), \
                          np.concatenate(obs_dist_u[2])]).T
        obs_v = np.array([np.concatenate(obs_dist_v[0]), \
                          np.concatenate(obs_dist_v[1]), \
                          np.concatenate(obs_dist_v[2])]).T
        ldists_u.append(ldist_u)
        ldists_v.append(ldist_v)
        obss_u.append(obs_u)
        obss_v.append(obs_v)
        
    distances_u = np.concatenate(ldists_u).reshape(-1)
    obs_u = np.concatenate(obss_u)
    distances_v = np.concatenate(ldists_v).reshape(-1)
    obs_v = np.concatenate(obss_v)
    
    logger.info('U distance min: %f max: %f' % (np.min(distances_u), np.max(distances_u)))
    logger.info('V distance min: %f max: %f' % (np.min(distances_v), np.max(distances_v)))

    return (distances_u, obs_u, distances_v, obs_v)


        
def interp_soma_distances(comm, ip_dist_u, ip_dist_v, soma_coords, population_extents, interp_chunk_size=1000, populations=None, allgather=False):
    """Interpolates arc-distances of cell coordinates along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    comm : MPIComm
        mpi4py MPI communicator
    ip_dist_u : RBFInterpolant
        Interpolation function for computing arc distances along the first dimension of the volume.
    ip_dist_v : RBFInterpolant
        Interpolation function for computing arc distances along the second dimension of the volume.
    ip_volume: optional instance of RBFVolume
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
        uvl_obs = []
        gids    = []
        for gid, coords in coords_dict.iteritems():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                uvl_obs.append(np.array([soma_u,soma_v,soma_l]).reshape(1,3))
                try:
                    assert((limits[1][0] - soma_u + 0.001 >= 0.) and (soma_u - limits[0][0] + 0.001 >= 0.))
                    assert((limits[1][1] - soma_v + 0.001 >= 0.) and (soma_v - limits[0][1] + 0.001 >= 0.))
                    assert((limits[1][2] - soma_l + 0.001 >= 0.) and (soma_l - limits[0][2] + 0.001 >= 0.))
                except Exception as e:
                    logger.error("gid %i: out of limits error for coordinates: %f %f %f limits: %f:%f %f:%f %f:%f )" % \
                                     (gid, soma_u, soma_v, soma_l, limits[0][0], limits[1][0], limits[0][1], limits[1][1], limits[0][2], limits[1][2]))
                    raise e
                uvl_obs.append(np.array([soma_u,soma_v,limits[1][2]]).reshape(1,3))
                gids.append(gid)
        if len(uvl_obs) > 0:
            uvl_obs_array = np.vstack(uvl_obs)
            distance_u = ip_dist_u(uvl_obs_array)
            distance_v = ip_dist_v(uvl_obs_array)
            try:
                assert(np.all(np.isfinite(distance_u)))
                assert(np.all(np.isfinite(distance_v)))
            except Exception as e:
                print 'distance_u: ', distance_u
                print 'distance_v: ', distance_v
                raise e
                
        for (i,gid) in enumerate(gids):
            local_dist_dict[gid] = (distance_u[i], distance_v[i])
            if rank == 0:
                soma_u = uvl_obs_array[i,0]
                soma_v = uvl_obs_array[i,1]
                soma_l = uvl_obs_array[i,2]
                logger.info('gid %i: coordinates: %f %f %f distances: %f %f' % (gid, soma_u, soma_v, soma_l, distance_u[i], distance_v[i]))
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



def get_soma_distances(comm, ip_vol, soma_coords, population_extents, res=3, interp_chunk_size=1000, populations=None, allgather=False):
    """Computes arc-distances of cell coordinates along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    comm : MPIComm
        mpi4py MPI communicator
    ip_vol: instance of RBFVolume
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

    span_U, span_V, span_L  = ip_vol._resample_uvl(res, res, res)
    origin_coords = np.asarray([np.median(span_U), np.median(span_V), np.max(span_L)])

    soma_distances = {}
    for pop in populations:
        coords_dict = soma_coords[pop]
        if rank == 0:
            logger.info('Computing soma distances for population %s...' % pop)
        count = 0
        local_dist_dict = {}
        limits = population_extents[pop]
        uvl_obs = []
        gids    = []
        for gid, coords in coords_dict.iteritems():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                uvl_obs.append(np.array([soma_u,soma_v,soma_l]).reshape(1,3))
                try:
                    assert((limits[1][0] - soma_u + 0.001 >= 0.) and (soma_u - limits[0][0] + 0.001 >= 0.))
                    assert((limits[1][1] - soma_v + 0.001 >= 0.) and (soma_v - limits[0][1] + 0.001 >= 0.))
                    assert((limits[1][2] - soma_l + 0.001 >= 0.) and (soma_l - limits[0][2] + 0.001 >= 0.))
                except Exception as e:
                    logger.error("gid %i: out of limits error for coordinates: %f %f %f limits: %f:%f %f:%f %f:%f )" % \
                                     (gid, soma_u, soma_v, soma_l, limits[0][0], limits[1][0], limits[0][1], limits[1][1], limits[0][2], limits[1][2]))
                    raise e
                uvl_obs.append(np.array([soma_u,soma_v,limits[1][2]]).reshape(1,3))
                gids.append(gid)

        distance_u = []
        distance_v = []
        if len(uvl_obs) > 0:
            for (gid, uvl) in itertools.izip(gids, uvl_obs):
                soma_u = uvl[0]
                soma_v = uvl[1]
                soma_l = uvl[2]

                origin_u = origin_coords[0]
                origin_v = origin_coords[1]
                origin_l = origin_coords[2]
                
                usteps = round(abs(soma_u - origin_u) / 0.01)
                vsteps = round(abs(soma_v - origin_v) / 0.01)
                uu = np.linspace(origin_u, soma_u, usteps)
                uv = np.linspace(origin_v, soma_v, 3)
                vv = np.linspace(origin_v, soma_v, vsteps)
                vu = np.linspace(origin_u, soma_v, 3)
                l = origin_coords[2]
                cdistance_u, coords_u = ip_vol.point_distance(uu, uv, l, axis=0, chunk_size=interp_chunk_size)
                cdistance_v, coords_v = ip_vol.point_distance(vu, vv, l, axis=1, chunk_size=interp_chunk_size)

                print cdistance_u
                print cdistance_v
                
                try:
                    assert(np.all(np.isfinite(cdistance_u)))
                    assert(np.all(np.isfinite(cdistance_v)))
                except Exception as e:
                    logger.error('Invalid distances: distance_u: %f; distance_v: %f', distance_u, distance_v)
                    raise e
                distance_u.append(np.mean(cdistance_u[np.where(coords_u[:,0] == uvl[0])]))
                distance_v.append(np.mean(cdistance_v[np.where(coords_v[:,1] == uvl[1])]))
                local_dist_dict[gid] = (distance_u, distance_v)
                
                if rank == 0:
                    logger.info('gid %i: coordinates: %f %f %f distances: %f %f' % \
                                    (gid, soma_u, soma_v, soma_l, distance_u, distance_v))
                    
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
