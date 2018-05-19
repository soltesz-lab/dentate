
##
## Classes and procedures related to neuronal geometry and distance calculation.
##

import sys, time, gc, itertools
from collections import defaultdict
import numpy as np
import dlib
import rbf, rbf.basis
from rbf.interpolate import RBFInterpolant
from rbf.nodes import snap_to_boundary,disperse,menodes
from rbf.geometry import contains
from dentate.alphavol import alpha_shape
from dentate.rbf_volume import RBFVolume, rotate3d

from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, bcast_cell_attributes, read_population_ranges, append_graph
import logging

logger = logging.getLogger(__name__)


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

def make_volume(lmin, lmax, basis=rbf.basis.phs3, rotate=None, ures=33, vres=30, lres=10):  
    
#    obs_u = np.linspace(-0.016*np.pi, 1.01*np.pi, ures)
#    obs_v = np.linspace(-0.23*np.pi, 1.425*np.pi, vres)
    obs_u = np.linspace(-0.02*np.pi, 1.01*np.pi, ures)
    obs_v = np.linspace(-0.26*np.pi, 1.455*np.pi, vres)
    obs_l = np.linspace(lmin, lmax, num=lres)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = DG_volume (u, v, l, rotate=rotate)

    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, basis=basis, order=2)

    return vol


def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a-b)**2,axis=1))


def make_uvl_distance(xyz_coords,rotate=None):
      f = lambda u, v, l: euclidean_distance(DG_volume(u,v,l,rotate=rotate), xyz_coords)
      return f

def make_u_distance(xyz_coords,v,l,rotate=None):
      f = lambda u: euclidean_distance(DG_volume(u,v,l,rotate=rotate), xyz_coords)
      return f

def make_v_distance(xyz_coords,u,l,rotate=None):
      f = lambda v: euclidean_distance(DG_volume(u,v,l,rotate=rotate), xyz_coords)
      return f


def get_volume_distances (ip_vol, res=2, step=1, interp_chunk_size=1000, verbose=False):
    """Computes arc-distances along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    ip_vol : RBFVolume
        An interpolated volume instance of class RBFVolume.
    res : int
        Resampling factor for the U, V, L coordinates of the volume. 
        This parameter will be used to resample the volume and reduce the error of the arc distance calculation.
    step : int (default=1)
        Used to subsample the arrays of computed distances.
    Returns
    -------
    (Y1, X1, ... , YN, XN) where N is the number of dimensions of the volume.
    X : array of coordinates
        The sampled coordinates.
    Y : array of distances
        The arc-distance from the starting index of the coordinate space to the corresponding coordinates in X.
    """

    if verbose:
        logger.setLevel(logging.INFO)

    logger.info('Resampling volume...')
    U, V, L = ip_vol._resample_uvl(res, res, res)

    axis_origins = [np.median(U), np.median(V), np.max(L)]
    logger.info('Axis origins: %f %f %f' % (tuple(axis_origins)))
    
    logger.info('Computing U distances...')
    ldist_u, obs_dist_u = ip_vol.point_distance(U, V, L, axis=0, axis_origin=axis_origins[0], interp_chunk_size=interp_chunk_size)
    obs_uvl = np.array([np.concatenate(obs_dist_u[0]), \
                        np.concatenate(obs_dist_u[1]), \
                        np.concatenate(obs_dist_u[2])]).T
    sample_inds = np.arange(0, obs_uvl.shape[0], step)
    obs_u = obs_uvl[sample_inds,:]
    distances_u = np.concatenate(ldist_u)[sample_inds]

    logger.info('U coord min: %f max: %f' % (np.min(U), np.max(U)))
    logger.info('U distance min: %f max: %f' % (np.min(distances_u), np.max(distances_u)))
    
    logger.info('Computing V distances...')
    ldist_v, obs_dist_v = ip_vol.point_distance(U, V, L, axis=1, axis_origin=axis_origins[1], interp_chunk_size=interp_chunk_size)
    obs_uvl = np.array([np.concatenate(obs_dist_v[0]), \
                        np.concatenate(obs_dist_v[1]), \
                        np.concatenate(obs_dist_v[2])]).T
    sample_inds = np.arange(0, obs_uvl.shape[0], step)
    obs_v = obs_uvl[sample_inds,:]
    distances_v = np.concatenate(ldist_v)[sample_inds]

    logger.info('V coord min: %f max: %f' % (np.min(V), np.max(V)))
    logger.info('V distance min: %f max: %f' % (np.min(distances_v), np.max(distances_v)))
        
    return (distances_u, obs_u, distances_v, obs_v)


        
def get_soma_distances(comm, dist_u, dist_v, soma_coords, population_extents, interp_chunk_size=1000, populations=None, allgather=False, verbose=False):
    """Computes arc-distances of cell coordinates along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    comm : MPIComm
        mpi4py MPI communicator
    dist_u : RBFInterpolant
        Interpolation function for computing arc distances along the first dimension of the volume.
    dist_v : RBFInterpolant
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

    if verbose:
        logger.setLevel(logging.INFO)

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
                gids.append(gid)
        if len(uvl_obs) > 0:
            uvl_obs_array = np.vstack(uvl_obs)
            k = uvl_obs_array.shape[0]
            distance_u = dist_u(uvl_obs_array, chunk_size=interp_chunk_size)
            distance_v = dist_v(uvl_obs_array, chunk_size=interp_chunk_size)
            assert(np.all(np.isfinite(distance_u)))
            assert(np.all(np.isfinite(distance_v)))
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


def project_points(comm, soma_coords, population_extents, projection_ls, rotate=None, populations=None, allgather=False, verbose=False, optiter=150):

    rank = comm.rank
    size = comm.size

    if verbose:
        logger.setLevel(logging.INFO)

    if populations is None:
        populations = soma_coords.keys()
        
    soma_u_projections = {}
    soma_v_projections = {}
    for pop in populations:
        coords_dict = soma_coords[pop]
        if rank == 0:
            logger.info('Computing point projections for population %s...' % pop)
        count = 0
        limits = population_extents[pop]
        prj_u_dict = {}
        prj_v_dict = {}
        for gid, coords in coords_dict.iteritems():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                xyz_coords = DG_volume(soma_u, soma_v, soma_l, rotate=rotate)
                prj_u_coords = []
                prj_v_coords = []
                
                for prj_l in projection_ls:
                    f_u_distance = make_u_distance(xyz_coords,soma_v,prj_l,rotate=rotate)
                    f_v_distance = make_v_distance(xyz_coords,soma_u,prj_l,rotate=rotate)
                    u_coord,u_dist = \
                      dlib.find_min_global(f_u_distance, [limits[0][0]], [limits[1][0]], optiter)
                    v_coord,v_dist = \
                      dlib.find_min_global(f_v_distance, [limits[0][1]], [limits[1][1]], optiter)
                    prj_u_coords.append(u_coord[0])
                    prj_v_coords.append(v_coord[0])
                    if rank == 0:
                        logger.info('gid %i: u: %f v: %f l: %f prj l: %f prj u: %f prj v: %f' % (gid, soma_u, soma_v, soma_l, prj_l, u_coord[0], v_coord[0]))


                prj_u_dict[gid] = np.asarray(prj_u_coords, dtype=np.float32)
                prj_v_dict[gid] = np.asarray(prj_v_coords, dtype=np.float32)

        if allgather:
            prj_u_dicts = comm.allgather(prj_u_dict)
            prj_v_dicts = comm.allgather(prj_v_dict)
            combined_prj_u_dict = {}
            combined_prj_v_dict = {}
            for prj_u_dict in prj_u_dicts:
                for k, v in prj_u_dict.iteritems():
                    combined_prj_u_dict[k] = v
            for prj_v_dict in prj_v_dicts:
                for k, v in prj_v_dict.iteritems():
                    combined_prj_v_dict[k] = v
            soma_u_projections[pop] = combined_prj_u_dict
            soma_v_projections[pop] = combined_prj_v_dict
        else:
            soma_u_projections[pop] = prj_u_dict
            soma_v_projections[pop] = prj_v_dict

    return (soma_u_projections, soma_v_projections)
