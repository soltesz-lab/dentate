from function_lib import *
from mpi4py import MPI
from neurotrees.io import append_cell_attributes
from neurotrees.io import NeurotreeAttrGen
from neurotrees.io import population_ranges
import scipy.optimize as optimize

coords_dir = '../morphologies/'
coords_file = 'dentate_Sampled_Soma_Locations_test.h5'

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

if rank == 0:
    print '%i ranks have been allocated' % comm.size
sys.stdout.flush()

# U, V, L
p0 = [1.555, 1.876, -1.]  # center of the volume
pmin = [-0.064, -0.810, -4.1]
pmax = [3.187, 4.564, 3.1]

populations = population_ranges(MPI._addressof(comm), coords_dir+coords_file).keys()


def rotate3d(vec, rot_deg):
    """
    
    :param vec: array 
    :param rot: array
    :return: array
    """
    xrad, yrad, zrad = [rot_rad*2.*np.pi/360. for rot_rad in rot_deg]
    Mx = np.array([[1, 0, 0], [0, np.cos(xrad), np.sin(xrad)], [0, -np.sin(xrad), np.cos(xrad)]])
    My = np.array([[np.cos(yrad), 0, -np.sin(yrad)], [0, 1, 0], [np.sin(yrad), 0, np.cos(yrad)]])
    Mz = np.array([[np.cos(zrad), np.sin(zrad), 0], [-np.sin(zrad), np.cos(zrad), 0], [0, 0, 1]])
    new_vec = np.array(vec)
    new_vec = new_vec.dot(Mx).dot(My).dot(Mz)

    return new_vec


def interp_points(params):
    """
    
    :param params: 
    :return: array 
    """
    u, v, l = params
    xi = -500. * np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v))
    yi = 750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114 * l) * np.cos(v))
    zi = 2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi - u))

    params_rot = rotate3d([xi, yi, zi], [-35., 0., 0.])

    return params_rot


def euc_distance(params, args):
    """
    
    :param params: array 
    :param args: array
    :return: float 
    """
    #if not check_bounds(params):
    #    return 1e9
    params_rot = interp_points(params)

    error = 0.
    for d, di in zip(args, params_rot):
        error += ((d - di)/0.1) ** 2.
    error = np.sqrt(error)

    return error


for population in populations:
#for population in ['MC']:
    start_time = time.time()
    count = 0
    coords_gen = NeurotreeAttrGen(MPI._addressof(comm), coords_dir+coords_file, population, io_size=comm.size,
                                  cache_size=50, namespace='Coordinates')
    for gid, orig_coords_dict in coords_gen:
    # gid, orig_coords_dict = coords_gen.next()
        coords_dict = {}
        if gid is not None:
            orig_coords_dict = orig_coords_dict['Coordinates']

            coords_dict[gid] = {parameter: np.array([], dtype='float32') for parameter in
                                ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 'U Coordinate', 'V Coordinate',
                                 'L Coordinate', 'Interpolation Error']}
            x = orig_coords_dict['X Coordinate'][0]
            y = orig_coords_dict['Y Coordinate'][0]
            z = orig_coords_dict['Z Coordinate'][0]
            result = optimize.minimize(euc_distance, p0, method='Powell', args=([x, y, z],), options={'disp': False})
            formatted_x = '[' + ', '.join(['%.4E' % xi for xi in [x, y, z]]) + ']'
            interp_coords = interp_points(result.x)
            formatted_xi = '[' + ', '.join(['%.4E' % xi for xi in interp_coords]) + ']'
            coords_dict[gid]['X Coordinate'] = np.append(coords_dict[gid]['X Coordinate'],
                                                         interp_coords[0]).astype('float32', copy=False)
            coords_dict[gid]['Y Coordinate'] = np.append(coords_dict[gid]['Y Coordinate'],
                                                         interp_coords[1]).astype('float32', copy=False)
            coords_dict[gid]['Z Coordinate'] = np.append(coords_dict[gid]['Z Coordinate'],
                                                         interp_coords[2]).astype('float32', copy=False)
            coords_dict[gid]['U Coordinate'] = np.append(coords_dict[gid]['U Coordinate'],
                                                         result.x[0]).astype('float32', copy=False)
            coords_dict[gid]['V Coordinate'] = np.append(coords_dict[gid]['V Coordinate'],
                                                         result.x[1]).astype('float32', copy=False)
            coords_dict[gid]['L Coordinate'] = np.append(coords_dict[gid]['L Coordinate'],
                                                         result.x[2]).astype('float32', copy=False)
            coords_dict[gid]['Interpolation Error'] = np.append(coords_dict[gid]['Interpolation Error'],
                                                                result.fun).astype('float32', copy=False)
            print 'Rank %i: %s gid: %i, target: %s, result: %s, error: %.4E' % (rank, population, gid, formatted_x,
                                                                                formatted_xi, result.fun)
            count += 1
        sys.stdout.flush()
        append_cell_attributes(MPI._addressof(comm), coords_dir+coords_file, population, coords_dict,
                               namespace='Interpolated Coordinates', io_size=comm.size, chunk_size=100000,
                               value_chunk_size=2000000)
        del coords_dict
        gc.collect()
    global_count = comm.gather(count, root=0)
    if rank == 0:
        print 'Interpolation of %i %s cells took %i s' % (np.sum(global_count), population, time.time()-start_time)