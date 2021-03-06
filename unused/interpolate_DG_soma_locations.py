from function_lib import *
from mpi4py import MPI
from neuroh5.io import append_cell_attributes
from neuroh5.io import NeurotreeAttrGen
from neuroh5.io import population_ranges
import scipy.optimize as optimize
import random
import click  # CLI argument processing


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


class CheckBounds(object):
    """

    """

    def __init__(self, xmin, xmax):
        """

        :param xmin: array of float
        :param xmax: array of float
        """
        self.xmin = xmin
        self.xmax = xmax

    def within_bounds(self, x):
        """
        For optimize.minimize, check that the current set of parameters are within the bounds.
        :param x: array
        :param param_name: str
        :return: bool
        """
        for i in range(len(x)):
            if ((self.xmin[i] is not None and x[i] < self.xmin[i]) or
                    (self.xmax[i] is not None and x[i] > self.xmax[i])):
                return False
        return True

    def return_to_bounds(self, x):
        """
        If a parameter is out of bounds, choose a random value within the bounds
        :param x: array
        :return: array
        """
        new_x = list(x)
        for i in range(len(new_x)):
            if self.xmin[i] is not None and self.xmax[i] is not None:
                if new_x[i] < self.xmin[i] or new_x[i] > self.xmax[i]:
                    new_x[i] = random.uniform(self.xmin[i], self.xmax[i])
        return new_x

    def point_in_bounds(self):
        """
        Choose a random value within the bounds
        :param x: array
        :return: array
        """
        new_x = list(self.xmin)
        for i in range(len(new_x)):
            new_x[i] = random.uniform(self.xmin[i], self.xmax[i])
        return new_x

    def fix_periodic_coords(self, x):
        """
        Shifts periodic u and v back into range. 
        :param x: array
        :return: array
        """
        u = x[0]
        v = x[1]
        u_f = u % (2.*np.pi)
        v_f = v % (2.*np.pi)
        if u_f > self.xmax[0]:
            u_f -= 2. * np.pi
        if v_f > self.xmax[1]:
            v_f -= 2. * np.pi
        return [u_f, v_f, x[2]]


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
    params_rot = interp_points(params)

    error = 0.
    for d, di in zip(args, params_rot):
        error += ((d - di)/0.1) ** 2.

    return error


"""
HIL -3.95: -1.95
GCL: -1.95: 0.
IML: 0.: 1.
MML: 1.: 2.
OML: 2.: 3.1

NGFC: MML, OML
ACC: HIL, GCL, IML
BC: HIL, GCL, IML
MOPP: IML, MML, OML
HCC: HIL, GCL
HC: HIL
IS: HIL
"""

pmin = {'GC': [0.03142, -0.7225, -1.95], 'MEC': [-0.0502, -0.7225, 1.], 'LEC': [-0.0502, -0.7225, 2.],
        'MC': [0.03142, -0.7225, -3.95], 'NGFC': [-0.0502, -0.7225, 1.], 'AAC': [-0.0502, -0.7225, -3.95],
        'BC': [-0.0502, -0.7225, -3.95], 'MOPP': [-0.0502, -0.7225, 0.], 'HCC': [0.03142, -0.7225, -3.95],
        'HC': [0.03142, -0.7225, -3.95], 'IS': [0.03142, -0.7225, -3.95]}
pmax = {'GC': [3.0787, 4.4767, 0.], 'MEC': [3.1730, 4.4767, 2.], 'LEC': [3.1730, 4.4767, 3.1],
        'MC': [3.0787, 4.4767, -1.95], 'NGFC': [3.1730, 4.4767, 3.1], 'AAC': [3.1730, 4.4767, 1.],
        'BC': [3.1730, 4.4767, 1.], 'MOPP': [3.1730, 4.4767, 3.1], 'HCC': [3.0787, 4.4767, 0.],
        'HC': [3.0787, 4.4767, -1.95], 'IS': [3.0787, 4.4767, -1.95]}


@click.command()
@click.option("--log-dir", default=None, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
def main(log_dir, coords_path, io_size, chunk_size, value_chunk_size, cache_size):
    """
    
    :param log_dir: 
    :param coords_path: 
    :param io_size: 
    :param chunk_size: 
    :param value_chunk_size: 
    """
    if log_dir is not None:
        log_filename = str(time.strftime('%m%d%Y', time.localtime()))+'_'+\
                       str(time.strftime('%H%M%S', time.localtime()))+'_interpolate_DG_soma_locations.o'
        sys.stdout = Logger(log_dir+'/'+log_filename)

    comm = MPI.COMM_WORLD
    rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

    if io_size==-1:
        io_size = comm.size

    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    populations = population_ranges(MPI._addressof(comm), coords_path).keys()

    for population in populations:
        print 'population: ', population
    # for population in ['MC']:
        # U, V, L
        this_pmin = pmin[population]
        this_pmax = pmax[population]
        check_bounds = CheckBounds(this_pmin, this_pmax)
        # center of the subvolume
        p0 = [this_pmin[i] + (this_pmax[i] - this_pmin[i]) / 2. for i in range(len(this_pmin))]

        start_time = time.time()
        count = 0
        coords_gen = NeurotreeAttrGen(MPI._addressof(comm), coords_path, population, io_size=io_size,
                                      cache_size=cache_size, namespace='Sampled Coordinates')
        for gid, orig_coords_dict in coords_gen:
        # gid, orig_coords_dict = coords_gen.next()
            coords_dict = {}
            if gid is not None:
                orig_coords_dict = orig_coords_dict['Sampled Coordinates']

                coords_dict[gid] = {parameter: np.array([], dtype='float32') for parameter in
                                    ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 'U Coordinate', 'V Coordinate',
                                     'L Coordinate', 'Interpolation Error']}
                x = orig_coords_dict['X Coordinate'][0]
                y = orig_coords_dict['Y Coordinate'][0]
                z = orig_coords_dict['Z Coordinate'][0]
                this_p0 = p0
                formatted_x = '[' + ', '.join(['%.4E' % xi for xi in [x, y, z]]) + ']'
                min_error = 1e9
                best_coords = None
                for i in range(5):
                    formatted_p0 = '[' + ', '.join(['%.4E' % pi for pi in this_p0]) + ']'
                    print 'Rank %i: %s gid: %i, initial params: %s' % (rank, population, gid, formatted_p0)
                    result = optimize.minimize(euc_distance, this_p0, method='Powell', args=([x, y, z],),
                                               options={'disp': False})
                    # result = optimize.minimize(euc_distance, this_p0, method='SLSQP', bounds=zip(pmin, pmax),
                    #                           args=([x, y, z],), options={'disp': False})
                    fixed_coords = check_bounds.fix_periodic_coords(result.x)
                    error = euc_distance(fixed_coords, [x, y, z])
                    if error < 1. and check_bounds.within_bounds(fixed_coords):
                        best_coords = fixed_coords
                        min_error = error
                        break
                    else:
                        fixed_coords = check_bounds.return_to_bounds(fixed_coords)
                        error = euc_distance(fixed_coords, [x, y, z])
                        interp_coords = interp_points(fixed_coords)
                        formatted_xi = '[' + ', '.join(['%.4E' % xi for xi in interp_coords]) + ']'
                        formatted_pi = '[' + ', '.join(['%.4E' % pi for pi in fixed_coords]) + ']'
                        print 'Rank %i: %s gid: %i, target: %s, result: %s, params: %s, error: %.4E, iteration: %i' % \
                              (rank, population, gid, formatted_x, formatted_xi, formatted_pi, error, i)
                        if error < min_error:
                            min_error = error
                            best_coords = fixed_coords
                        else:
                            if best_coords is None:
                                best_coords = fixed_coords
                                min_error = error
                        this_p0 = check_bounds.point_in_bounds()
                interp_coords = interp_points(best_coords)
                formatted_xi = '[' + ', '.join(['%.4E' % xi for xi in interp_coords]) + ']'
                formatted_pi = '[' + ', '.join(['%.4E' % pi for pi in best_coords]) + ']'
                print 'Rank %i: %s gid: %i, target: %s, result: %s, params: %s, error: %.4E, final' % \
                      (rank, population, gid, formatted_x, formatted_xi, formatted_pi, min_error)
                coords_dict[gid]['X Coordinate'] = np.append(coords_dict[gid]['X Coordinate'],
                                                             interp_coords[0]).astype('float32', copy=False)
                coords_dict[gid]['Y Coordinate'] = np.append(coords_dict[gid]['Y Coordinate'],
                                                             interp_coords[1]).astype('float32', copy=False)
                coords_dict[gid]['Z Coordinate'] = np.append(coords_dict[gid]['Z Coordinate'],
                                                             interp_coords[2]).astype('float32', copy=False)
                coords_dict[gid]['U Coordinate'] = np.append(coords_dict[gid]['U Coordinate'],
                                                             best_coords[0]).astype('float32', copy=False)
                coords_dict[gid]['V Coordinate'] = np.append(coords_dict[gid]['V Coordinate'],
                                                             best_coords[1]).astype('float32', copy=False)
                coords_dict[gid]['L Coordinate'] = np.append(coords_dict[gid]['L Coordinate'],
                                                             best_coords[2]).astype('float32', copy=False)
                coords_dict[gid]['Interpolation Error'] = np.append(coords_dict[gid]['Interpolation Error'],
                                                                    min_error).astype('float32', copy=False)
                count += 1
            sys.stdout.flush()
            append_cell_attributes(MPI._addressof(comm), coords_path, population, coords_dict,
                                   namespace='Interpolated Coordinates', io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
            del coords_dict
            gc.collect()
        global_count = comm.gather(count, root=0)
        if rank == 0:
            print 'Interpolation of %i %s cells took %i s' % (np.sum(global_count), population, time.time()-start_time)

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("interpolate_DG_soma_locations.py") != -1,sys.argv)+1):])

