from function_lib import *
import scipy.optimize as optimize
import random
from mpl_toolkits.mplot3d import Axes3D


coords_dir = '../morphologies/'
# coords_file = 'dentate_Sampled_Soma_Locations_test.h5'
coords_file = 'dentate_Sampled_Soma_Locations_050517.h5'

f = h5py.File(coords_dir+coords_file, 'r')


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

    def within_bounds(self, x, i=None):
        """
        For optimize.minimize, check that the current set of parameters are within the bounds.
        :param x: array
        :param i: int
        :return: bool
        """
        if i is None:
            for i in range(len(x)):
                if ((self.xmin[i] is not None and x[i] < self.xmin[i]) or
                        (self.xmax[i] is not None and x[i] > self.xmax[i])):
                    return False
        else:
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

    def fix_coords(self, x):
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
    #if not check_bounds(params):
    #    return 1e9
    params_rot = interp_points(params)

    error = 0.
    for d, di in zip(args, params_rot):
        error += ((d - di)/0.1) ** 2.

    return error


pmin = {'GC': [0.0314, -0.811, -2.5], 'MPP': [-0.051, -0.723, 0.5], 'LPP': [-0.051, -0.723, 1.5],
        'MC': [0.0314, -0.811, -4.5], 'NGFC': [-0.051, -0.723, 0.5], 'AAC': [-0.051, -0.723, -4.5],
        'BC': [-0.051, -0.723, -4.5], 'MOPP': [-0.051, -0.723, -0.5], 'HCC': [0.0314, -0.811, -4.5],
        'HC': [0.0314, -0.811, -4.5], 'IS': [0.0314, -0.811, -4.5]}
pmax = {'GC': [3.079, 4.476, 0.5], 'MPP': [3.173, 4.476, 2.5], 'LPP': [3.173, 4.476, 3.5],
        'MC': [3.079, 4.476, -1.5], 'NGFC': [3.173, 4.476, 3.5], 'AAC': [3.173, 4.476, 1.5],
        'BC': [3.173, 4.476, 1.5], 'MOPP': [3.173, 4.476, 3.5], 'HCC': [3.079, 4.476, 0.5],
        'HC': [3.079, 4.476, -1.5], 'IS': [3.079, 4.476, -1.5]}

start_time = time.time()
coords_dict = {}

"""
for population in (population for population in f['Populations']
                   if 'Interpolated Coordinates' in f['Populations'][population]):
    # U, V, L
    pmin = [-0.064, -0.810, lmin[population]]
    pmax = [3.187, 4.564, lmax[population]]
    l0 = lmin[population] + (lmax[population] - lmin[population]) / 2.
    p0 = [1.555, 1.876, l0]  # center of the volume
    check_bounds = CheckBounds(pmin, pmax)
    # indexes = np.where(f['Populations'][population]['Interpolated Coordinates']['Interpolation Error']['value'][:] >
    #                   1.)[0]
    # pop_size = len(f['Populations'][population]['Coordinates']['X Coordinate']['value'])
    # interval = max(1,int(pop_size/500))
    # for gid in range(pop_size)[::interval]:
    for gid_index in range(50):
    #for gid_index in range(pop_size):
    #for gid_index in indexes:
        gid = f['Populations'][population]['Coordinates']['X Coordinate']['gid'][gid_index]
        coords_dict[gid] = {}
        x = f['Populations'][population]['Coordinates']['X Coordinate']['value'][gid_index]
        y = f['Populations'][population]['Coordinates']['Y Coordinate']['value'][gid_index]
        z = f['Populations'][population]['Coordinates']['Z Coordinate']['value'][gid_index]

        # result = optimize.minimize(euc_distance, p0, method='L-BFGS-B', bounds=zip(pmin, pmax), args=([x, y, z],),
        #                           options={'disp': True})
        # result = optimize.minimize(euc_distance, p0, method='Nelder-Mead', args=([x, y, z],), options={'disp': True})
        this_p0 = p0
        formatted_x = '[' + ', '.join(['%.4E' % xi for xi in [x, y, z]]) + ']'
        for i in range(50):
            result = optimize.minimize(euc_distance, this_p0, method='Powell', args=([x, y, z],), options={'disp': False})
            formatted_xi = '[' + ', '.join(['%.4E' % xi for xi in interp_points(result.x)]) + ']'
            formatted_pi = '[' + ', '.join(['%.4E' % pi for pi in result.x]) + ']'
            print 'gid: %i, target: %s, result: %s, params: %s, error: %.4E, iterations: %i' % \
                  (gid, formatted_x, formatted_xi, formatted_pi, result.fun, i)
            fixed_coords = check_bounds.fix_coords(result.x)
            error = euc_distance(fixed_coords, [x, y, z])
            if error < 1. and check_bounds.within_bounds(fixed_coords):
                break
            else:
                # this_p0 = check_bounds.return_to_bounds(fixed_coords)
                this_p0 = check_bounds.point_in_bounds()
        coords_dict[gid]['u'] = result.x[0]
        coords_dict[gid]['v'] = result.x[1]
        coords_dict[gid]['l']  = result.x[2]
        coords_dict[gid]['err'] = result.fun
"""

namespace = 'Interpolated Coordinates'

populations = [population for population in f['Populations'] if namespace in f['Populations'][population]]
for population in populations:
    re_positioned = []
    # U, V, L
    this_pmin = pmin[population]
    this_pmax = pmax[population]
    check_bounds = CheckBounds(this_pmin, this_pmax)
    pop_size = len(f['Populations'][population][namespace]['U Coordinate']['value'])
    if 'Interpolation Error' in f['Populations'][population][namespace]:
        re_positioned = np.where(f['Populations'][population][namespace]['Interpolation Error']['value'][:] > 1.)[0]
    print 'population: %s, re-positioned %i out of %i' % (population, len(re_positioned), pop_size)
    indexes = random.sample(range(pop_size), min(pop_size, 5000))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(f['Populations'][population][namespace]['X Coordinate']['value'][:][indexes],
                f['Populations'][population][namespace]['Y Coordinate']['value'][:][indexes],
                f['Populations'][population][namespace]['Z Coordinate']['value'][:][indexes], c='grey', alpha=0.1)
    ax1.scatter(f['Populations'][population]['Coordinates']['X Coordinate']['value'][:][re_positioned],
                f['Populations'][population]['Coordinates']['Y Coordinate']['value'][:][re_positioned],
                f['Populations'][population]['Coordinates']['Z Coordinate']['value'][:][re_positioned], c='c', alpha=0.1)
    ax1.scatter(f['Populations'][population][namespace]['X Coordinate']['value'][:][re_positioned],
                f['Populations'][population][namespace]['Y Coordinate']['value'][:][re_positioned],
                f['Populations'][population][namespace]['Z Coordinate']['value'][:][re_positioned], c='r', alpha=0.1)
    ax1.set_title('Re-positioned '+population)
    plt.show()
    plt.close()


"""
from matplotlib import cm
this_cm = cm.get_cmap()
n = len(f['Populations'])
#colors = [this_cm(1.*i/(n-1)) for i in range(n)]
colors = ['k', 'b']

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
#for i, population in enumerate(f['Populations']):
#for i, population in enumerate(['MC', 'MPP']):
for i, population in enumerate(['MC']):
    if population != 'GC':
        pop_size = len(f['Populations'][population]['Coordinates']['X Coordinate']['value'])
        interval = 1  # max(1,int(pop_size/500))
        for gid in range(pop_size)[::interval]:
            x = f['Populations'][population]['Coordinates']['X Coordinate']['value'][gid]
            y = f['Populations'][population]['Coordinates']['Y Coordinate']['value'][gid]
            z = f['Populations'][population]['Coordinates']['Z Coordinate']['value'][gid]
            ax.scatter(x, y, z, c=colors[i], alpha=0.1)

plt.show()
"""