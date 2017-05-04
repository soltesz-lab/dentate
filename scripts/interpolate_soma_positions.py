from function_lib import *
import scipy.optimize as optimize
import random


coords_dir = '../morphologies/'
coords_file = 'dentate_Sampled_Soma_Locations_test.h5'

f = h5py.File(coords_dir+coords_file, 'r')

# U, V, L
p0 = [1.555, 1.876, -1.]
pmin = [-0.064, -0.810, -4.1]
pmax = [3.187, 4.564, 3.1]


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


check_bounds = CheckBounds(pmin, pmax)


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


start_time = time.time()
coords_dict = {}

population = 'MC'

indexes = np.where(f['Populations'][population]['Interpolated Coordinates']['Interpolation Error']['value'][:] > 1.)[0]

# pop_size = len(f['Populations'][population]['Coordinates']['X Coordinate']['value'])
# interval = max(1,int(pop_size/500))
# for gid in range(pop_size)[::interval]:

#for gid_index in range(1000):
#for gid_index in range(pop_size):
for gid_index in indexes:
    gid = f['Populations'][population]['Coordinates']['X Coordinate']['gid'][gid_index]
    coords_dict[gid] = {}
    x = f['Populations'][population]['Coordinates']['X Coordinate']['value'][gid_index]
    y = f['Populations'][population]['Coordinates']['Y Coordinate']['value'][gid_index]
    z = f['Populations'][population]['Coordinates']['Z Coordinate']['value'][gid_index]

    # result = optimize.minimize(euc_distance, p0, method='L-BFGS-B', bounds=zip(pmin, pmax), args=([x, y, z],),
    #                           options={'disp': True})
    # result = optimize.minimize(euc_distance, p0, method='Nelder-Mead', args=([x, y, z],), options={'disp': True})
    this_p0 = p0
    for i in range(5):
        result = optimize.minimize(euc_distance, this_p0, method='Powell', args=([x, y, z],), options={'disp': False})
        formatted_x = '[' + ', '.join(['%.4E' % xi for xi in [x, y, z]]) + ']'
        formatted_xi = '[' + ', '.join(['%.4E' % xi for xi in interp_points(result.x)]) + ']'
        print 'gid: %i, target: %s, result: %s, error: %.4E, iterations: %i' % (gid, formatted_x, formatted_xi,
                                                                                result.fun, i)
        if result.fun < 1.:
            break
        else:
            this_p0 = check_bounds.return_to_bounds(result.x)
    coords_dict[gid]['u'] = result.x[0]
    coords_dict[gid]['v'] = result.x[1]
    coords_dict[gid]['l']  = result.x[2]
    coords_dict[gid]['err'] = result.fun



"""
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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