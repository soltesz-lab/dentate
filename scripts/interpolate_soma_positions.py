from function_lib import *
import scipy.optimize as optimize

coords_dir = '../morphologies/'
coords_file = 'dentate_Sampled_Soma_Locations.h5'

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


check_bounds = CheckBounds(pmin, pmax).within_bounds


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

start_time = time.time()
coords_dict = {}

population = 'MC'
pop_size = len(f['Populations'][population]['Coordinates']['X Coordinate']['value'])
# interval = max(1,int(pop_size/500))
# for gid in range(pop_size)[::interval]:
if population not in coords_dict:
    coords_dict[population] = {'gid': [], 'u': [], 'v': [], 'l': [], 'err': []}
#for gid_index in range(1000):
for gid_index in range(pop_size):
    gid = f['Populations'][population]['Coordinates']['X Coordinate']['gid'][gid_index]
    coords_dict[population]['gid'].append(gid)
    x = f['Populations'][population]['Coordinates']['X Coordinate']['value'][gid_index]
    y = f['Populations'][population]['Coordinates']['Y Coordinate']['value'][gid_index]
    z = f['Populations'][population]['Coordinates']['Z Coordinate']['value'][gid_index]

    # result = optimize.minimize(euc_distance, p0, method='L-BFGS-B', bounds=zip(pmin, pmax), args=([x, y, z],),
    #                           options={'disp': True})
    # result = optimize.minimize(euc_distance, p0, method='Nelder-Mead', args=([x, y, z],), options={'disp': True})
    result = optimize.minimize(euc_distance, p0, method='Powell', args=([x, y, z],), options={'disp': False})
    formatted_x = '[' + ', '.join(['%.4E' % xi for xi in [x, y, z]]) + ']'
    formatted_xi = '[' + ', '.join(['%.4E' % xi for xi in interp_points(result.x)]) + ']'
    coords_dict[population]['u'].append(result.x[0])
    coords_dict[population]['v'].append(result.x[1])
    coords_dict[population]['l'].append(result.x[2])
    coords_dict[population]['err'].append(result.fun)
    print 'gid: %i, target: %s, result: %s, error: %.4E' % (gid, formatted_x, formatted_xi, result.fun)
print 'Interpolation of %i %s cells took %i s' % (len(coords_dict[population]['u']), population, time.time()-start_time)


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