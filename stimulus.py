import sys, time, gc
import numpy as np
import h5py
from scipy.spatial.distance import euclidean
from neuroh5.io import read_cell_attributes, read_population_ranges, NeuroH5CellAttrGen
try:
    import rbf
    from rbf.nodes import disperse
    from rbf.halton import halton
except ImportError as e:
    print('dentate.stimulus: problem importing rbf module:', e)

#  custom data type for type of feature selectivity
selectivity_grid = 0
selectivity_place_field = 1


def generate_spatial_offsets(N, arena_dimension=100., scale_factor=2.0, maxit=10): 
    # Define the problem domain with line segments.
    vert = np.array([[-arena_dimension,-arena_dimension],[-arena_dimension,arena_dimension],
                    [arena_dimension,arena_dimension],[arena_dimension,-arena_dimension]])
    smp = np.array([[0,1],[1,2],[2,3],[3,0]])

    # create N quasi-uniformly distributed nodes over the unit square
    nodes = halton(N,2)

    # scale/translate the nodes to encompass the arena
    nodes -= 0.5
    scaled_nodes = (nodes * scale_factor * arena_dimension)
    
    # evenly disperse the nodes over the domain using maxit iterative steps
    for i in range(maxit):
        scaled_nodes = disperse(scaled_nodes,vert,smp)
    nodes = scaled_nodes / scale_factor
    return (scaled_nodes, nodes,vert,smp)



def generate_trajectory(arena_dimension = 100., velocity = 30., spatial_resolution = 1.):  # cm

    # arena_dimension - minimum distance from origin to boundary (cm)

    x = np.arange(-arena_dimension, arena_dimension, spatial_resolution)
    y = np.arange(-arena_dimension, arena_dimension, spatial_resolution)
    distance = np.insert(np.cumsum(np.sqrt(np.sum([np.diff(x) ** 2., np.diff(y) ** 2.], axis=0))), 0, 0.)
    interp_distance = np.arange(distance[0], distance[-1], spatial_resolution)
    t = interp_distance / velocity * 1000.  # ms
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)
    d = interp_distance

    return t, interp_x, interp_y, d

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

    

def generate_spatial_ratemap(selectivity_type, features_dict, interp_t, interp_x, interp_y,
                             grid_peak_rate, place_peak_rate, ramp_up_period=500.0, **kwargs):
    """

    :param selectivity_type: int
    :param features_dict: dict
    :param interp_x: array
    :param interp_y: array
    :param grid_peak_rate: float (Hz)
    :param place_peak_rate: float (Hz)
    :return: array
    """

    if interp_x.shape != interp_y.shape:
        raise Exception('x and y coordinates must have same size')
    
    selectivity_grid = 0
    selectivity_place = 1

    a = kwargs.get('a', 0.3)
    b = kwargs.get('b', -1.5)
 
    if 'X Offset Scaled' and 'Y Offset Scaled' in features_dict:
        x_offset = features_dict['X Offset Scaled']
        y_offset = features_dict['Y Offset Scaled']
    else:
        x_offset = features_dict['X Offset']
        y_offset = features_dict['Y Offset']

    rate_map = None
    if selectivity_type == selectivity_grid:
        grid_orientation = features_dict['Grid Orientation'][0]
        grid_spacing = features_dict['Grid Spacing'][0]
        theta_k   = [np.deg2rad(-30.), np.deg2rad(30.), np.deg2rad(90.)]
        inner_sum = np.zeros_like(interp_x)
        for theta in theta_k:
            inner_sum += np.cos( ((4. * np.pi) / (np.sqrt(3.) * grid_spacing)) * \
                         (np.cos(theta - grid_orientation) * (interp_x - x_offset[0]) \
                          + np.sin(theta - grid_orientation) * (interp_y - y_offset[0])))
        transfer = lambda z: np.exp(a * (z - b)) - 1.
        max_rate = transfer(3.)
        rate_map = grid_peak_rate * transfer(inner_sum) / max_rate

    elif selectivity_type == selectivity_place:
        field_width = features_dict['Field Width']
        nfields  = features_dict['Num Fields'][0]
        rate_map = np.zeros_like(interp_x)
        for n in xrange(nfields):
            rate_map += place_peak_rate * np.exp(-((interp_x - x_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.) * np.exp(-((interp_y  - y_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.)
    else:
        raise Exception('Could not find proper cell type')

    response = rate_map

   # u = lambda ori: (np.cos(ori), np.sin(ori))
   # ori_array = 2. * np.pi * np.array([-30., 30., 90.]) / 360.  # rads
   # g = lambda x: np.exp(a * (x - b)) - 1.
   # scale_factor = g(3.)
   # grid_rate = lambda grid_spacing, ori_offset, x_offset, y_offset: \
   #   lambda x, y: grid_peak_rate / scale_factor * \
   #   g(np.sum([np.cos(4. * np.pi / np.sqrt(3.) /
   #                        grid_spacing * np.dot(u(theta - ori_offset), (x - x_offset, y - y_offset)))
   #                 for theta in ori_array]))

   # place_rate = lambda field_width, x_offset, y_offset: \
   #   lambda x, y: place_peak_rate * np.exp(-((x - x_offset) / (field_width / 3. / np.sqrt(2.))) ** 2.) * \
   #   np.exp(-((y - y_offset) / (field_width / 3. / np.sqrt(2.))) ** 2.)
      

   # if selectivity_type == selectivity_grid:
   #     ori_offset = features_dict['Grid Orientation'][0]
   #     grid_spacing = features_dict['Grid Spacing'][0]
   #     x_offset = features_dict['X Offset'][0]
   #     y_offset = features_dict['Y Offset'][0]
   #     rate = np.vectorize(grid_rate(grid_spacing, ori_offset, x_offset, y_offset))
   # elif selectivity_type == selectivity_place_field:
   #     field_width = features_dict['Field Width'][0]
   #     x_offset = features_dict['X Offset'][0]
   #     y_offset = features_dict['Y Offset'][0]
   #     rate = np.vectorize(place_rate(field_width, x_offset, y_offset))

   # response = rate(interp_x, interp_y).astype('float32', copy=False)

    if ramp_up_period is not None:
        import scipy.signal as signal
        timestep = interp_t[1] - interp_t[0]
        fwhm = int(ramp_up_period*2 / timestep)
        ramp_up_region = np.where(interp_t <= ramp_up_period)[0]
        orig_response = response[ramp_up_region].copy()
        sigma = fwhm2sigma(fwhm)
        window = signal.gaussian(len(ramp_up_region)*2, std=sigma)
        half_window = window[:int(len(window)/2)]
        response[ramp_up_region] = response[ramp_up_region] * half_window
    
    return response

def fraction_active(cells, threshold):
    rate_maps = []
    for gid in cells:
        cell = cells[gid]
        nx, ny = cell['Nx'][0], cell['Ny'][0]
        rate_map = cell['Rate Map'].reshape(nx, ny)
        rate_maps.append(rate_map)
    rate_maps = np.asarray(rate_maps, dtype='float32')
    nxx, nyy  = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = zip(nxx.reshape(-1,), nyy.reshape(-1,))
  
    factive = lambda px, py: calculate_fraction_active(rate_maps[:, px, py], threshold)
    return {(px, py): factive(px, py) for (px, py) in coords}

def calculate_fraction_active(rates, threshold):
    N = len(rates)
    num_active = len(np.where(rates > threshold)[0])
    fraction_active = np.divide(float(num_active), float(N))
    return fraction_active


def gid2module_dictionary(cell_lst, modules):
    module_dict = {module: {} for module in modules}
    for cells in cell_lst:
        for gid in cells:
            cell = cells[gid]
            this_module = cell['Module'][0]
            module_dict[this_module][cell['gid'][0]] = cell
    return module_dict

def module2gid_dictionary(module_dict):
    gid_dict = dict()
    for module in module_dict:
        gid_dict.update(module_dict[module])
    return gid_dict
        

def read_cell_attributes(features_path, population, namespace, comm, io_size, cache_size):
    attr_gen = NeuroH5CellAttrGen(features_path, population, namespace=namespace, comm=comm, \
                                  io_size=io_size, cache_size=cache_size)
    cells = {gid: feature_dict for (gid, feature_dict) in attr_gen if gid is not None}
    return cells


def read_trajectory (input_path, trajectory_id):

    trajectory_namespace = 'Trajectory %s' % str(trajectory_id)

    with h5py.File(input_path, 'a') as f:
        group = f[trajectory_namespace]
        dataset = group['x']
        x = dataset[:]
        dataset = group['y']
        y = dataset[:]
        dataset = group['d']
        d = dataset[:]
        dataset = group['t']
        t = dataset[:]
    return (x,y,d,t)


def read_stimulus (stimulus_path, stimulus_namespace, population):
        ratemap_lst = []
        attr_gen = read_cell_attributes(stimulus_path, population, namespace=stimulus_namespace)
        for gid, stimulus_dict in attr_gen:
            rate = stimulus_dict['rate']
            spiketrain = stimulus_dict['spiketrain']
            modulation = stimulus_dict['modulation']
            peak_index = stimulus_dict['peak index']
            ratemap_lst.append((gid, rate, spiketrain, peak_index))

        ## sort by peak_index
        ratemap_lst.sort(key=lambda item: item[3])

        return ratemap_lst
            

##
## Linearize position
##
def linearize_trajectory (x, y):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    
    T   = np.concatenate((x,y))
    T_transform  = pca.fit_transform(T)
    T_linear     = pca.inverse_transform(T_transform)

    return T_linear



