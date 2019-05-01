import sys, time, gc
import numpy as np
import h5py
from scipy.spatial.distance import euclidean
from neuroh5.io import read_cell_attributes, read_population_ranges, NeuroH5CellAttrGen

#  custom data type for type of feature selectivity
selectivity_grid = 0
selectivity_place_field = 1

def generate_expected_width(field_width_params, module_widths, offsets, positions=None):
    if positions is None:
        positions = np.linspace(0, 1, 1000)

    p_module = lambda width, offset: lambda x: np.exp(-((x - offset) / (width / 3. / np.sqrt(2.))) ** 2.)
    p_modules = [p_module(2./3, offset)(positions) for offset in offsets]
    p_sum = np.sum(p_modules, axis=0)

    expected_width = np.multiply(module_widths, np.transpose(p_modules / p_sum))
    mean_expected_width = np.sum(expected_width, axis=1)
  
    return mean_expected_width, positions



def generate_mesh(scale_factor=1., arena_dimension=100., resolution=5.):
    arena_x_bounds = [-arena_dimension * scale_factor, arena_dimension * scale_factor]
    arena_y_bounds = [-arena_dimension * scale_factor, arena_dimension * scale_factor]

    arena_x = np.arange(arena_x_bounds[0], arena_x_bounds[1], resolution)
    arena_y = np.arange(arena_y_bounds[0], arena_y_bounds[1], resolution)
    return np.meshgrid(arena_x, arena_y, indexing='ij')


def generate_spatial_offsets(N, arena_dimension=100., scale_factor=2.0, maxit=10):
    import rbf
    from rbf.nodes import disperse
    from rbf.halton import halton

    # Define the problem domain with line segments.
    vert = np.array([[-arena_dimension,-arena_dimension],[-arena_dimension,arena_dimension],
                    [arena_dimension,arena_dimension],[arena_dimension,-arena_dimension]])
    smp = np.array([[0,1],[1,2],[2,3],[3,0]])

    # create N quasi-uniformly distributed nodes over the unit square
    nodes = halton(N,2)

    # scale/translate the nodes to encompass the arena
    nodes -= 0.5
    nodes = 2 * nodes * arena_dimension
    # evenly disperse the nodes over the domain using maxit iterative steps
    for i in range(maxit):
        nodes = disperse(nodes,vert,smp)
    scaled_nodes = nodes * scale_factor
    
    return (scaled_nodes, nodes,vert,smp)



def generate_trajectory(arena_dimension = 100., velocity = 30., spatial_resolution = 1., ramp_up_period=500.):  # cm
    xy_offset, t_offset, d_offset = 0., 0., 0.
    if ramp_up_period is not None:
        ramp_up_distance = (ramp_up_period / 1000.) * velocity  # cm
        xy_offset = ramp_up_distance / np.sqrt(2)
        t_offset = ramp_up_period
        d_offset = ramp_up_distance

    x = np.arange(-arena_dimension - xy_offset, arena_dimension, spatial_resolution)
    y = np.arange(-arena_dimension - xy_offset, arena_dimension, spatial_resolution)

    distance = np.insert(np.cumsum(np.sqrt(np.sum([np.diff(x) ** 2., np.diff(y) ** 2.], axis=0))), 0, 0.)
    interp_distance = np.arange(distance[0], distance[-1], spatial_resolution)
    t = interp_distance / velocity * 1000.  # ms
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)
    d = interp_distance
    
    t -= t_offset
    d -= d_offset

    return t, interp_x, interp_y, d


def generate_concentric_trajectory(arena_dimension = 100., velocity = 30., spatial_resolution = 1., 
                                   origin_X = 0., origin_Y = 0., radius_range = np.arange(100, 5, -5),
                                   initial_theta = np.deg2rad(180.), theta_step = np.deg2rad(300)):

    # arena_dimension - minimum distance from origin to boundary (cm)

    start_theta = initial_theta
    start_x = origin_X + np.cos(start_theta) * arena_dimension
    start_y = origin_Y + np.sin(start_theta) * arena_dimension

    xs = []
    ys = []
    for radius in radius_range[1:]:

        end_theta = start_theta + theta_step
        theta = np.arange(start_theta, end_theta, np.deg2rad(spatial_resolution))

        end_x = origin_X + np.cos(start_theta) * radius
        end_y = origin_Y + np.sin(start_theta) * radius

        xsteps = abs(end_x - start_x)  / spatial_resolution
        ysteps = abs(end_y - start_y)  / spatial_resolution
        nsteps = max(xsteps, ysteps)
        
        linear_x = np.linspace(start_x, end_x, nsteps)
        linear_y = np.linspace(start_y, end_y, nsteps)

        radial_x = origin_X + np.cos(theta) * radius
        radial_y = origin_Y + np.sin(theta) * radius

        x = np.concatenate([linear_x, radial_x])
        y = np.concatenate([linear_y, radial_y])

        xs.append(x)
        ys.append(y)
    
        start_theta = end_theta
        start_x = x[-1]
        start_y = y[-1]

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    
    distance = np.insert(np.cumsum(np.sqrt(np.sum([np.diff(x) ** 2., np.diff(y) ** 2.], axis=0))), 0, 0.)
    interp_distance = np.arange(distance[0], distance[-1], spatial_resolution)
    
    t = (interp_distance / velocity * 1000.)  # ms
    
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
            current_map = place_peak_rate * np.exp(-((interp_x - x_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.) * np.exp(-((interp_y  - y_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.)
            rate_map    = np.maximum(current_map, rate_map)
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
        ramp_up_region = np.where(interp_t <= 0.0)[0]
        nsteps = len(ramp_up_region)
        window = signal.hann(nsteps*2, sym=False)
        half_window = window[:int(nsteps)]
        half_window /= np.max(half_window)
        orig_response = response[ramp_up_region].copy()
        response[ramp_up_region] = response[ramp_up_region] * half_window
    
    return response

def get_rate_maps(cells):
    rate_maps = []
    for gid in cells:
        cell = cells[gid]
        nx, ny = cell['Nx'][0], cell['Ny'][0]
        rate_map = cell['Rate Map'].reshape(nx, ny)
        rate_maps.append(rate_map)
    return np.asarray(rate_maps, dtype='float32')

def fraction_active(cells, threshold):
    temp_cell = cells.values()[0]
    nx, ny = temp_cell['Nx'][0], temp_cell['Ny'][0]
    del temp_cell

    rate_maps = get_rate_maps(cells)
    nxx, nyy  = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = zip(nxx.reshape(-1,), nyy.reshape(-1,))
  
    factive = lambda px, py: calculate_fraction_active(rate_maps[:, px, py], threshold)
    return {(px, py): factive(px, py) for (px, py) in coords}

def calculate_fraction_active(rates, threshold):
    N = len(rates)
    num_active = len(np.where(rates > threshold)[0])
    fraction_active = np.divide(float(num_active), float(N))
    return fraction_active

def coefficient_of_variation(cells, eps=1.0e-6):
    rate_maps = get_rate_maps(cells)
    summed_map = np.sum(rate_maps, axis=0)
    
    mean = np.mean(summed_map)
    std  = np.std(summed_map)
    cov  = np.divide(std, mean + eps)
    return cov

def peak_to_trough(cells):
    rate_maps  = get_rate_maps(cells)
    summed_map = np.sum(rate_maps, axis=0)
    var_map    = np.var(rate_maps, axis=0)
    minmax_eval = 0.0
    var_eval    = 0.0

    return minmax_eval, var_eval      

def calculate_field_distribution(pi, pr):
    p1 = (1. - pi) / (1. + (7./4.) * pr)
    p2 = p1 * pr
    p3 = 0.5 * p2
    p4 = 0.5 * p3
    probabilities = np.array([pi, p1, p2, p3, p4], dtype='float32')
    assert(np.abs(np.sum(probabilities) - 1.) < 1.e-5)
    return probabilities

def gid2module_dictionary(cell_lst, modules):
    module_dict = {module: {} for module in modules}
    for cells in cell_lst:
        for (gid, cell) in cells:
            this_module = cell['Module'][0]
            module_dict[this_module][cell['gid'][0]] = cell
    return module_dict

def module2gid_dictionary(module_dict):
    gid_dict = dict()
    for module in module_dict:
        gid_dict.update(module_dict[module])
    return gid_dict
        

def read_trajectory(input_path, trajectory_id):

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


def read_stimulus (stimulus_path, stimulus_namespace, population, module=None):
        ratemap_lst    = []
        module_gid_lst = []
        if module is not None:
            if not isinstance(module, int):
                raise Exception('module variable must be an integer')
            gid_module_gen = read_cell_attributes(stimulus_path, population, namespace='Cell Attributes')
            for (gid, attr_dict) in gid_module_gen:
                this_module = attr_dict['Module'][0]
                if this_module == module:
                    module_gid_lst.append(gid)

        attr_gen = read_cell_attributes(stimulus_path, population, namespace=stimulus_namespace)
        for gid, stimulus_dict in attr_gen:
            if gid in module_gid_lst or module_gid_lst == []:
                rate       = stimulus_dict['rate']
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



