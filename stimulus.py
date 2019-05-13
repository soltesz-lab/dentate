import sys, time, gc
import numpy as np
import h5py
from scipy.spatial.distance import euclidean
from neuroh5.io import read_cell_attributes, read_population_ranges, NeuroH5CellAttrGen
from InputCell import make_input_cell

#  custom data type for type of feature selectivity
selectivity_grid = 0
selectivity_place = 1

def generate_expected_width(field_width_params, module_widths, offsets, positions=None):
    if positions is None:
        positions = np.linspace(0, 1, 1000)

    p_module = lambda width, offset: lambda x: np.exp(-((x - offset) / (width / 3. / np.sqrt(2.))) ** 2.)
    p_modules = [p_module(2./3, offset)(positions) for offset in offsets]
    p_sum = np.sum(p_modules, axis=0)

    expected_width = np.multiply(module_widths, np.transpose(p_modules / p_sum))
    mean_expected_width = np.sum(expected_width, axis=1)
  
    return mean_expected_width, positions



def generate_spatial_mesh(arena, scale_factor=1., resolution=5.):

    vertices_x = np.asarray([v[0] for v in arena.domain.vertices])
    vertices_y = np.asarray([v[1] for v in arena.domain.vertices])
    arena_x_bounds = [np.min(vertices_x) * scale_factor,
                      np.max(vertices_x) * scale_factor]
    arena_y_bounds = [np.min(vertices_y) * scale_factor,
                      np.max(vertices_y) * scale_factor]

    arena_x = np.arange(arena_x_bounds[0], arena_x_bounds[1], resolution)
    arena_y = np.arange(arena_y_bounds[0], arena_y_bounds[1], resolution)
    return np.meshgrid(arena_x, arena_y, indexing='ij')


def generate_spatial_offsets(N, arena, start=0, scale_factor=2.0):
    import rbf
    from rbf.pde.nodes import min_energy_nodes

    vert = arena.domain.vertices
    smp = arena.domain.simplices

    # evenly disperse the nodes over the domain
    out = min_energy_nodes(N,(vert,smp),iterations=50,dispersion_delta=0.15,start=start)
    nodes = out[0]
    scaled_nodes = nodes * scale_factor
    
    return (scaled_nodes, nodes, vert, smp)



def generate_linear_trajectory(arena, trajectory_name, spatial_resolution = 1.):
    t_offset, d_offset = 0., 0.

    trajectory = arena.trajectories[trajectory_name]

    velocity = trajectory.velocity
    path = trajectory.path
    x = path[:,0]
    y = path[:,1]

    dr = np.sqrt((np.diff(x)**2 + np.diff(y)**2)) # segment lengths
    distance = np.zeros_like(x)
    distance[1:] = np.cumsum(dr) # integrate path
    interp_distance = np.arange(distance.min(), distance.max(), spatial_resolution)
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)
    d = interp_distance
    t = interp_distance / velocity * 1000.  # ms
    
    t -= t_offset
    d -= d_offset

    return t, interp_x, interp_y, d


def generate_concentric_trajectory(arena, velocity = 30., spatial_resolution = 1., 
                                   origin_X = 0., origin_Y = 0., radius_range = np.arange(1., 0.05, -0.05),
                                   initial_theta = np.deg2rad(180.), theta_step = np.deg2rad(300)):

    # arena_dimension - minimum distance from origin to boundary (cm)
    arena_x_bounds = [np.min(arena.domain.vertices[:,0]) * scale_factor,
                      np.max(arena.domain.vertices[:,0]) * scale_factor]
    arena_y_bounds = [np.min(arena.domain.vertices[:,1]) * scale_factor,
                      np.max(arena.domain.vertices[:,1]) * scale_factor]

    start_theta = initial_theta
    start_x = origin_X + np.cos(start_theta) * arena_dimension[0]
    start_y = origin_Y + np.sin(start_theta) * arena_dimension[1]

    radius_range = np.min(arena_dimension) * radius_range
    
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


def acquire_fields_per_cell(ncells, field_probabilities, generator):
    field_probabilities = np.asarray(field_probabilities, dtype='float32')
    field_set = [i for i in range(field_probabilities.shape[0])]
    return generator.choice(field_set, p=field_probabilities, size=(ncells,))

def get_rate_maps(cells):
    rate_maps = []
    for gid in cells:
        cell = cells[gid]
        nx, ny = cell.nx, cell.ny
        rate_map = cell.rate_map.reshape(nx, ny)
        rate_maps.append(rate_map)
    return np.asarray(rate_maps, dtype='float32')

def fraction_active(cells, threshold):
    temp_cell = cells.values()[0]
    nx, ny = temp_cell.nx, temp_cell.ny
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
        for (gid, cell_dict) in cells:
            feature_type = cell_dict['Cell Type'][0]
            cell = make_input_cell(gid, feature_type, cell_dict)
            this_module = cell.module
            module_dict[this_module][gid] = cell
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



        
