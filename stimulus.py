import sys, time, gc, copy
import numpy as np
import h5py
from scipy.spatial.distance import euclidean
from neuroh5.io import read_cell_attributes, read_population_ranges, NeuroH5CellAttrGen
from dentate.utils import *


class SelectivityConfig(object):
    def __init__(self, input_config, local_random):
        """
        

        :param input_config: dict
        :param local_random: :class:'np.random.RandomState
        """
        self.num_modules = input_config['Number Modules']
        self.module_ids = list(range(input_config['Number Modules']))

        self.module_probability_width = input_config['Selectivity Module Parameters']['width']
        self.module_probability_displacement = input_config['Selectivity Module Parameters']['displacement']
        self.module_probability_offsets = \
            np.linspace(-self.module_probability_displacement, 1. + self.module_probability_displacement,
                        self.num_modules)
        self.get_module_probability = \
            np.vectorize(lambda distance, offset:
                         np.exp(-(old_div((distance - offset), (self.module_probability_width / 3. / np.sqrt(2.)))) ** 2.),
                         excluded=['offset'])

        self.get_grid_module_spacing = \
            lambda distance: input_config['Grid Spacing Parameters']['offset'] + \
                      input_config['Grid Spacing Parameters']['slope'] * \
                      (np.exp(old_div(distance, input_config['Grid Spacing Parameters']['tau'])) - 1.)
        self.grid_module_spacing = \
            [self.get_grid_module_spacing(distance) for distance in np.linspace(0., 1., self.num_modules)]
        self.grid_spacing_sigma = input_config['Grid Spacing Variance'] / 6.
        self.grid_field_width_concentration_factor = input_config['Field Width Concentration Factor']['grid']
        self.grid_module_orientation = [local_random.uniform(0., np.pi / 3.) for i in range(self.num_modules)]
        self.grid_orientation_sigma = input_config['Grid Orientation Variance'] / 6.

        self.place_field_width_concentration_factor = input_config['Field Width Concentration Factor']['place']
        self.place_module_field_widths = np.multiply(self.grid_module_spacing,
                                                     self.place_field_width_concentration_factor)
        self.place_module_field_width_sigma = input_config['Modular Place Field Width Variance'] / 6.
        self.non_modular_place_field_width_sigma = input_config['Non-modular Place Field Width Variance'] / 6.

    def get_module_probabilities(self, distance):
        p_modules = []
        for offset in self.module_probability_offsets:
            p_modules.append(self.get_module_probability(distance, offset))
        p_modules = np.array(p_modules, dtype='float32')
        p_sum = np.sum(p_modules, axis=0)
        if p_sum == 0.:
            raise RuntimeError('SelectivityModuleConfig: get_module_probabilities: problem computing selectivity module'
                               'identity probabilities for normalized distance: %.4f' % distance)
        p_density = np.divide(p_modules, p_sum)
        return p_density

    def plot_module_probabilities(self, **kwargs):
        import matplotlib.pyplot as plt
        from dentate.plot import clean_axes, default_fig_options, save_figure
        fig_options = copy.copy(default_fig_options)
        fig_options.update(kwargs)

        distances = np.linspace(0., 1., 1000)
        p_modules = [self.get_module_probability(distances, offset) for offset in self.module_probability_offsets]
        p_modules = np.array(p_modules)

        p_sum = np.sum(p_modules, axis=0)
        p_density = np.divide(p_modules, p_sum)
        fig, axes = plt.subplots(1,2, figsize=(10., 4.8))
        for i in range(len(p_modules)):
            axes[0].plot(distances, p_density[i,:], label='Module %i' % i)
        axes[0].set_title('Selectivity module assignment probabilities', fontsize=fig_options.fontSize)
        axes[0].set_xlabel('Normalized cell position', fontsize=fig_options.fontSize)
        axes[0].set_ylabel('Probability', fontsize=fig_options.fontSize)
        expected_field_widths = np.matmul(self.place_module_field_widths, p_density)
        axes[1].plot(distances, expected_field_widths, c='k')
        axes[1].set_title('Expected place field width', fontsize=fig_options.fontSize)
        axes[1].set_xlabel('Normalized cell position', fontsize=fig_options.fontSize)
        axes[1].set_ylabel('Place field width (cm)', fontsize=fig_options.fontSize)
        clean_axes(axes)
        fig.tight_layout()

        if fig_options.saveFig is not None:
            save_figure('%s selectivity module probabilities' % str(fig_options.saveFig), **fig_options())

        if fig_options.showFig:
            fig.show()

    def get_expected_place_field_width(self, p_modules):
        """
        While feedforward inputs to the DG (MPP and LPP) exhibit modular spatial selectivity, the populations in the
        hippocampus receive convergent input from multiple discrete modules. Their place fields are, therefore,
        "non-modular", but their widths will vary with position along the septo-temporal axis of the hippocampus.
        This method computes the expected width of a place field as a weighted mean of the input field widths. The
        provided probabilities (p_modules) should be pre-computed with get_module_probabilities(distance).
        :param p_modules: array
        :return: float
        """
        return np.average(self.place_module_field_widths, weights=p_modules)


class GridCell(object):
    def __init__(self, selectivity_type=None, arena=None, selectivity_config=None, 
                 peak_rate=None, distance=None, local_random=None, selectivity_attr_dict=None):
        """

        :param selectivity_type: int
        :param arena: namedtuple
        :param selectivity_config: :class:'SelectivityConfig'
        :param peak_rate: float
        :param distance: float; u arc distance normalized to reference layer
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """
        if selectivity_attr_dict is not None:
            self.init_from_attr_dict(selectivity_attr_dict)
        elif any([arg is None for arg in [selectivity_type, arena, selectivity_config, peak_rate, distance]]):
            raise RuntimeError('GridCell: missing argument(s) required for object construction')
        else:
            if local_random is None:
                local_random = np.random.RandomState()
                
            self.selectivity_type = selectivity_type
            self.peak_rate = peak_rate
            p_modules = selectivity_config.get_module_probabilities(distance)
            self.module_id = local_random.choice(selectivity_config.module_ids, p=p_modules)

            self.grid_spacing = selectivity_config.grid_module_spacing[self.module_id]
            if selectivity_config.grid_spacing_sigma > 0.:
                delta_grid_spacing_factor = local_random.normal(0., selectivity_config.grid_spacing_sigma)
                self.grid_spacing += self.grid_spacing * delta_grid_spacing_factor

            self.grid_orientation = selectivity_config.grid_module_orientation[self.module_id]
            if selectivity_config.grid_orientation_sigma > 0.:
                delta_grid_orientation = local_random.normal(0., selectivity_config.grid_orientation_sigma)
                self.grid_orientation += delta_grid_orientation

            x_bounds, y_bounds = get_2D_arena_bounds(arena=arena, margin=self.grid_spacing / 2.)

            self.x0 = local_random.uniform(*x_bounds)
            self.y0 = local_random.uniform(*y_bounds)
            
            self.grid_field_width_concentration_factor = selectivity_config.grid_field_width_concentration_factor

    def init_from_attr_dict(self, selectivity_attr_dict):
        self.selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        self.peak_rate = selectivity_attr_dict['Peak Rate'][0]
        self.module_id = selectivity_attr_dict['Module ID'][0]
        self.grid_spacing = selectivity_attr_dict['Grid Spacing'][0]
        self.grid_orientation = selectivity_attr_dict['Grid Orientation'][0]
        self.x0 = selectivity_attr_dict['X Offset'][0]
        self.y0 = selectivity_attr_dict['Y Offset'][0]
        self.grid_field_width_concentration_factor = selectivity_attr_dict['Field Width Concentration Factor'][0]

    def get_selectivity_attr_dict(self):
        return {'Selectivity Type': np.array([self.selectivity_type], dtype='uint8'),
                'Peak Rate': np.array([self.peak_rate], dtype='float32'),
                'Module ID': np.array([self.module_id], dtype='uint8'),
                'Grid Spacing': np.array([self.grid_spacing], dtype='float32'),
                'Grid Orientation': np.array([self.grid_orientation], dtype='float32'),
                'X Offset': np.array([self.x0], dtype='float32'),
                'Y Offset': np.array([self.y0], dtype='float32'),
                'Field Width Concentration Factor':
                    np.array([self.grid_field_width_concentration_factor], dtype='float32')
                }

    def get_rate_map(self, x, y):
        """

        :param x: array
        :param y: array
        :return: array
        """
        return np.multiply(get_grid_rate_map(self.x0, self.y0, self.grid_spacing, self.grid_orientation, x, y,
                                 a=self.grid_field_width_concentration_factor), self.peak_rate)


class PlaceCell(object):
    def __init__(self, selectivity_type=None, arena=None, selectivity_config=None, peak_rate=None, distance=None,
                 modular=None, num_field_probabilities=None, local_random=None, selectivity_attr_dict=None):
        """

        :param selectivity_type: int
        :param arena: namedtuple
        :param selectivity_config: :class:'SelectivityModuleConfig'
        :param peak_rate: float
        :param distance: float; u arc distance normalized to reference layer
        :param modular: bool
        :param num_field_probabilities: dict
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """
        if selectivity_attr_dict is not None:
            self.init_from_attr_dict(selectivity_attr_dict)
        elif any([arg is None for arg in [selectivity_type, arena, selectivity_config, peak_rate, distance, modular,
                                          num_field_probabilities]]):
            raise RuntimeError('PlaceCell: missing argument(s) required for object construction')
        else:
            if local_random is None:
                local_random = np.random.RandomState()
            self.selectivity_type = selectivity_type
            self.peak_rate = peak_rate
            p_modules = selectivity_config.get_module_probabilities(distance)
            if modular:
                self.module_id = local_random.choice(selectivity_config.module_ids, p=p_modules)
                self.mean_field_width = selectivity_config.place_module_field_widths[self.module_id]
            else:
                self.module_id = -1
                self.mean_field_width = selectivity_config.get_expected_place_field_width(p_modules)

            num_fields_array, p_num_fields = \
                normalize_num_field_probabilities(num_field_probabilities, return_item_arrays=True)
            self.num_fields = local_random.choice(num_fields_array, p=p_num_fields)
            self.field_width = []
            self.x0 = []
            self.y0 = []
            for i in range(self.num_fields):
                this_field_width = self.mean_field_width
                if modular:
                    if selectivity_config.place_module_field_width_sigma > 0.:
                        delta_field_width_factor = local_random.normal(0., selectivity_config.place_module_field_width_sigma)
                        this_field_width += self.mean_field_width * delta_field_width_factor
                else:
                    if selectivity_config.non_modular_place_field_width_sigma > 0.:
                        delta_field_width_factor = \
                            local_random.normal(0., selectivity_config.non_modular_place_field_width_sigma)
                        this_field_width += self.mean_field_width * delta_field_width_factor
                self.field_width.append(this_field_width)

                x_bounds, y_bounds = get_2D_arena_bounds(arena=arena, margin=this_field_width / 2.)
                this_x0 = local_random.uniform(*x_bounds)
                this_y0 = local_random.uniform(*y_bounds)
                
                self.x0.append(this_x0)
                self.y0.append(this_y0)

    def init_from_attr_dict(self, selectivity_attr_dict):
        self.selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        self.peak_rate = selectivity_attr_dict['Peak Rate'][0]
        self.module_id = selectivity_attr_dict['Module ID'][0]
        self.num_fields = selectivity_attr_dict['Num Fields'][0]
        self.field_width = selectivity_attr_dict['Field Width']
        self.x0 = selectivity_attr_dict['X Offset']
        self.y0 = selectivity_attr_dict['Y Offset']

    def get_selectivity_attr_dict(self):
        return {'Selectivity Type': np.array([self.selectivity_type], dtype='uint8'),
                'Peak Rate': np.array([self.peak_rate], dtype='float32'),
                'Module ID': np.array([self.module_id], dtype='int8'),
                'Num Fields': np.array([self.num_fields], dtype='uint8'),
                'Field Width': np.asarray(self.field_width, dtype='float32'),
                'X Offset': np.asarray(self.x0, dtype='float32'),
                'Y Offset': np.asarray(self.y0, dtype='float32')
                }

    def get_rate_map(self, x, y):
        """

        :param x: array
        :param y: array
        :return: array
        """
        rate_map = np.zeros_like(x, dtype='float32')
        for i in range(self.num_fields):
            rate_map = np.maximum(rate_map, get_place_rate_map(self.x0[i], self.y0[i], self.field_width[i], x, y))
        return np.multiply(rate_map, self.peak_rate)


def get_place_rate_map(x0, y0, width, x, y):
    """

    :param x0: float
    :param y0: float
    :param width: float
    :param x: array
    :param y: array
    :return: array
    """
    return np.exp(-(old_div((x - x0), (width / 3. / np.sqrt(2.)))) ** 2.) * \
           np.exp(-(old_div((y - y0), (width / 3. / np.sqrt(2.)))) ** 2.)


def get_grid_rate_map(x0, y0, spacing, orientation, x, y, a=0.7):
    """

    :param x0: float
    :param y0: float
    :param spacing: float
    :param orientation: float
    :param x: array
    :param y: array
    :param a: concentrates field width relative to grid spacing
    :return: array
    """
    b = -1.5
    theta_k = [np.deg2rad(-30.), np.deg2rad(30.), np.deg2rad(90.)]

    inner_sum = np.zeros_like(x)
    for theta in theta_k:
        inner_sum += np.cos((old_div((4. * np.pi), (np.sqrt(3.) * spacing))) *
                            (np.cos(theta - orientation) * (x - x0) +
                             np.sin(theta - orientation) * (y - y0)))
    transfer = lambda z: np.exp(a * (z - b)) - 1.
    max_rate = transfer(3.)
    rate_map = old_div(transfer(inner_sum), max_rate)

    return rate_map


def get_input_cell(selectivity_type, selectivity_type_names, population=None, input_config=None, arena=None,
                   selectivity_config=None, distance=None, local_random=None, selectivity_attr_dict=None):
    """

    :param selectivity_type: int
    :param selectivity_type_names: dict: {int: str}
    :param population: str
    :param input_config: dict
    :param arena: namedtuple
    :param selectivity_config: :class:'SelectivityConfig'
    :param distance: float; u arc distance normalized to reference layer
    :param local_random: :class:'np.random.RandomState'
    :param selectivity_attr_dict: dict
    :return: instance of one of various InputCell classes
    """
    selectivity_type_name = selectivity_type_names[selectivity_type]
    if selectivity_type not in selectivity_type_names:
        raise RuntimeError('get_input_cell: enumerated selectivity type: %i not recognized' % selectivity_type)

    if selectivity_attr_dict is not None:
        if selectivity_type_name == 'grid':
            input_cell = GridCell(selectivity_attr_dict=selectivity_attr_dict)
        elif selectivity_type_name == 'place':
            input_cell = PlaceCell(selectivity_attr_dict=selectivity_attr_dict)
        else:
            RuntimeError('get_input_cell: selectivity type: %s not yet implemented' % selectivity_type_name)
    elif any([arg is None for arg in [population, input_config, arena]]):
        raise RuntimeError('get_input_cell: missing argument(s) required to construct %s cell config object' %
                           selectivity_type_name)
    else:
        if population not in input_config['Peak Rate'] or selectivity_type not in input_config['Peak Rate'][population]:
            raise RuntimeError('get_input_cell: peak rate not specified for population: %s, selectivity type: '
                               '%s' % (population, selectivity_type_name))
        peak_rate = input_config['Peak Rate'][population][selectivity_type]

        if selectivity_type_name in ['grid', 'place']:
            if selectivity_config is None:
                raise RuntimeError('get_input_cell: missing required argument: selectivity_config')
            if distance is None:
                raise RuntimeError('get_input_cell: missing required argument: distance')
            if local_random is None:
                local_random = np.random.RandomState()
                print('get_input_cell: warning: local_random argument not provided - randomness will not be '
                      'reproducible')
        if selectivity_type_name == 'grid':
            input_cell = \
                GridCell(selectivity_type=selectivity_type, arena=arena, selectivity_config=selectivity_config,
                         peak_rate=peak_rate, distance=distance, local_random=local_random)
        elif selectivity_type_name == 'place':
            if population in input_config['Non-modular Place Selectivity Populations']:
                modular = False
            else:
                modular = True
            if population not in input_config['Number Place Fields Probabilities']:
                raise RuntimeError('get_input_cell: probabilities for number of place fields not specified for '
                                   'population: %s' % population)
            num_field_probabilities = input_config['Number Place Fields Probabilities'][population]
            input_cell = \
                PlaceCell(selectivity_type=selectivity_type, arena=arena, selectivity_config=selectivity_config,
                          peak_rate=peak_rate, distance=distance, modular=modular,
                          num_field_probabilities=num_field_probabilities, local_random=local_random)
        else:
            RuntimeError('get_input_cell: selectivity type: %s not yet implemented' % selectivity_type_name)

    return input_cell


def choose_input_selectivity_type(p, local_random):
    """

    :param p: dict: {str: float}
    :param local_random: :class:'np.random.RandomState'
    :return: str
    """
    if len(p) == 1:
        return list(p.keys())[0]
    return local_random.choice(list(p.keys()), p=list(p.values()))


def get_active_cell_matrix(pop_activity, threshold=2.):
    active_cell_matrix = np.zeros_like(pop_activity, dtype='float32')
    active_indexes = np.where(pop_activity >= threshold)
    active_cell_matrix[active_indexes] = 1.
    return active_cell_matrix


def normalize_num_field_probabilities(num_field_probabilities, return_item_arrays=False):
    """
    Normalize the values in a dictionary to sum to 1.
    :param p_dict: dict: {int: float}
    :param return_item_arrays: bool
    :return: dict or tuple of array
    """
    num_fields_array = np.arange(len(num_field_probabilities))
    p_num_fields = np.array([num_field_probabilities[i] for i in num_fields_array])
    p_num_fields_sum = np.sum(p_num_fields)
    if p_num_fields_sum <= 0.:
        raise RuntimeError('normalize_num_field_probabilities: invalid num_field_probabilities')
    p_num_fields /= p_num_fields_sum
    if return_item_arrays:
        return num_fields_array, p_num_fields
    return {i: p_num_fields[i] for i in range(len(p_num_fields))}


def calibrate_num_field_probabilities(num_field_probabilities, field_width, target_fraction_active=None, pop_size=10000,
                                      bins=100, threshold=2., peak_rate=20., random_seed=0, plot=False):
    """
    Distribute 2D gaussian place fields within a square arena with length 2 * field_width according to the specified
    probability distribution for number of place fields per cell. Modify the relative weight of the zero place field
    category to achieve the target average fraction active across bins of the specified resolution. Return the resulting
    modified num_field_probabilities.
    Field density is defined as # of place fields / per cell in population / square cm.
    :param num_field_probabilities: dict: {int: float}
    :param field_width: float (cm)
    :param target_fraction_active: float
    :param pop_size: int
    :param bins: int (divide field width into square bins to compute fraction active)
    :param threshold: float (Hz)
    :param peak_rate: float (Hz)
    :param random_seed: int
    :return: dict
    """
    x = np.linspace(-field_width / 2., field_width / 2., bins)
    y = np.linspace(-field_width / 2., field_width / 2., bins)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    arena_area = (2. * field_width) ** 2.

    local_np_random = np.random.RandomState()

    num_fields_array, p_num_fields = \
        normalize_num_field_probabilities(num_field_probabilities, return_item_arrays=True)

    iteration_label = ' before:' if target_fraction_active is not None else ''
    for iteration in range(2):
        local_np_random.seed(random_seed)
        population_num_fields = []
        pop_activity = np.zeros((pop_size, len(x), len(y)))
        for i in range(pop_size):
            num_fields = local_np_random.choice(num_fields_array, p=p_num_fields)
            population_num_fields.append(num_fields)
            for j in range(num_fields):
                coords = local_np_random.uniform(-field_width, field_width, size=(2,))
                pop_activity[i, :, :] = \
                    np.add(pop_activity[i, :, :],
                           peak_rate * get_place_rate_map(coords[0], coords[1], field_width, x_mesh, y_mesh))
        active_cell_matrix = get_active_cell_matrix(pop_activity, threshold)
        fraction_active_array = np.mean(active_cell_matrix, axis=0)
        fraction_active_mean = np.mean(fraction_active_array)
        fraction_active_variance = np.var(fraction_active_array)
        num_fields_mean = np.mean(population_num_fields)
        field_density = old_div(num_fields_mean, arena_area)

        print('calibrate_num_field_probabilities:%s field_width: %.2f, fraction active: mean: %.4f, var: %.4f; '
              'field_density: %.4E' % (iteration_label, field_width, fraction_active_mean, fraction_active_variance,
                                       field_density))
        if target_fraction_active is None:
            break
        if iteration == 0:
            correction_factor = old_div(target_fraction_active, fraction_active_mean)
            p_num_fields *= correction_factor
            p_active = np.sum(p_num_fields[1:])
            if p_active > 1.:
                raise RuntimeError('calibrate_num_field_probabilities: it is not possible to achieve the requested'
                                   'target fraction active: %.4f with the provided num_field_probabilities' %
                                   target_fraction_active)
            p_num_fields[0] = 1. - p_active
            iteration_label = ' after:'
    pop_activity_sum = np.sum(pop_activity, axis=0)

    if plot:
        import matplotlib.pyplot as plt
        import math
        from dentate.plot import clean_axes
        fig, axes = plt.subplots(3, 3, figsize=(9., 9.))
        for count, i in enumerate(range(0, pop_size, int(math.ceil(pop_size / 6.)))):
            axes[old_div(count, 3)][count % 3].pcolor(x_mesh, y_mesh, pop_activity[i])
        hist, edges = np.histogram(population_num_fields, bins=len(num_field_probabilities),
                                   range=(-0.5, len(num_field_probabilities) - 0.5), density=True)
        axes[2][0].bar(edges[1:] - 0.5, hist)
        axes[2][0].set_title('Number of place fields')
        axes[2][1].pcolor(x_mesh, y_mesh, pop_activity_sum, vmin=0.)
        axes[2][1].set_title('Summed population activity')
        axes[2][2].pcolor(x_mesh, y_mesh, fraction_active_array, vmin=0.)
        axes[2][2].set_title('Fraction active')
        clean_axes(axes)
        fig.suptitle('Field width: %.2f; Fraction active: %.4f' % (field_width, fraction_active_mean))
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.show()
    modified_num_field_probabilities = {i: p_num_fields[i] for i in range(len(p_num_fields))}
    from dentate.utils import print_param_dict_like_yaml
    print_param_dict_like_yaml(modified_num_field_probabilities)
    return modified_num_field_probabilities


def get_2D_arena_bounds(arena, margin=0.):
    """

    :param arena: namedtuple
    :return: tuple of (tuple of float)
    """
    vertices_x = np.asarray([v[0] for v in arena.domain.vertices])
    vertices_y = np.asarray([v[1] for v in arena.domain.vertices])
    arena_x_bounds = (np.min(vertices_x) - margin, np.max(vertices_x) + margin)
    arena_y_bounds = (np.min(vertices_y) - margin, np.max(vertices_y) + margin)

    return arena_x_bounds, arena_y_bounds


def get_2D_arena_spatial_mesh(arena, spatial_resolution=5., margin=0.):
    """

    :param arena: namedtuple
    :param spatial_resolution: float (cm)
    :param margin: float
    :return: tuple of array
    """
    arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena=arena, margin=margin)
    arena_x = np.arange(arena_x_bounds[0], arena_x_bounds[1] + spatial_resolution / 2., spatial_resolution)
    arena_y = np.arange(arena_y_bounds[0], arena_y_bounds[1] + spatial_resolution / 2., spatial_resolution)

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


def generate_linear_trajectory(trajectory, temporal_resolution=1., equilibration_duration=None):
    """
    Construct coordinate arrays for a spatial trajectory, considering run velocity to interpolate at the specified
    temporal resolution. Optionally, the trajectory can be prepended with extra distance traveled for a specified
    network equilibration time, with the intention that the user discards spikes generated during this period before
    analysis.
    :param trajectory: namedtuple
    :param temporal_resolution: float (ms)
    :param equilibration_duration: float (ms)
    :return: tuple of array
    """
    velocity = trajectory.velocity  # (cm / s)
    spatial_resolution = velocity / 1000. * temporal_resolution
    x = trajectory.path[:,0]
    y = trajectory.path[:,1]

    if equilibration_duration is not None:
        equilibration_distance = velocity / 1000. * equilibration_duration
        x = np.insert(x, 0, x[0] - equilibration_distance)
        y = np.insert(y, 0, y[0])
    else:
        equilibration_duration = 0.
        equilibration_distance = 0.

    segment_lengths = np.sqrt((np.diff(x) ** 2. + np.diff(y) ** 2.))
    distance = np.insert(np.cumsum(segment_lengths), 0, 0.)

    interp_distance = np.arange(distance.min(), distance.max() + spatial_resolution / 2., spatial_resolution)
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)
    t = old_div(interp_distance, velocity * 1000.)  # ms

    t -= equilibration_duration
    interp_distance -= equilibration_distance

    return t, interp_x, interp_y, interp_distance


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

        xsteps = old_div(abs(end_x - start_x), spatial_resolution)
        ysteps = old_div(abs(end_y - start_y), spatial_resolution)
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
    
    t = (old_div(interp_distance, velocity * 1000.))  # ms
    
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)
    
    d = interp_distance

    return t, interp_x, interp_y, d


def read_trajectory(input_path, arena_id, trajectory_id):
    """

    :param input_path: str (path to file)
    :param arena_id: str
    :param trajectory_id: str
    :return: tuple of array
    """
    trajectory_namespace = 'Trajectory %s %s' % (str(arena_id), str(trajectory_id))

    with h5py.File(input_path, 'r') as f:
        group = f[trajectory_namespace]
        x = group['x'][:]
        y = group['y'][:]
        d = group['d'][:]
        t = group['t'][:]
    return x, y, d, t


def read_stimulus(stimulus_path, stimulus_namespace, population, module=None):

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


def read_feature (feature_path, feature_namespace, population, module=None):

    feature_lst    = []

    attr_gen = read_cell_attributes(feature_path, population, namespace=feature_namespace)
    for gid, feature_dict in attr_gen:
        gid_module = feature_dict['Module'][0]
        if (module is None) or (module == gid_module):
            rate       = feature_dict['Rate Map']
            num_fields = feature_dict['Num Fields']
            feature_lst.append((gid, rate, num_fields))
 
    return feature_lst


def bin_stimulus_features(features, t, bin_size, time_range):
    """
    Continuous stimulus feature binning.

    Parameters
    ----------
    features: matrix of size "number of times each feature was recorded" x "number of features"
    t: a vector of size "number of times each feature was recorded"
    bin_size: size of time bins
    time_range: the start and end times for binning the stimulus


    Returns
    -------
    matrix of size "number of time bins" x "number of features in the output"
        the average value of each output feature in every time bin
    """

    t_start, t_end = time_range


    edges = np.arange(t_start, t_end, bin_size)
    nbins = edges.shape[0]-1 
    nfeatures = features.shape[1] 
    binned_features = np.empty([nbins, nfeatures])
    for i in range(nbins): 
        for j in range(nfeatures):
            delta = edges[i+1] - edges[i]
            bin_range = np.arange(edges[i], edges[i+1], delta / 5.)
            ip_vals = np.interp(bin_range, t, features[:,j])
            binned_features[i,j] = np.mean(ip_vals)

    return binned_features


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



        
