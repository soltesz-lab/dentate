import os, sys, gc, copy, time
import numpy as np
from collections import defaultdict, ChainMap
from mpi4py import MPI
from dentate.utils import get_module_logger, object, range, str, Struct
from dentate.stgen import get_inhom_poisson_spike_times_by_thinning
from neuroh5.io import read_cell_attributes, append_cell_attributes, NeuroH5CellAttrGen
import h5py


## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = get_module_logger(__name__)


class InputSelectivityConfig(object):
    def __init__(self, stimulus_config, local_random):
        """
        We assume that "stimulus input cells" from MEC and LEC have grid spacing and spatial field widths that are
        topographically organized septo-temporally. Cells are assigned to one of ten discrete modules with distinct
        grid spacing and field width probabilistically according to septo-temporal position. The grid spacing across
        modules increases exponentially from 40 cm to 8 m. We then assume that "proxy input cells" from GC, MC, and
        CA3c neurons have place fields that result from sampling input from multiple discrete modules, and therefore
        have field widths that vary continuously with septo-temporal position, rather than clustering into discrete
        modules. Features are imposed on these "proxy input cells" during microcircuit clamp or network clamp
        simulations.
        :param stimulus_config: dict
        :param local_random: :class:'np.random.RandomState
        """
        self.num_modules = stimulus_config['Number Modules']
        self.module_ids = list(range(stimulus_config['Number Modules']))

        self.module_probability_width = stimulus_config['Selectivity Module Parameters']['width']
        self.module_probability_displacement = stimulus_config['Selectivity Module Parameters']['displacement']
        self.module_probability_offsets = \
            np.linspace(-self.module_probability_displacement, 1. + self.module_probability_displacement,
                        self.num_modules)
        self.get_module_probability = \
            np.vectorize(lambda distance, offset:
                         np.exp(
                             -(((distance - offset) / (self.module_probability_width / 3. / np.sqrt(2.)))) ** 2.),
                         excluded=['offset'])

        self.get_grid_module_spacing = \
            lambda distance: stimulus_config['Grid Spacing Parameters']['offset'] + \
                             stimulus_config['Grid Spacing Parameters']['slope'] * \
                             (np.exp(distance / stimulus_config['Grid Spacing Parameters']['tau']) - 1.)
        self.grid_module_spacing = \
            [self.get_grid_module_spacing(distance) for distance in np.linspace(0., 1., self.num_modules)]
        self.grid_spacing_sigma = stimulus_config['Grid Spacing Variance'] / 6.
        self.grid_field_width_concentration_factor = stimulus_config['Field Width Concentration Factor']['grid']
        self.grid_module_orientation = [local_random.uniform(0., np.pi / 3.) for i in range(self.num_modules)]
        self.grid_orientation_sigma = np.deg2rad(stimulus_config['Grid Orientation Variance'] / 6.)

        self.place_field_width_concentration_factor = stimulus_config['Field Width Concentration Factor']['place']
        self.place_module_field_widths = np.multiply(self.grid_module_spacing,
                                                     self.place_field_width_concentration_factor)
        self.place_module_field_width_sigma = stimulus_config['Modular Place Field Width Variance'] / 6.
        self.non_modular_place_field_width_sigma = stimulus_config['Non-modular Place Field Width Variance'] / 6.

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
        fig, axes = plt.subplots(1, 2, figsize=(10., 4.8))
        for i in range(len(p_modules)):
            axes[0].plot(distances, p_density[i, :], label='Module %i' % i)
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

    

class GridInputCellConfig(object):
    def __init__(self, selectivity_type=None, arena=None, selectivity_config=None,
                 peak_rate=None, distance=None, local_random=None, selectivity_attr_dict=None):
        """

        :param selectivity_type: int
        :param arena: namedtuple
        :param selectivity_config: :class:'InputSelectivityConfig'
        :param peak_rate: float
        :param distance: float; u arc distance normalized to reference layer
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """
        if selectivity_attr_dict is not None:
            self.init_from_attr_dict(selectivity_attr_dict)
        elif any([arg is None for arg in [selectivity_type, selectivity_config, peak_rate, distance]]):
            raise RuntimeError('GridInputCellConfig: missing argument(s) required for object construction')
        else:
            if local_random is None:
                local_random = np.random.RandomState()

            self.selectivity_type = selectivity_type
            self.peak_rate = peak_rate
            p_modules = selectivity_config.get_module_probabilities(distance)
            self.module_id = local_random.choice(selectivity_config.module_ids, p=p_modules)

            self.grid_spacing = selectivity_config.grid_module_spacing[self.module_id]
            if arena is None:
                arena_x_bounds, arena_y_bounds = (-self.grid_spacing, self.grid_spacing)
            else:
                arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena, margin=self.grid_spacing / 2.)

            if selectivity_config.grid_spacing_sigma > 0.:
                delta_grid_spacing_factor = local_random.normal(0., selectivity_config.grid_spacing_sigma)
                self.grid_spacing += self.grid_spacing * delta_grid_spacing_factor

            self.grid_orientation = selectivity_config.grid_module_orientation[self.module_id]
            if selectivity_config.grid_orientation_sigma > 0.:
                delta_grid_orientation = local_random.normal(0., selectivity_config.grid_orientation_sigma)
                self.grid_orientation += delta_grid_orientation

            self.x0 = local_random.uniform(*arena_x_bounds)
            self.y0 = local_random.uniform(*arena_y_bounds)

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

    def gather_attributes(self):
        """
        Select a subset of selectivity attributes to gather across a population. Cell attributes have one value per
        cell, component attributes have variable length per cell. A count is returned for the length of component
        attribute lists.
        :return: dict(val), count, None
        """
        gathered_cell_attr_dict = dict()
        gathered_cell_attr_dict['Module ID'] = self.module_id
        gathered_cell_attr_dict['Grid Spacing'] = self.grid_spacing
        gathered_cell_attr_dict['Grid Orientation'] = self.grid_orientation

        return gathered_cell_attr_dict, 0, None

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


class PlaceInputCellConfig(object):
    def __init__(self, selectivity_type=None, arena=None, normalize_scale=True, selectivity_config=None,
                 peak_rate=None, distance=None, modular=None, num_place_field_probabilities=None, field_width=None,
                 local_random=None, selectivity_attr_dict=None):
        """

        :param selectivity_type: int
        :param arena: namedtuple
        :param normalize_scale: bool; whether to interpret the scale of the num_place_field_probabilities distribution
                                        as normalized to the scale of the mean place field width
        :param selectivity_config: :class:'SelectivityModuleConfig'
        :param peak_rate: float
        :param distance: float; u arc distance normalized to reference layer
        :param modular: bool
        :param num_place_field_probabilities: dict
        :param field_width: float; option to enforce field_width rather than choose from distance-dependent distribution
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """
        if selectivity_attr_dict is not None:
            self.init_from_attr_dict(selectivity_attr_dict)
        elif any([arg is None for arg in [selectivity_type, selectivity_config, peak_rate, arena,
                                          num_place_field_probabilities]]):
            raise RuntimeError('PlaceInputCellConfig: missing argument(s) required for object construction')
        else:
            if local_random is None:
                local_random = np.random.RandomState()
            self.selectivity_type = selectivity_type
            self.peak_rate = peak_rate
            if field_width is not None:
                self.mean_field_width = field_width
                self.module_id = -1
                self.modular = False
            elif distance is None or modular is None:
                raise RuntimeError('PlaceInputCellConfig: missing argument(s) required for object construction')
            else:
                p_modules = selectivity_config.get_module_probabilities(distance)
                if modular:
                    self.module_id = local_random.choice(selectivity_config.module_ids, p=p_modules)
                    self.mean_field_width = selectivity_config.place_module_field_widths[self.module_id]
                else:
                    self.module_id = -1
                    self.mean_field_width = selectivity_config.get_expected_place_field_width(p_modules)
            self.num_fields = get_num_place_fields(num_place_field_probabilities, self.mean_field_width, arena=arena,
                                                   normalize_scale=normalize_scale, local_random=local_random)
            arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena=arena, margin=self.mean_field_width / 2.)
            self.field_width = []
            self.x0 = []
            self.y0 = []
            for i in range(self.num_fields):
                this_field_width = self.mean_field_width
                if modular is not None:
                    if modular:
                        if selectivity_config.place_module_field_width_sigma > 0.:
                            delta_field_width_factor = \
                                local_random.normal(0., selectivity_config.place_module_field_width_sigma)
                            this_field_width += self.mean_field_width * delta_field_width_factor
                    else:
                        if selectivity_config.non_modular_place_field_width_sigma > 0.:
                            delta_field_width_factor = \
                                local_random.normal(0., selectivity_config.non_modular_place_field_width_sigma)
                            this_field_width += self.mean_field_width * delta_field_width_factor
                self.field_width.append(this_field_width)

                this_x0 = local_random.uniform(*arena_x_bounds)
                this_y0 = local_random.uniform(*arena_y_bounds)

                self.x0.append(this_x0)
                self.y0.append(this_y0)

    def init_from_attr_dict(self, selectivity_attr_dict):
        self.selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        self.peak_rate = selectivity_attr_dict['Peak Rate'][0]
        self.module_id = selectivity_attr_dict.get('Module ID', [0])[0]
        self.num_fields = selectivity_attr_dict['Num Fields'][0]
        self.field_width = selectivity_attr_dict['Field Width']
        self.x0 = selectivity_attr_dict['X Offset']
        self.y0 = selectivity_attr_dict['Y Offset']

    def gather_attributes(self):
        """
        Select a subset of selectivity attributes to gather across a population. Cell attributes have one value per
        cell, component attributes have variable length per cell. A count is returned for the length of component
        attribute lists.
        :return: dict(val), count, dict(list)
        """
        gathered_cell_attr_dict = dict()
        gathered_comp_attr_dict = dict()

        gathered_cell_attr_dict['Module ID'] = self.module_id
        gathered_cell_attr_dict['Num Fields'] = self.num_fields

        count = len(self.field_width)
        gathered_comp_attr_dict['Field Width'] = self.field_width

        return gathered_cell_attr_dict, count, gathered_comp_attr_dict

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


    
class ConstantInputCellConfig(object):
    def __init__(self, selectivity_type=None, arena=None, selectivity_config=None,
                 peak_rate=None, local_random=None, selectivity_attr_dict=None):
        """

        :param selectivity_type: int
        :param arena: namedtuple
        :param selectivity_config: :class:'SelectivityModuleConfig'
        :param peak_rate: float
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """
        if selectivity_attr_dict is not None:
            self.init_from_attr_dict(selectivity_attr_dict)
        elif any([arg is None for arg in [selectivity_type, selectivity_config, peak_rate, arena]]):
            raise RuntimeError('ConstantInputCellConfig: missing argument(s) required for object construction')
        else:
            if local_random is None:
                local_random = np.random.RandomState()
            self.selectivity_type = selectivity_type
            self.peak_rate = peak_rate

    def init_from_attr_dict(self, selectivity_attr_dict):
        self.selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        self.peak_rate = selectivity_attr_dict['Peak Rate'][0]

    def get_selectivity_attr_dict(self):
        return {'Selectivity Type': np.array([self.selectivity_type], dtype='uint8'),
                'Peak Rate': np.array([self.peak_rate], dtype='float32'),
                }

    def get_rate_map(self, x, y):
        """

        :param x: array
        :param y: array
        :return: array
        """
        rate_map = np.ones_like(x, dtype='float32') * self.peak_rate
        return rate_map

def get_place_rate_map(x0, y0, width, x, y):
    """

    :param x0: float
    :param y0: float
    :param width: float
    :param x: array
    :param y: array
    :return: array
    """
    return np.exp(-(((x - x0) / (width / 3. / np.sqrt(2.)))) ** 2.) * \
           np.exp(-(((y - y0) / (width / 3. / np.sqrt(2.)))) ** 2.)


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
        inner_sum += np.cos(((4. * np.pi) / (np.sqrt(3.) * spacing)) *
                            (np.cos(theta - orientation) * (x - x0) +
                             np.sin(theta - orientation) * (y - y0)))
    transfer = lambda z: np.exp(a * (z - b)) - 1.
    max_rate = transfer(3.)
    rate_map = transfer(inner_sum) / max_rate

    return rate_map


def get_input_cell_config(selectivity_type, selectivity_type_names, population=None, stimulus_config=None,
                          arena=None, selectivity_config=None, distance=None, local_random=None,
                          selectivity_attr_dict=None):
    """

    :param selectivity_type: int
    :param selectivity_type_names: dict: {int: str}
    :param population: str
    :param stimulus_config: dict
    :param arena: namedtuple
    :param selectivity_config: :class:'InputSelectivityConfig'
    :param distance: float; u arc distance normalized to reference layer
    :param local_random: :class:'np.random.RandomState'
    :param selectivity_attr_dict: dict
    :return: instance of one of various InputCell classes
    """
    if selectivity_type not in selectivity_type_names:
        raise RuntimeError('get_input_cell_config: enumerated selectivity type: %i not recognized' % selectivity_type)
    selectivity_type_name = selectivity_type_names[selectivity_type]

    if selectivity_attr_dict is not None:
        if selectivity_type_name == 'grid':
            input_cell_config = GridInputCellConfig(selectivity_attr_dict=selectivity_attr_dict)
        elif selectivity_type_name == 'place':
            input_cell_config = PlaceInputCellConfig(selectivity_attr_dict=selectivity_attr_dict)
        elif selectivity_type_name == 'constant':
            input_cell_config = ConstantInputCellConfig(selectivity_attr_dict=selectivity_attr_dict)
        else:
            RuntimeError('get_input_cell_config: selectivity type %s is not supported' % selectivity_type_name)
    elif any([arg is None for arg in [population, stimulus_config, arena]]):
        raise RuntimeError('get_input_cell_config: missing argument(s) required to construct %s cell config object' %
                           selectivity_type_name)
    else:
        if population not in stimulus_config['Peak Rate'] or \
                selectivity_type not in stimulus_config['Peak Rate'][population]:
            raise RuntimeError('get_input_cell_config: peak rate not specified for population: %s, selectivity type: '
                               '%s' % (population, selectivity_type_name))
        peak_rate = stimulus_config['Peak Rate'][population][selectivity_type]

        if selectivity_type_name in ['grid', 'place']:
            if selectivity_config is None:
                raise RuntimeError('get_input_cell_config: missing required argument: selectivity_config')
            if distance is None:
                raise RuntimeError('get_input_cell_config: missing required argument: distance')
            if local_random is None:
                local_random = np.random.RandomState()
                logger.warning('get_input_cell_config: local_random argument not provided - randomness will not be '
                               'reproducible')
        if selectivity_type_name == 'grid':
            input_cell_config = \
                GridInputCellConfig(selectivity_type=selectivity_type, arena=arena,
                                    selectivity_config=selectivity_config, peak_rate=peak_rate, distance=distance,
                                    local_random=local_random)
        elif selectivity_type_name == 'place':
            if population in stimulus_config['Non-modular Place Selectivity Populations']:
                modular = False
            else:
                modular = True
            if population not in stimulus_config['Num Place Field Probabilities']:
                raise RuntimeError('get_input_cell_config: probabilities for number of place fields not specified for '
                                   'population: %s' % population)
            num_place_field_probabilities = stimulus_config['Num Place Field Probabilities'][population]
            input_cell_config = \
                PlaceInputCellConfig(selectivity_type=selectivity_type, arena=arena,
                                     selectivity_config=selectivity_config, peak_rate=peak_rate, distance=distance,
                                     modular=modular, num_place_field_probabilities=num_place_field_probabilities,
                                     local_random=local_random)
        elif selectivity_type_name == 'constant':
            input_cell_config = ConstantInputCellConfig(selectivity_type=selectivity_type, arena=arena,
                                                        selectivity_config=selectivity_config, peak_rate=peak_rate)
        else:
            RuntimeError('get_input_cell_config: selectivity type: %s not yet implemented' % selectivity_type_name)

    return input_cell_config


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


def get_num_place_fields(num_place_field_probabilities, field_width, arena=None, normalize_scale=True,
                         local_random=None):
    """
    Probability distributions for the number of place fields per cell are defined relative to the area of a standard
    arena size with dimension length = 2. * field_width. Given an arena with arbitrary dimensions, the number of place
    fields to distribute within the area of the arena is equivalent to a series of biased dice rolls.
    :param num_place_field_probabilities: dict: {int: float}
    :param field_width: float
    :param arena: namedtuple
    :param normalize_scale: bool
    :param local_random: :class:'np.random.RandomState'
    :return: int
    """
    num_fields_array, p_num_fields = normalize_num_place_field_probabilities(num_place_field_probabilities,
                                                                             return_item_arrays=True)
    if not normalize_scale:
        scale = 1.
    else:
        calibration_x_bounds = calibration_y_bounds = (-field_width, field_width)
        calibration_area = abs(calibration_x_bounds[1] - calibration_x_bounds[0]) * \
                           abs(calibration_y_bounds[1] - calibration_y_bounds[0])
        arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena, margin=field_width / 2.)
        arena_area = abs(arena_x_bounds[1] - arena_x_bounds[0]) * abs(arena_y_bounds[1] - arena_y_bounds[0])
        scale = arena_area / calibration_area
    num_fields = 0
    while scale > 0.:
        if scale >= 1.:
            scale -= 1.
        else:
            num_fields_array, p_num_fields = \
                rescale_non_zero_num_place_field_probabilities(num_place_field_probabilities, scale,
                                                               return_item_arrays=True)
            scale = 0.
        num_fields += local_random.choice(num_fields_array, p=p_num_fields)

    return num_fields


def normalize_num_place_field_probabilities(num_place_field_probabilities, return_item_arrays=False):
    """
    Normalizes provided probability distribution for the number of place fields per cell to sum to one. Can return
    either a dictionary or a tuple of array.
    :param num_place_field_probabilities: dict: {int: float}
    :param return_item_arrays: bool
    :return: dict or tuple of array
    """
    num_fields_array = np.arange(len(num_place_field_probabilities))
    p_num_fields = np.array([num_place_field_probabilities[i] for i in num_fields_array])
    p_num_fields_sum = np.sum(p_num_fields)
    if p_num_fields_sum <= 0.:
        raise RuntimeError('normalize_num_place_field_probabilities: invalid num_place_field_probabilities')
    p_num_fields /= p_num_fields_sum
    if return_item_arrays:
        return num_fields_array, p_num_fields
    return {i: p_num_fields[i] for i in range(len(p_num_fields))}


def rescale_non_zero_num_place_field_probabilities(num_place_field_probabilities, scale, return_item_arrays=False):
    """
    Modify a probability distribution for the number of place fields per cell by scaling the probabilities of one or
    greater fields, and compensating the probability of zero fields. Normalizes the modified probability distribution to
    sum to one. Can return either a dictionary or a tuple of array.
    :param num_place_field_probabilities: dict: {int: float}
    :param scale: float
    :param return_item_arrays: bool
    :return: dict or tuple of array
    """
    num_fields_array, p_num_fields = normalize_num_place_field_probabilities(num_place_field_probabilities,
                                                                             return_item_arrays=True)
    if scale <= 0.:
        raise RuntimeError('rescale_non_zero_num_place_field_probabilities: specified scale is invalid: %.3f' % scale)
    p_num_fields *= scale
    non_zero_p_num_fields_sum = np.sum(p_num_fields[1:])
    if non_zero_p_num_fields_sum > 1.:
        raise RuntimeError('rescale_non_zero_place_field_num_probabilities: the provided scale factor would generate '
                           'an invalid distribution of place_field_num_probabilities')
    p_num_fields[0] = 1. - non_zero_p_num_fields_sum

    if return_item_arrays:
        return num_fields_array, p_num_fields
    return {i: p_num_fields[i] for i in range(len(p_num_fields))}


def calibrate_num_place_field_probabilities(num_place_field_probabilities, field_width, peak_rate=20., threshold=2.,
                                            target_fraction_active=None, pop_size=10000, bins=100, selectivity_type=1,
                                            arena=None, normalize_scale=True, selectivity_config=None, random_seed=0,
                                            plot=False):
    """
    Given a probability distribution of the number of place fields per cell and a 2D arena, this method can either
    report the fraction of the population that will be active per unit area, or re-scale the probability distribution to
    achieve a target fraction active. The argument "normalize_scale" interprets the probability distribution as being
    defined for an intrinsic area that scales with field_width. When "normalize_scale" is set to False, this method will
    instead interpret the probability distribution as being defined over the area of the provided arena, buffered by
    margins that scale with field_width. If a target_fraction_active is provided, the distribution is modified by
    scaling the probabilities of one or greater fields, and compensating the probability of zero fields. Resulting
    modified probability distribution sums to one. Returns a dictionary.
    :param num_place_field_probabilities: dict: {int: float}
    :param field_width: float
    :param peak_rate: float
    :param threshold: float
    :param target_fraction_active: float
    :param pop_size: int
    :param bins: int
    :param selectivity_type: int
    :param arena: namedtuple
    :param normalize_scale: bool
    :param selectivity_config: :class:'InputSelectivityConfig'
    :param random_seed: int
    :param plot: bool
    :return: dict: {int: float}
    """
    if arena is None:
        inner_arena_x_bounds, inner_arena_y_bounds = (-field_width / 2., field_width / 2.)
        outer_arena_x_bounds, outer_arena_y_bounds = (-field_width, field_width)
    else:
        inner_arena_x_bounds, inner_arena_y_bounds = get_2D_arena_bounds(arena)
        outer_arena_x_bounds, outer_arena_y_bounds = get_2D_arena_bounds(arena, margin=field_width / 2.)

    x = np.linspace(inner_arena_x_bounds[0], inner_arena_x_bounds[1], bins)
    y = np.linspace(inner_arena_y_bounds[0], inner_arena_y_bounds[1], bins)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')

    arena_area = abs(outer_arena_x_bounds[1] - outer_arena_x_bounds[0]) * \
                 abs(outer_arena_y_bounds[1] - outer_arena_y_bounds[0])

    local_random = np.random.RandomState()

    iteration_label = ' before:' if target_fraction_active is not None else ''
    for iteration in range(2):
        local_random.seed(random_seed)
        population_num_fields = []
        pop_activity = np.zeros((pop_size, len(x), len(y)))
        for i in range(pop_size):
            input_cell_config = \
                PlaceInputCellConfig(selectivity_type=selectivity_type, arena=arena, normalize_scale=normalize_scale,
                                     selectivity_config=selectivity_config, peak_rate=peak_rate,
                                     num_place_field_probabilities=num_place_field_probabilities,
                                     field_width=field_width, local_random=local_random)
            num_fields = input_cell_config.num_fields
            population_num_fields.append(num_fields)
            if num_fields > 0:
                rate_map = input_cell_config.get_rate_map(x_mesh, y_mesh)
                pop_activity[i, :, :] = rate_map
        active_cell_matrix = get_active_cell_matrix(pop_activity, threshold)
        fraction_active_array = np.mean(active_cell_matrix, axis=0)
        fraction_active_mean = np.mean(fraction_active_array)
        fraction_active_variance = np.var(fraction_active_array)
        field_density = np.mean(population_num_fields) / arena_area

        print('calibrate_num_place_field_probabilities:%s field_width: %.2f, fraction active: mean: %.4f, var: %.4f; '
              'field_density: %.4E' % (iteration_label, field_width, fraction_active_mean, fraction_active_variance,
                                       field_density))
        sys.stdout.flush()
        if target_fraction_active is None:
            break
        if iteration == 0:
            scale = target_fraction_active / fraction_active_mean
            num_place_field_probabilities = \
                rescale_non_zero_num_place_field_probabilities(num_place_field_probabilities, scale)
            iteration_label = ' after:'
    pop_activity_sum = np.sum(pop_activity, axis=0)

    if plot:
        import matplotlib.pyplot as plt
        import math
        from dentate.plot import clean_axes
        fig, axes = plt.subplots(2, 3, figsize=(9., 6.))
        for count, i in enumerate(range(0, pop_size, int(math.ceil(pop_size / 6.)))):
            axes[count // 3][count % 3].pcolor(x_mesh, y_mesh, pop_activity[i])
        clean_axes(axes)
        fig.suptitle('Field width: %.2f; Fraction active: %.4f' % (field_width, fraction_active_mean))
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.show()

        fig, axes = plt.subplots()
        bins = max(population_num_fields) + 1
        hist, edges = np.histogram(population_num_fields, bins=bins, range=(-0.5, bins - 0.5), density=True)
        axes.bar(edges[1:] - 0.5, hist)
        fig.suptitle('Number of place fields\nField width: %.2f' % field_width)
        clean_axes(axes)
        fig.subplots_adjust(top=0.85)
        fig.show()

        fig, axes = plt.subplots()
        pc = axes.pcolor(x_mesh, y_mesh, pop_activity_sum, vmin=0.)
        cb = fig.colorbar(pc, ax=axes)
        cb.set_label('Firing rate (Hz)', rotation=270., labelpad=20.)
        fig.suptitle('Summed population activity\nField width: %.2f' % field_width)
        clean_axes(axes)
        fig.subplots_adjust(top=0.85)
        fig.show()

        fig, axes = plt.subplots()
        pc = axes.pcolor(x_mesh, y_mesh, fraction_active_array, vmin=0.)
        cb = fig.colorbar(pc, ax=axes)
        cb.set_label('Fraction active', rotation=270., labelpad=20.)
        fig.suptitle('Fraction active\nField width: %.2f' % field_width)
        clean_axes(axes)
        fig.subplots_adjust(top=0.85)
        fig.show()

    return num_place_field_probabilities


def get_2D_arena_bounds(arena, margin=0.):
    """

    :param arena: namedtuple
    :return: tuple of (tuple of float)
    """
    vertices_x = np.asarray([v[0] for v in arena.domain.vertices], dtype='float32')
    vertices_y = np.asarray([v[1] for v in arena.domain.vertices], dtype='float32')
    arena_x_bounds = (np.min(vertices_x) - margin, np.max(vertices_x) + margin)
    arena_y_bounds = (np.min(vertices_y) - margin, np.max(vertices_y) + margin)

    return arena_x_bounds, arena_y_bounds


def get_2D_arena_spatial_mesh(arena, spatial_resolution=5., margin=0., indexing='ij'):
    """

    :param arena: namedtuple
    :param spatial_resolution: float (cm)
    :param margin: float
    :return: tuple of array
    """
    arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena=arena, margin=margin)
    arena_x = np.arange(arena_x_bounds[0], arena_x_bounds[1] + spatial_resolution / 2., spatial_resolution)
    arena_y = np.arange(arena_y_bounds[0], arena_y_bounds[1] + spatial_resolution / 2., spatial_resolution)

    return np.meshgrid(arena_x, arena_y, indexing=indexing)


def generate_spatial_offsets(N, arena, start=0, scale_factor=2.0):
    import rbf
    from rbf.pde.nodes import min_energy_nodes

    vert = arena.domain.vertices
    smp = arena.domain.simplices

    # evenly disperse the nodes over the domain
    out = min_energy_nodes(N, (vert, smp), iterations=50, dispersion_delta=0.15, start=start)
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
    x = trajectory.path[:, 0]
    y = trajectory.path[:, 1]

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
    t = interp_distance / (velocity / 1000.)  # ms

    t = np.subtract(t, equilibration_duration)
    interp_distance -= equilibration_distance

    return t, interp_x, interp_y, interp_distance


def generate_concentric_trajectory(arena, velocity=30., spatial_resolution=1., scale_factor=1.0,
                                   origin_X=0., origin_Y=0., radius_range=np.arange(1., 0.05, -0.05),
                                   initial_theta=np.deg2rad(180.), theta_step=np.deg2rad(300)):
    # arena_dimension - minimum distance from origin to boundary (cm)
    arena_x_bounds = [np.min(arena.domain.vertices[:, 0]) * scale_factor,
                      np.max(arena.domain.vertices[:, 0]) * scale_factor]
    arena_y_bounds = [np.min(arena.domain.vertices[:, 1]) * scale_factor,
                      np.max(arena.domain.vertices[:, 1]) * scale_factor]

    start_theta = initial_theta
    start_x = origin_X + np.cos(start_theta) * arena_x_bounds[0]
    start_y = origin_Y + np.sin(start_theta) * arena_y_bounds[0]

    radius_range = (arena_x_bounds[1] - arena_x_bounds[0]) * radius_range

    xs = []
    ys = []
    for radius in radius_range[1:]:
        end_theta = start_theta + theta_step
        theta = np.arange(start_theta, end_theta, np.deg2rad(spatial_resolution))

        end_x = origin_X + np.cos(start_theta) * radius
        end_y = origin_Y + np.sin(start_theta) * radius

        xsteps = abs(end_x - start_x) / spatial_resolution
        ysteps = abs(end_y - start_y) / spatial_resolution
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

    t = (interp_distance / velocity) * 1000.  # ms

    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)

    d = interp_distance

    return t, interp_x, interp_y, d


def generate_input_selectivity_features(env, population, arena, selectivity_config, selectivity_type_names,
                                        selectivity_type_namespaces, coords_path, output_path,
                                        distances_namespace='Arc Distances', comm=None, io_size=-1, cache_size=10,
                                        write_every=1, chunk_size=1000, value_chunk_size=1000, dry_run=False,
                                        debug=False):
    """
    Generates input selectivity features for the given population and
    writes to the given output path, in the selectivity type-specific
    namespace provided through argument selectivity_type_namespaces.
    The set of selectivity attributes is determined by procedure
    get_selectivity_attr_dict in the respective input cell
    configuration class (e.g. PlaceInputCellConfig or GridInputCellConfig).

    :param env
    :param population: str
    :param arena: str
    :param selectivity_config: 
    :param selectivity_type_names: 
    :param selectivity_type_namespaces: 
    :param coords_path: str
    :param output_path: str
    :param distances_namespace: str
    :param comm: 
    :param io_size: int
    :param cache_size: int
    :param chunk_size: int
    :param write_every: int
    :param cache_size: int
    :param value_chunk_size: int
    :param verbose: bool
    :param debug: bool
    :param dry_run: bool

    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if io_size <= 0:
        io_size = comm.size
    rank = comm.rank

    if rank == 0:
        logger.info('Generating input selectivity features for population %s...' % population)

    reference_u_arc_distance_bounds = None
    if rank == 0:
        try:
            with h5py.File(coords_path, 'r') as coords_f:
                reference_u_arc_distance_bounds = \
                  coords_f['Populations'][population][distances_namespace].attrs['Reference U Min'], \
                  coords_f['Populations'][population][distances_namespace].attrs['Reference U Max']
        except Exception:
            raise RuntimeError('generate_input_selectivity_features: problem locating attributes '
                                'containing reference bounds in namespace: %s for population: %s from '
                                'coords_path: %s' % (distances_namespace, population, coords_path))

    reference_u_arc_distance_bounds = comm.bcast(reference_u_arc_distance_bounds, root=0)

    arena_x_mesh, arena_y_mesh = None, None
    if rank == 0:
        arena_x_mesh, arena_y_mesh = \
            get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=env.stimulus_config['Spatial Resolution'])
    arena_x_mesh = comm.bcast(arena_x_mesh, root=0)
    arena_y_mesh = comm.bcast(arena_y_mesh, root=0)
    
    selectivity_seed_offset = int(env.model_config['Random Seeds']['Input Selectivity'])
    local_random = np.random.RandomState()
    
    pop_norm_distances = {}
    rate_map_sum = defaultdict(lambda: np.zeros_like(arena_x_mesh))
    start_time = time.time()
    gid_count = defaultdict(lambda: 0)
    
    distances_attr_gen = NeuroH5CellAttrGen(coords_path, population, namespace=distances_namespace, comm=comm, io_size=io_size, cache_size=cache_size)

    if rank == 0:
        logger.info('generating selectivity features for %s cells...' % (population))

    selectivity_attr_dict = dict((key, dict()) for key in env.selectivity_types)
    for iter_count, (gid, distances_attr_dict) in enumerate(distances_attr_gen):
        if gid is not None:
            u_arc_distance = distances_attr_dict['U Distance'][0]
            norm_u_arc_distance = ((u_arc_distance - reference_u_arc_distance_bounds[0]) /
                                   (reference_u_arc_distance_bounds[1] - reference_u_arc_distance_bounds[0]))

            local_random.seed(int(selectivity_seed_offset + gid))
            this_selectivity_type = \
              choose_input_selectivity_type(p=env.stimulus_config['Selectivity Type Probabilities'][population],
                                            local_random=local_random)
            input_cell_config = get_input_cell_config(
                    selectivity_type=this_selectivity_type, selectivity_type_names=selectivity_type_names,
                    population=population, stimulus_config=env.stimulus_config, arena=arena,
                    selectivity_config=selectivity_config, distance=norm_u_arc_distance, local_random=local_random)
            this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
            selectivity_attr_dict[this_selectivity_type_name][gid] = input_cell_config.get_selectivity_attr_dict()
            rate_map = input_cell_config.get_rate_map(x=arena_x_mesh, y=arena_y_mesh)
            if debug and rank == 0:
                callback, context = debug
                this_context = Struct(**dict(context()))
                this_context.update(dict(locals()))
                callback(this_context)
            selectivity_attr_dict[this_selectivity_type_name][gid]['Arena Rate Map'] = \
              np.asarray(rate_map, dtype='float32').reshape(-1,)
            gid_count[this_selectivity_type_name] += 1

            pop_norm_distances[gid] = norm_u_arc_distance
            rate_map_sum[this_selectivity_type_name] = \
              np.add(rate_map_sum[this_selectivity_type_name], rate_map)

        gid_count_dict = dict(gid_count.items())
        gid_count_dict = comm.gather(gid_count_dict, root=0)
        if rank == 0:
            merged_gid_count = defaultdict(lambda: 0)
            for each_gid_count in gid_count_dict:
                for selectivity_type_name in each_gid_count:
                    merged_gid_count[selectivity_type_name] += each_gid_count[selectivity_type_name]
            total_gid_count = np.sum(list(merged_gid_count.values()))

        if (iter_count > 0 and iter_count % write_every == 0) or (debug and iter_count == 10):
            if rank == 0:
                logger.info('generated selectivity features for %i %s cells' % (total_gid_count, population))
            if not dry_run:
                for selectivity_type_name in sorted(selectivity_attr_dict.keys()):
                    if rank == 0:
                        logger.info('writing selectivity type %s...' % selectivity_type_name)
                    selectivity_type_namespace = selectivity_type_namespaces[selectivity_type_name]
                    append_cell_attributes(output_path, population, selectivity_attr_dict[selectivity_type_name],
                                           namespace=selectivity_type_namespace, comm=comm, io_size=io_size,
                                           chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            del selectivity_attr_dict
            selectivity_attr_dict = dict((key, dict()) for key in env.selectivity_types)

    if not dry_run:
        for selectivity_type_name in sorted(selectivity_attr_dict.keys()):
            if rank == 0:
                logger.info('writing selectivity type %s...' % selectivity_type_name)
            selectivity_type_namespace = selectivity_type_namespaces[selectivity_type_name]
            append_cell_attributes(output_path, population, selectivity_attr_dict[selectivity_type_name],
                                   namespace=selectivity_type_namespace, comm=comm, io_size=io_size,
                                   chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        del selectivity_attr_dict
    if rank == 0:
        for selectivity_type_name in merged_gid_count:
            logger.info('generated selectivity features for %i/%i %s %s cells in %.2f s' %
                (merged_gid_count[selectivity_type_name], total_gid_count, population, selectivity_type_name,
                (time.time() - start_time)))


    comm.barrier()
    
    return pop_norm_distances, rate_map_sum

def generate_input_spike_trains(env, population, trajectory, selectivity_path, selectivity_type_names, selectivity_type_namespaces, output_path, output_namespace='Input Spikes', spike_train_attr_name='Spike Train', spike_hist_resolution=1000, equilibrate=None, comm=None, io_size=-1, cache_size=10, write_every=1, chunk_size=1000, value_chunk_size=1000, dry_run=False, debug=False):
    """
    Generates spike trains for the given population according to the
    input selectivity rate maps contained in the given selectivity
    file, and writes the spike trains to the given output path and
    namespace provided through argument output_namespace.

    :param env:
    :param population: str
    :param trajectory: 
    :param selectivity_path: str (path to file)
    :param selectivity_type_names: 
    :param selectivity_type_namespaces: 
    :param output_path: str (path to file)
    :param output_namespace: str
    :param spike_train_attr_name: str
    :param spike_hist_resolution: str
    :param equilibrate: bool
    :param comm: 
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param write_every: int
    :param debug: bool
    :param dry_run: bool
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    if io_size <= 0:
        io_size = comm.size
    rank = comm.rank

    t, x, y, d = trajectory

    
    local_random = np.random.RandomState()
    input_spike_train_offset = int(env.model_config['Random Seeds']['Input Spiketrains'])

    spike_hist_sum = defaultdict(lambda: np.zeros(spike_hist_resolution))

    process_time = dict()
    for this_selectivity_namespace in sorted(selectivity_type_namespaces[population]):

        if rank == 0:
            logger.info('Generating input source spike trains for population %s [%s]...' % (population, this_selectivity_namespace))
            
        start_time = time.time()
        selectivity_attr_gen = NeuroH5CellAttrGen(selectivity_path, population,
                                                  namespace=this_selectivity_namespace, comm=comm, io_size=io_size,
                                                  cache_size=cache_size)
        spikes_attr_dict = dict()
        selectivity_type_name = None
        gid_count = 0
        for iter_count, (gid, selectivity_attr_dict) in enumerate(selectivity_attr_gen):
            if gid is not None:
                this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
                this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
                if selectivity_type_name is None:
                    selectivity_type_name = this_selectivity_type_name
                input_cell_config = \
                  get_input_cell_config(selectivity_type=this_selectivity_type,
                                            selectivity_type_names=selectivity_type_names,
                                            selectivity_attr_dict=selectivity_attr_dict)
                rate_map = input_cell_config.get_rate_map(x=x, y=y)
                if equilibrate is not None:
                    equilibrate_filter, equilibrate_len = equilibrate
                    rate_map[:equilibrate_len] = np.multiply(rate_map[:equilibrate_len], equilibrate_filter)
                local_random.seed(int(input_spike_train_offset + gid))
                spike_train = get_inhom_poisson_spike_times_by_thinning(rate_map, t, dt=0.025,
                                                                        generator=local_random)
                if debug and rank == 0:
                    callback, context = debug
                    this_context = Struct(**dict(context()))
                    this_context.update(dict(locals()))
                    callback(this_context)

                spikes_attr_dict[gid] = dict()
                spikes_attr_dict[gid]['Selectivity Type'] = np.array([this_selectivity_type], dtype='uint8')
                spikes_attr_dict[gid]['Trajectory Rate Map'] = np.asarray(rate_map, dtype='float32')
                spikes_attr_dict[gid][spike_train_attr_name] = np.asarray(spike_train, dtype='float32')
                gid_count += 1
                spike_hist_edges = np.linspace(min(t), max(t), spike_hist_resolution + 1)
                hist, edges = np.histogram(spike_train, bins=spike_hist_edges)
                spike_hist_sum[this_selectivity_type_name] = \
                      np.add(spike_hist_sum[this_selectivity_type_name], hist)

            if (iter_count > 0 and iter_count % write_every == 0) or (debug and iter_count == 10):

                total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
                if rank == 0:
                    logger.info('generated spike trains for %i %s %s cells' %
                                (total_gid_count, population, selectivity_type_name))
                    
                if not dry_run:
                    append_cell_attributes(output_path, population, spikes_attr_dict,
                                            namespace=output_namespace, comm=comm, io_size=io_size,
                                            chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                    del spikes_attr_dict
                    spikes_attr_dict = dict()

            if debug and iter_count == 10:
                break
            
        if not dry_run:
            append_cell_attributes(output_path, population, spikes_attr_dict,
                                    namespace=output_namespace, comm=comm, io_size=io_size,
                                    chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            del spikes_attr_dict
            spikes_attr_dict = dict()
        process_time = time.time() - start_time
            
        total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
        if rank == 0:
            logger.info('generated spike trains for %i %s %s cells in %.2f s' %
                        (total_gid_count, population,
                         selectivity_type_name, process_time))

    return spike_hist_sum


def remap_input_selectivity_features(env, arena, population, selectivity_path, selectivity_type_names, selectivity_type_namespaces, output_path, comm=None, io_size=-1, cache_size=10, write_every=1, chunk_size=1000, value_chunk_size=1000, dry_run=False, debug=False):
    """
    Remap input selectivity features.

    :param env:
    :param population: str
    :param selectivity_path: str (path to file)
    :param selectivity_type_names: 
    :param selectivity_type_namespaces: 
    :param output_path: str (path to file)
    :param comm: 
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param write_every: int
    :param debug: bool
    :param dry_run: bool
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    if io_size <= 0:
        io_size = comm.size
    rank = comm.rank
    
    local_random = np.random.RandomState()
    remap_offset = int(env.model_config['Random Seeds']['Input Remap'])

    num_modules = env.stimulus_config['Number Modules']
    grid_orientation_offset = [local_random.uniform(0., np.pi / 3.) for i in range(num_modules)]

    arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena)

    for selectivity_type_namespace in sorted(selectivity_type_namespaces[population]):
        selectivity_type = None
        selectivity_type_name = None
        gid_count = 0
        if rank == 0:
            logger.info('Remapping input selectivity features for population %s [%s]...' % (population, selectivity_type_namespace))
            
        process_time = 0
        start_time = time.time()
        selectivity_attr_gen = NeuroH5CellAttrGen(selectivity_path, population,
                                                  namespace=selectivity_type_namespace,
                                                  comm=comm, io_size=io_size,
                                                  cache_size=cache_size)
        remap_attr_dict = {}
        for iter_count, (gid, selectivity_attr_dict) in enumerate(selectivity_attr_gen):
            if gid is not None:
                local_random.seed(int(remap_offset + gid))
                this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
                this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
                if selectivity_type is None:
                    selectivity_type = this_selectivity_type
                    selectivity_type_name = this_selectivity_type_name
                assert(this_selectivity_type_name == selectivity_type_name)
                input_cell_config = \
                  get_input_cell_config(selectivity_type=this_selectivity_type,
                                        selectivity_type_names=selectivity_type_names,
                                        selectivity_attr_dict=selectivity_attr_dict)
                if this_selectivity_type_name == 'grid':
                    module_id = input_cell_config.module_id
                    input_cell_config.grid_orientation += grid_orientation_offset[module_id]
                    input_cell_config.x0 = local_random.uniform(*arena_x_bounds)
                    input_cell_config.y0 = local_random.uniform(*arena_y_bounds)
                elif this_selectivity_type_name == 'place':
                    input_cell_config.x0 = np.asarray([local_random.uniform(*arena_x_bounds)
                                                        for x0 in input_cell_config.x0],
                                                       dtype=np.float32)
                    input_cell_config.y0 = np.asarray([local_random.uniform(*arena_y_bounds)
                                                        for y0 in input_cell_config.y0],
                                                       dtype=np.float32)

                remap_attr_dict[gid] = input_cell_config.get_selectivity_attr_dict()
                
            if (iter_count > 0 and iter_count % write_every == 0) or (debug and iter_count == 10):
                
                gid_count += len(remap_attr_dict)
                total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
                if rank == 0:
                   logger.info('remapped input features for %i %s %s cells' %
                                   (total_gid_count, population, selectivity_type_name))
                   
                if not dry_run:
                    append_cell_attributes(output_path, population, remap_attr_dict,
                                           namespace=selectivity_type_namespace, comm=comm,
                                           io_size=io_size,
                                           chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                    del remap_attr_dict
                    remap_attr_dict = {}

            if debug and iter_count == 10:
                break
            
        gid_count += len(remap_attr_dict)
        total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
        if not dry_run:
            append_cell_attributes(output_path, population, remap_attr_dict,
                                   namespace=selectivity_type_namespace, comm=comm, io_size=io_size,
                                   chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            del remap_attr_dict
            remap_attr_dict = {}
        process_time = time.time() - start_time
            
        if rank == 0:
                logger.info('remapped input features for %i %s %s cells' %
                            (total_gid_count, population, selectivity_type_name))





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
    ratemap_lst = []
    module_gid_set = set([])
    if module is not None:
        if not isinstance(module, int):
            raise Exception('module variable must be an integer')
        gid_module_gen = read_cell_attributes(stimulus_path, population, namespace='Cell Attributes')
        for (gid, attr_dict) in gid_module_gen:
            this_module = attr_dict['Module'][0]
            if this_module == module:
                module_gid_set.add(gid)

    attr_gen = read_cell_attributes(stimulus_path, population, namespace=stimulus_namespace)
    for gid, stimulus_dict in attr_gen:
        if gid in module_gid_set or len(module_gid_set) == 0:
            rate = stimulus_dict['Trajectory Rate Map']
            spiketrain = stimulus_dict['Spike Train']
            peak_index = np.where(rate == np.max(rate))[0][0]
            ratemap_lst.append((gid, rate, spiketrain, peak_index))

    ## sort by peak_index
    ratemap_lst.sort(key=lambda item: item[3])
    return ratemap_lst


def read_feature(feature_path, feature_namespace, population):
    feature_lst = []

    attr_gen = read_cell_attributes(feature_path, population, namespace=feature_namespace)
    for gid, feature_dict in attr_gen:
        if 'Module ID' in feature_dict:
            gid_module = feature_dict['Module ID'][0]
        else:
            gid_module = None
        rate = feature_dict['Arena Rate Map']
        feature_lst.append((gid, rate, gid_module))

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
    nbins = edges.shape[0] - 1
    nfeatures = features.shape[1]
    binned_features = np.empty([nbins, nfeatures])
    for i in range(nbins):
        for j in range(nfeatures):
            delta = edges[i + 1] - edges[i]
            bin_range = np.arange(edges[i], edges[i + 1], delta / 5.)
            ip_vals = np.interp(bin_range, t, features[:, j])
            binned_features[i, j] = np.mean(ip_vals)

    return binned_features


##
## Linearize position
##
def linearize_trajectory(x, y):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)

    T = np.concatenate((x, y))
    T_transform = pca.fit_transform(T)
    T_linear = pca.inverse_transform(T_transform)

    return T_linear
