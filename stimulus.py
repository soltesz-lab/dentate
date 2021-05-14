import os, sys, gc, copy, time
import numpy as np
from scipy.interpolate import Rbf
from collections import defaultdict, ChainMap
from mpi4py import MPI
from dentate.utils import get_module_logger, object, range, str, Struct, gauss2d, gaussian, viewitems
from dentate.stgen import get_inhom_poisson_spike_times_by_thinning
from neuroh5.io import read_cell_attributes, append_cell_attributes, NeuroH5CellAttrGen, scatter_read_cell_attribute_selection
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
        self.place_module_field_width_sigma = stimulus_config['Modular Place Field Width Variance'] / 6

        
        self.non_modular_place_field_width_sigma = stimulus_config['Non-modular Place Field Width Variance'] / 6.
        

    def get_module_probabilities(self, distance):
        p_modules = []
        for offset in self.module_probability_offsets:
            p_modules.append(self.get_module_probability(distance, offset))
        p_modules = np.array(p_modules, dtype=np.float32)
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
                 peak_rate=None, distance=None, local_random=None, selectivity_attr_dict=None,
                 phase_mod_function=None):
        """
        :param selectivity_type: int
        :param arena: namedtuple
        :param selectivity_config: :class:'InputSelectivityConfig'
        :param peak_rate: float
        :param distance: float; u arc distance normalized to reference layer
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """
        self.phase_mod_function = phase_mod_function
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
        return {'Selectivity Type': np.array([self.selectivity_type], dtype=np.uint8),
                'Peak Rate': np.array([self.peak_rate], dtype=np.float32),
                'Module ID': np.array([self.module_id], dtype=np.uint8),
                'Grid Spacing': np.array([self.grid_spacing], dtype=np.float32),
                'Grid Orientation': np.array([self.grid_orientation], dtype=np.float32),
                'X Offset': np.array([self.x0], dtype=np.float32),
                'Y Offset': np.array([self.y0], dtype=np.float32),
                'Field Width Concentration Factor':
                    np.array([self.grid_field_width_concentration_factor], dtype=np.float32)
                }

    def get_rate_map(self, x, y, scale=1.0, phi=None):
        """

        :param x: array
        :param y: array
        :return: array
        """
        if (phi is not None) and (self.phase_mod_function is None):
            raise RuntimeError("GridInputCellConfig.get_rate_map: when phase phi is provided, cell must have phase_mod_function configured")
        
        rate_array =  np.multiply(get_grid_rate_map(self.x0, self.y0, scale * self.grid_spacing, self.grid_orientation, x, y,
                                                    a=self.grid_field_width_concentration_factor), self.peak_rate)
        mean_rate = np.mean(rate_array)
        if phi is not None:
            rate_array *= self.phase_mod_function(phi)
            mean_rate_mod = np.mean(rate_array)
            rate_array *= mean_rate / mean_rate_mod
        return rate_array


class PlaceInputCellConfig(object):
    def __init__(self, selectivity_type=None, arena=None, normalize_scale=True, selectivity_config=None,
                 peak_rate=None, distance=None, modular=None, num_place_field_probabilities=None, field_width=None,
                 local_random=None, selectivity_attr_dict=None, phase_mod_function=None):
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
        self.phase_mod_function = phase_mod_function

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
        return {'Selectivity Type': np.array([self.selectivity_type], dtype=np.uint8),
                'Peak Rate': np.array([self.peak_rate], dtype=np.float32),
                'Module ID': np.array([self.module_id], dtype=np.int8),
                'Num Fields': np.array([self.num_fields], dtype=np.uint8),
                'Field Width': np.asarray(self.field_width, dtype=np.float32),
                'X Offset': np.asarray(self.x0, dtype=np.float32),
                'Y Offset': np.asarray(self.y0, dtype=np.float32)
                }

    def get_rate_map(self, x, y, scale=1.0, phi=None):
        """

        :param x: array
        :param y: array
        :return: array
        """
        
        if (phi is not None) and (self.phase_mod_function is None):
            raise RuntimeError("PlaceInputCellConfig.get_rate_map: when phase phi is provided, cell must have phase_mod_function configured")
        
        rate_array = np.zeros_like(x, dtype=np.float32)
        for i in range(self.num_fields):
            rate_array = np.maximum(rate_array, get_place_rate_map(self.x0[i], self.y0[i], self.field_width[i] * scale, x, y))
        rate_array *= self.peak_rate
        mean_rate = np.mean(rate_array)
        
        if phi is not None:
            rate_array *= self.phase_mod_function(phi)
            mean_rate_mod = np.mean(rate_array)
            rate_array *= mean_rate / mean_rate_mod

        return rate_array


    
class ConstantInputCellConfig(object):
    def __init__(self, selectivity_type=None, arena=None, selectivity_config=None,
                 peak_rate=None, local_random=None, selectivity_attr_dict=None, phase_mod_function=None):
        """
        :param selectivity_type: int
        :param arena: namedtuple
        :param selectivity_config: :class:'SelectivityModuleConfig'
        :param peak_rate: float
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """
        self.phase_mod_function = phase_mod_function
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
        return {'Selectivity Type': np.array([self.selectivity_type], dtype=np.uint8),
                'Peak Rate': np.array([self.peak_rate], dtype=np.float32),
                }

    def get_rate_map(self, x, y, phi=None):
        """

        :param x: array
        :param y: array
        :return: array
        """

        if (phi is not None) and (self.phase_mod_function is None):
            raise RuntimeError("ConstantInputCellConfig.get_rate_map: when phase phi is provided, cell must have phase_mod_function configured")
        
        rate_array = np.ones_like(x, dtype=np.float32) * self.peak_rate
        mean_rate = np.mean(rate_array)
        if phi is not None:
            rate_array *= self.phase_mod_function(phi)
            mean_rate_mod = np.mean(rate_array)
            rate_array *= mean_rate / mean_rate_mod

        return rate_array

    
    
def get_place_rate_map(x0, y0, width, x, y):
    """

    :param x0: float
    :param y0: float
    :param width: float
    :param x: array
    :param y: array
    :return: array
    """

    fw = 2. * np.sqrt(2. * np.log(100.))
    return gauss2d(x=x, y=y, mx=x0, my=y0, sx=width / fw, sy=width / fw)
           


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
                          selectivity_attr_dict=None, phase_mod_function=None):
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
    :param phase_mod_function: function; oscillatory phase modulation
    :return: instance of one of various InputCell classes
    """
    if selectivity_type not in selectivity_type_names:
        raise RuntimeError('get_input_cell_config: enumerated selectivity type: %i not recognized' % selectivity_type)
    selectivity_type_name = selectivity_type_names[selectivity_type]

    if selectivity_attr_dict is not None:
        if selectivity_type_name == 'grid':
            input_cell_config = GridInputCellConfig(selectivity_attr_dict=selectivity_attr_dict,
                                                    phase_mod_function=phase_mod_function)
        elif selectivity_type_name == 'place':
            input_cell_config = PlaceInputCellConfig(selectivity_attr_dict=selectivity_attr_dict,
                                                     phase_mod_function=phase_mod_function)
        elif selectivity_type_name == 'constant':
            input_cell_config = ConstantInputCellConfig(selectivity_attr_dict=selectivity_attr_dict,
                                                        phase_mod_function=phase_mod_function)
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
                                    local_random=local_random, phase_mod_function=phase_mod_function)
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
                                     local_random=local_random, phase_mod_function=phase_mod_function)
        elif selectivity_type_name == 'constant':
            input_cell_config = ConstantInputCellConfig(selectivity_type=selectivity_type, arena=arena,
                                                        selectivity_config=selectivity_config, peak_rate=peak_rate,
                                                        phase_mod_function=phase_mod_function)
        else:
            RuntimeError('get_input_cell_config: selectivity type: %s not implemented' % selectivity_type_name)

    return input_cell_config


def get_equilibration(env):
    if 'Equilibration Duration' in env.stimulus_config and env.stimulus_config['Equilibration Duration'] > 0.:
        equilibrate_len = int(env.stimulus_config['Equilibration Duration'] /
                              env.stimulus_config['Temporal Resolution'])
        from scipy.signal import hann
        equilibrate_hann = hann(2 * equilibrate_len)[:equilibrate_len]
        equilibrate = (equilibrate_hann, equilibrate_len)
    else:
        equilibrate = None

    return equilibrate


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
    active_cell_matrix = np.zeros_like(pop_activity, dtype=np.float32)
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
    vertices_x = np.asarray([v[0] for v in arena.domain.vertices], dtype=np.float32)
    vertices_y = np.asarray([v[1] for v in arena.domain.vertices], dtype=np.float32)
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



def generate_input_selectivity_features(env, population, arena, arena_x, arena_y,
                                        gid, norm_distances, 
                                        selectivity_config, selectivity_type_names,
                                        selectivity_type_namespaces,
                                        rate_map_sum=None, debug=False):
    """
    Generates input selectivity features for the given population and
    returns the selectivity type-specific dictionary provided through
    argument selectivity_type_namespaces.  The set of selectivity
    attributes is determined by procedure get_selectivity_attr_dict in
    the respective input cell configuration class
    (e.g. PlaceInputCellConfig or GridInputCellConfig).

    :param env
    :param population: str
    :param arena: str
    :param gid: int
    :param distances: (float, float)
    :param selectivity_config: 
    :param selectivity_type_names: 
    :param selectivity_type_namespaces: 
    :param debug: bool
    """

    if env.comm is not None:
        rank = env.comm.rank
    else:
        rank = 0
    
    norm_u_arc_distance = norm_distances[0]
    selectivity_seed_offset = int(env.model_config['Random Seeds']['Input Selectivity'])

    local_random = np.random.RandomState()
    local_random.seed(int(selectivity_seed_offset + gid))
    this_selectivity_type = \
     choose_input_selectivity_type(p=env.stimulus_config['Selectivity Type Probabilities'][population],
                                   local_random=local_random)
    
    input_cell_config = get_input_cell_config(selectivity_type=this_selectivity_type,
                                              selectivity_type_names=selectivity_type_names,
                                              stimulus_config=env.stimulus_config,
                                              arena=arena,
                                              selectivity_config=selectivity_config,
                                              distance=norm_u_arc_distance,
                                              local_random=local_random)
    
    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
    selectivity_attr_dict = input_cell_config.get_selectivity_attr_dict()
    rate_map = input_cell_config.get_rate_map(x=arena_x, y=arena_y)

    if debug and rank == 0:
        callback, context = debug
        this_context = Struct(**dict(context()))
        this_context.update(dict(locals()))
        callback(this_context)
        
    if rate_map_sum is not None:
        rate_map_sum[this_selectivity_type_name] = \
         np.add(rate_map_sum[this_selectivity_type_name], rate_map)
    
    return this_selectivity_type_name, selectivity_attr_dict


def generate_input_spike_trains(env, population, selectivity_type_names, trajectory, gid, selectivity_attr_dict, spike_train_attr_name='Spike Train',
                                selectivity_type_name=None, spike_hist_resolution=1000, equilibrate=None, phase_mod_function=False, osc_phi=None,
                                spike_hist_sum=None, return_selectivity_features=True, n_trials=1, merge_trials=True, time_range=None,
                                comm=None, seed=None, debug=False):
    """
    Generates spike trains for the given gid according to the
    input selectivity rate maps contained in the given selectivity
    file, and returns a dictionary with spike trains attributes.

    :param env:
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.rank

    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    t, x, y, d = trajectory

    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])

    local_random = np.random.RandomState()
    input_spike_train_seed = int(env.model_config['Random Seeds']['Input Spiketrains'])

    if seed is None:
        local_random.seed(int(input_spike_train_seed + gid))
    else:
        local_random.seed(int(seed))

    this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
    if selectivity_type_name is None:
        selectivity_type_name = this_selectivity_type_name

    input_cell_config = get_input_cell_config(selectivity_type=this_selectivity_type,
                                              selectivity_type_names=selectivity_type_names,
                                              selectivity_attr_dict=selectivity_attr_dict,
                                              phase_mod_function=phase_mod_function)
    rate_map = input_cell_config.get_rate_map(x=x, y=y, phi=osc_phi if phase_mod_function else None)
    if (selectivity_type_name != 'constant') and (equilibrate is not None):
        equilibrate_filter, equilibrate_len = equilibrate
        rate_map[:equilibrate_len] = np.multiply(rate_map[:equilibrate_len], equilibrate_filter)

    trial_duration = np.max(t) - np.min(t)
    if time_range is not None:
        trial_duration = max(trial_duration, (time_range[1] - time_range[0]) + equilibration_duration )
        
    spike_trains = []
    trial_indices = []
    for i in range(n_trials):
        spike_train = np.asarray(get_inhom_poisson_spike_times_by_thinning(rate_map, t, dt=temporal_resolution,
                                                                           generator=local_random),
                                 dtype=np.float32)
        if merge_trials:
            spike_train += float(i)*trial_duration
        spike_trains.append(spike_train)
        trial_indices.append(np.ones((spike_train.shape[0],), dtype=np.uint8) * i)

    if debug and rank == 0:
        callback, context = debug
        this_context = Struct(**dict(context()))
        this_context.update(dict(locals()))
        callback(this_context)

    spikes_attr_dict = dict()
    if merge_trials:
        spikes_attr_dict[spike_train_attr_name] = np.asarray(np.concatenate(spike_trains), dtype=np.float32)
        spikes_attr_dict['Trial Index'] = np.asarray(np.concatenate(trial_indices), dtype=np.uint8)
    else:
        spikes_attr_dict[spike_train_attr_name] = spike_trains
        spikes_attr_dict['Trial Index'] = trial_indices
        
    spikes_attr_dict['Trial Duration'] = np.asarray([trial_duration]*n_trials, dtype=np.float32)
    
    if return_selectivity_features:
        spikes_attr_dict['Selectivity Type'] = np.array([this_selectivity_type], dtype=np.uint8)
        spikes_attr_dict['Trajectory Rate Map'] = np.asarray(rate_map, dtype=np.float32)

    if spike_hist_sum is not None:
        spike_hist_edges = np.linspace(min(t), max(t), spike_hist_resolution + 1)
        hist, edges = np.histogram(spike_train, bins=spike_hist_edges)
        spike_hist_sum[this_selectivity_type_name] = np.add(spike_hist_sum[this_selectivity_type_name], hist)

    return spikes_attr_dict


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
    remap_seed = int(env.model_config['Random Seeds']['Input Remap'])

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
                local_random.seed(int(remap_seed + gid))
                this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
                this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
                if selectivity_type is None:
                    selectivity_type = this_selectivity_type
                    selectivity_type_name = this_selectivity_type_name
                assert(this_selectivity_type_name == selectivity_type_name)
                input_cell_config = \
                  get_input_cell_config(population=population,
                                        selectivity_type=this_selectivity_type,
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




def rate_maps_from_features (env, population, cell_index_set, input_features_path=None, input_features_namespace=None, 
                             input_features_dict=None, arena_id=None, trajectory_id=None, time_range=None,
                             include_time=False, phase_mod=False, distances_dict=None):
    
    """Initializes presynaptic spike sources from a file with input selectivity features represented as firing rates."""

    if input_features_dict is not None:
        if (input_features_path is not None) or  (input_features_namespace is not None):
            raise RuntimeError("rate_maps_from_features: when input_features_dict is provided, input_features_path must be None")
    else:
        if (input_features_path is None) or  (input_features_namespace is None):
            raise RuntimeError("rate_maps_from_features: either input_features_dict has to be provided, or input_features_path and input_features_namespace")
    
    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    if arena_id is None:
        arena_id = env.arena_id
    if trajectory_id is None:
        trajectory_id = env.trajectory_id

    spatial_resolution = float(env.stimulus_config['Spatial Resolution'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])

    if phase_mod and (distances_dict is None):
        raise RuntimeError("rate_maps_from_features: when phase_mod is True, distances_dict must be provided")
    
    soma_positions = None
    if distances_dict is not None:
        soma_positions = {}
        for k in distances_dict:
            soma_positions[k] = distances_dict[k][0]
    
    
    input_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                 'Module ID', 'Grid Spacing', 'Grid Orientation',
                                 'Field Width Concentration Factor', 
                                 'X Offset', 'Y Offset']
    
    selectivity_type_names = { i: n for n, i in viewitems(env.selectivity_types) }

    arena = env.stimulus_config['Arena'][arena_id]
    
    trajectory = arena.trajectories[trajectory_id]
    equilibration_duration = float(env.stimulus_config.get('Equilibration Duration', 0.))

    t, x, y, d = generate_linear_trajectory(trajectory, temporal_resolution=temporal_resolution,
                                            equilibration_duration=equilibration_duration)
    if time_range is not None:
        t_range_inds = np.where((t < time_range[1]) & (t >= time_range[0]))[0] 
        t = t[t_range_inds]
        x = x[t_range_inds]
        y = y[t_range_inds]
        d = d[t_range_inds]

    osc_t, osc_y, osc_phi = global_oscillation_signal(env, t)
    population_phase_prefs = global_oscillation_phase_pref(env, population, num_cells=len(cell_index_set))
    population_phase_shifts = None
    if soma_positions is not None:
        position_array = np.asarray([ soma_positions[k] for k in cell_index_set ])
        population_phase_shifts = global_oscillation_phase_shift(env, position_array)
    population_phase_dict = None
    if population_phase_shifts is not None:
        population_phase_dict = {}
        for i, gid in enumerate(cell_index_set):
            population_phase_dict[gid] = (population_phase_prefs[i], population_phase_shifts[i])
        
    input_rate_map_dict = {}
    pop_index = int(env.Populations[population])

    if input_features_path is not None:
        this_input_features_namespace = '%s %s' % (input_features_namespace, arena_id)
        input_features_iter = scatter_read_cell_attribute_selection(input_features_path, population,
                                                                    selection=cell_index_set,
                                                                    namespace=this_input_features_namespace,
                                                                    mask=set(input_features_attr_names), 
                                                                    comm=env.comm, io_size=env.io_size)
    else:
        input_features_iter = viewitems(input_features_dict)
        
    for gid, selectivity_attr_dict in input_features_iter:

        this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]

        phase_mod_function = None
        if phase_mod:
            this_phase_shift, this_phase_pref = population_phase_dict[gid]
            x, d = global_oscillation_phase_mod(env, population, this_phase_pref)
            phase_mod_ip = Rbf(x, d, function="gaussian")
            phase_mod_function=lambda phi: phase_mod_ip(np.mod(phi + this_phase_shift, 360.))
        input_cell_config = get_input_cell_config(selectivity_type=this_selectivity_type,
                                                  selectivity_type_names=selectivity_type_names,
                                                  selectivity_attr_dict=selectivity_attr_dict,
                                                  phase_mod_function=phase_mod_function)
        rate_maps = []
        rate_map = input_cell_config.get_rate_map(x=x, y=y, phi=osc_phi if phase_mod else None)
        rate_map[np.isclose(rate_map, 0., atol=1e-3, rtol=1e-3)] = 0.

        if include_time:
            input_rate_map_dict[gid] = (t, rate_map)
        else:
            input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict


def arena_rate_maps_from_features (env, population, input_features_path, input_features_namespace, cell_index_set,
                                   arena_id=None, time_range=None, n_trials=1):
    
    """Initializes presynaptic spike sources from a file with input selectivity features represented as firing rates."""
        
    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    if arena_id is None:
        arena_id = env.arena_id

    spatial_resolution = float(env.stimulus_config['Spatial Resolution'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    
    this_input_features_namespace = '%s %s' % (input_features_namespace, arena_id)
    
    input_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                 'Module ID', 'Grid Spacing', 'Grid Orientation',
                                 'Field Width Concentration Factor', 
                                 'X Offset', 'Y Offset']
    
    selectivity_type_names = { i: n for n, i in viewitems(env.selectivity_types) }

    arena = env.stimulus_config['Arena'][arena_id]
    arena_x, arena_y = get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=spatial_resolution)
    
    input_rate_map_dict = {}
    pop_index = int(env.Populations[population])

    input_features_iter = scatter_read_cell_attribute_selection(input_features_path, population,
                                                                selection=cell_index_set,
                                                                namespace=this_input_features_namespace,
                                                                mask=set(input_features_attr_names), 
                                                                comm=env.comm, io_size=env.io_size)
    for gid, selectivity_attr_dict in input_features_iter:

        this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
        input_cell_config = get_input_cell_config(population=population,
                                                  selectivity_type=this_selectivity_type,
                                                  selectivity_type_names=selectivity_type_names,
                                                  selectivity_attr_dict=selectivity_attr_dict)
        if input_cell_config.num_fields > 0:
            rate_map = input_cell_config.get_rate_map(x=arena_x, y=arena_y)
            rate_map[np.isclose(rate_map, 0., atol=1e-3, rtol=1e-3)] = 0.
            input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict


def global_oscillation_signal(env, t):
    """
    Generates an oscillatory signal and phase for phase modulation of input spike trains.
    Uses the "Global Oscillation" entry in the input configuration.
    The configuration format is:

      frequency: <float> # oscillation frequency
      Phase Distribution: # parameters of phase distribution along septotemporal axis
       slope: <float>
       offset: <float>
      Phase Modulation: # cell type-specific modulation
        <cell type>:
         phase: [<start float>, <end float>] # range of preferred phases
         depth: <float> # depth of modulation

    Returns: time, signal value, phase 0-360 degrees
    """
    
    from scipy.signal import hilbert

    global_oscillation_config = env.stimulus_config.get('Global Oscillation', None)
    if global_oscillation_config is None:
        return None, None, None
    F = global_oscillation_config['frequency']
    y = np.cos(2*np.pi*F*(t/1000.))
    yn = hilbert(y)
    phi = np.angle(yn)
    phiz = np.argwhere(np.isclose(phi, 0.0))
    phi[phiz] = 0.
    idxs = np.argwhere(np.isclose(phi, 0.0, rtol=4e-5, atol=4e-5)).flat
    phi_lst = np.split(phi, idxs)
    phi_udeg = np.concatenate([ np.rad2deg(np.unwrap(elem)) for elem in phi_lst ])

    return t, y, phi_udeg


def global_oscillation_phase_shift(env, position):
    """
    Computes the phase shift of the global oscillatory signal for the given position, assumed to be on the long axis. 
    Uses the "Global Oscillation" entry in the input configuration. See `global_oscillation_signal` for a description of the configuration format.
    """

    global_oscillation_config = env.stimulus_config['Global Oscillation']
    phase_dist_config = global_oscillation_config['Phase Distribution']
    phase_slope = phase_dist_config['slope']
    phase_offset = phase_dist_config['offset']
    x = position / 1000.

    return x*phase_slope + phase_offset


def global_oscillation_phase_pref(env, population, num_cells, local_random=None):
    """
    Computes oscillatory phase preferences for all cells in the given population.
    Uses the "Global Oscillation" entry in the input configuration. See `global_oscillation_signal` for a description of the configuration format.

    Returns: an array of phase preferences of length equal to the population size.
    """

    seed = int(env.model_config['Random Seeds']['Phase Preference'])

    if local_random is None:
        local_random = np.random.RandomState(seed)
    
    global_oscillation_config = env.stimulus_config['Global Oscillation']
    phase_mod_config = global_oscillation_config['Phase Modulation'][population]
    phase_range = phase_mod_config['phase']
    phase_loc = (phase_range[1] - phase_range[0]) / 2.
    fw = 2. * np.sqrt(2. * np.log(100.))
    phase_scale = (phase_range[1] - phase_range[0]) / fw
    s = local_random.normal(loc=phase_loc, scale=phase_scale, size=num_cells) + phase_range[0]
    s = np.clip(s, phase_range[0], phase_range[1])
    
    return s


def global_oscillation_phase_mod(env, population, phase_pref, bin_size=1):
    """
    Computes oscillatory phase preferences for all cells in the given population.
    Uses the "Global Oscillation" entry in the input configuration. See `global_oscillation_signal` for a description of the configuration format.

    Returns: a tuple of arrays x, d where x contains phases 0-360 degrees, and d contains the corresponding modulation [0 - 1].
    """

    global_oscillation_config = env.stimulus_config['Global Oscillation']
    phase_mod_config = global_oscillation_config['Phase Modulation'][population]
    phase_range = phase_mod_config['phase']
    mod_depth = phase_mod_config['depth']
    fw = 2. * np.sqrt(2. * np.log(100.))
    phase_sig = (phase_range[1] - phase_range[0]) / fw

    x = np.arange(0, 360, bin_size)
    d = np.ones_like(x) * (1.0 - mod_depth)

    d += gaussian(x, mu=phase_pref, sig=phase_sig, A=mod_depth)

    return x, d
