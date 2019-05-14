import os
import numpy as np
import yaml
from collections import namedtuple, defaultdict
from neuroh5.io import read_projection_names, read_population_ranges, read_population_names, read_cell_attribute_info
from dentate.synapses import SynapseAttributes
from mpi4py import MPI
from neuron import h
from dentate.utils import *
from dentate.neuron_utils import find_template


SynapseConfig = namedtuple('SynapseConfig',
                               ['type',
                                'sections',
                                'layers',
                                'proportions',
                                'contacts',
                                'mechanisms'])

GapjunctionConfig = namedtuple('GapjunctionConfig',
                                 ['sections',
                                  'connection_probability',
                                  'connection_parameters',
                                  'connection_bounds',
                                  'coupling_coefficients',
                                  'coupling_parameters',
                                  'coupling_bounds'])

NetclampConfig = namedtuple('NetclampConfig',
                            ['template_params',
                             'input_generators',
                             'weight_generators',
                             'optimize_parameters'])

ArenaConfig  = namedtuple('Arena',
                          ['name',
                           'domain',
                           'trajectories',
                           'properties'])

DomainConfig = namedtuple('Domain',
                            ['vertices',
                             'simplices'])

TrajectoryConfig = namedtuple('Trajectory',
                              ['velocity',
                               'path'])

class Env:
    """
    Network model configuration.
    """
    def __init__(self, comm=None, config_file=None, template_paths="templates", hoc_lib_path=None, dataset_prefix=None,
                 config_prefix=None, results_path=None, results_id=None, node_rank_file=None, io_size=0,
                 vrecord_fraction=0, coredat=False, tstop=0, v_init=-65, stimulus_onset=0.0, max_walltime_hours=0.5,
                 results_write_time=0, dt=0.025, ldbal=False, lptbal=False, transfer_debug=False,
                 cell_selection_path=None, spike_input_path=None, spike_input_namespace=None,
                 cleanup=True, cache_queries=False, profile_memory=False, verbose=False, **kwargs):
        """
        :param comm: :class:'MPI.COMM_WORLD'
        :param config_file: str; model configuration file name
        :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
        :param hoc_lib_path: str; path to directory containing required hoc libraries
        :param dataset_prefix: str; path to directory containing required neuroh5 data files
        :param config_prefix: str; path to directory containing network and cell mechanism config files
        :param results_path: str; path to directory to export output files
        :param results_id: str; label for neuroh5 namespaces to write spike and voltage trace data
        :param node_rank_file: str; name of file specifying assignment of node gids to MPI ranks
        :param io_size: int; the number of MPI ranks to be used for I/O operations
        :param vrecord_fraction: float; fraction of cells to record intracellular voltage from
        :param coredat: bool; Save CoreNEURON data
        :param tstop: int; physical time to simulate (ms)
        :param v_init: float; initialization membrane potential (mV)
        :param stimulus_onset: float; starting time of stimulus (ms)
        :param max_walltime_hours: float; maximum wall time (hours)
        :param results_write_time: float; time to write out results at end of simulation
        :param dt: float; simulation time step
        :param ldbal: bool; estimate load balance based on cell complexity
        :param lptbal: bool; calculate load balance with LPT algorithm
        :param cleanup: bool; clean up auxiliary cell and synapse structures after network init
        :param profile: bool; profile memory usage
        :param cache_queries: bool; whether to use a cache to speed up queries to filter_synapses
        :param verbose: bool; print verbose diagnostic messages while constructing the network
        """
        self.SWC_Types = {}
        self.Synapse_Types = {}
        self.layers = {}
        self.globals = {}
        
        self.gidset = set([])
        self.cells = []
        self.gjlist = []
        self.biophys_cells = defaultdict(dict)
        self.v_sample_dict = {}

        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        rank = self.comm.Get_rank()

        if self.comm is not None:
            self.pc = h.ParallelContext()
        else:
            self.pc = None

        # If true, the biophysical cells and synapses dictionary will be freed
        # as synapses and connections are instantiated.
        self.cleanup = cleanup
        
        # If true, compute and print memory usage at various points
        # during simulation initialization
        self.profile_memory = profile_memory
            
        # print verbose diagnostic messages
        self.verbose = verbose
        config_logging(verbose)
        self.logger = get_root_logger()
        
        # Directories for cell templates
        if template_paths is not None:
            self.template_paths = template_paths.split(':')
        else:
            self.template_paths = []

        # The location of required hoc libraries
        self.hoc_lib_path = hoc_lib_path

        # The location of all datasets
        self.dataset_prefix = dataset_prefix

        # The path where results files should be written
        self.results_path = results_path

        # Identifier used to construct results data namespaces
        self.results_id = results_id

        # Number of MPI ranks to be used for I/O operations
        self.io_size = io_size

        # Initialization voltage
        self.v_init = v_init

        # simulation time [ms]
        self.tstop = tstop

        # stimulus onset time [ms]
        self.stimulus_onset = stimulus_onset

        # maximum wall time in hours
        self.max_walltime_hours = max_walltime_hours

        # time to write out results at end of simulation
        self.results_write_time = results_write_time

        # time step
        self.dt = dt

        # used to estimate cell complexity
        self.cxvec = None

        # measure/perform load balancing
        self.optldbal = ldbal
        self.optlptbal = lptbal

        self.transfer_debug = transfer_debug

        # Fraction of cells to record intracellular voltage from
        self.vrecord_fraction = vrecord_fraction

        # Save CoreNEURON data
        self.coredat = coredat

        # cache queries to filter_synapses
        self.cache_queries = cache_queries

        # Cell selection for simulations of subsets of the network
        self.cell_selection = None
        self.cell_selection_path = cell_selection_path
        if cell_selection_path is not None:
            with open(cell_selection_path) as fp:
                self.cell_selection = yaml.load(fp, IncludeLoader)
        
        # Spike input path
        self.spike_input_path = spike_input_path
        self.spike_input_ns = spike_input_namespace
        
        self.node_ranks = None
        if node_rank_file:
            with open(node_rank_file) as fp:
                dval = {}
                lines = fp.readlines()
                for l in lines:
                    a = l.split(' ')
                    dval[int(a[0])] = int(a[1])
                self.node_ranks = dval

        self.config_prefix = config_prefix

        if config_file is not None:
            if config_prefix is not None:
                config_file_path = self.config_prefix + '/' + config_file
            else:
                config_file_path = config_file
            if not os.path.isfile(config_file_path):
                raise RuntimeError("configuration file %s was not found" % config_file_path)
            with open(config_file_path) as fp:
                self.modelConfig = yaml.load(fp, IncludeLoader)
        else:
            raise RuntimeError("missing configuration file")

        if 'Definitions' in self.modelConfig:
            self.parse_definitions()

        if 'Global Parameters' in self.modelConfig:
            self.parse_globals()

        if 'Geometry' in self.modelConfig:
            self.geometry = self.modelConfig['Geometry']
        else:
            self.geometry = None

        if 'Origin' in self.geometry['Parametric Surface']:
            self.parse_origin_coords()

        self.celltypes = self.modelConfig['Cell Types']
        self.cellAttributeInfo = {}

        # The name of this model
        if 'Model Name' in self.modelConfig:
            self.modelName = self.modelConfig['Model Name']
        # The dataset to use for constructing the network
        if 'Dataset Name' in self.modelConfig:
            self.datasetName = self.modelConfig['Dataset Name']

        if results_path:
            if self.results_id is None:
                self.results_file_path = "%s/%s_results.h5" % (self.results_path, self.modelName)
            else:
                self.results_file_path = "%s/%s_%s_results.h5" % (self.results_path, self.modelName, self.results_id)
        else:
            self.results_file_path = "%s_%s_results.h5" % (self.modelName, self.results_id)

        if 'Connection Generator' in self.modelConfig:
            self.parse_connection_config()
            self.parse_gapjunction_config()

        if rank == 0:
            self.logger.info('dataset_prefix = %s' % str(self.dataset_prefix))

        if self.dataset_prefix is not None:
            self.dataset_path = os.path.join(self.dataset_prefix, self.datasetName)
            self.data_file_path = os.path.join(self.dataset_path, self.modelConfig['Cell Data'])
            self.load_celltypes()
            self.connectivity_file_path = os.path.join(self.dataset_path, self.modelConfig['Connection Data'])
            self.forest_file_path = os.path.join(self.dataset_path, self.modelConfig['Cell Data'])
            if 'Gap Junction Data' in self.modelConfig:
                self.gapjunctions_file_path = os.path.join(self.dataset_path, self.modelConfig['Gap Junction Data'])
            else:
                self.gapjunctions_file_path = None
        else:
            self.dataset_path = None
            self.data_file_path = None
            self.connectivity_file_path = None
            self.forest_file_path = None
            self.gapjunctions_file_path = None
                
        if 'Network Clamp' in self.modelConfig:
            self.parse_netclamp_config()

        if 'Input' in self.modelConfig:
            self.parse_input_config()                                                                                                                                                     
            
        self.projection_dict = defaultdict(list)
        if self.dataset_prefix is not None:
            for (src, dst) in read_projection_names(self.connectivity_file_path, comm=self.comm):
                self.projection_dict[dst].append(src)
        if rank == 0:
            self.logger.info('projection_dict = %s' % str(self.projection_dict))

        self.lfpConfig = {}
        if 'LFP' in self.modelConfig:
            for label, config in viewitems(self.modelConfig['LFP']):
                self.lfpConfig[label] = {'position': tuple(config['position']),
                                         'maxEDist': config['maxEDist'],
                                         'fraction': config['fraction'],
                                         'rho': config['rho'],
                                         'dt': config['dt']}

        self.t_vec = h.Vector()  # Spike time of all cells on this host
        self.id_vec = h.Vector()  # Ids of spike times on this host
        self.recs_dict = {}  # Intracellular samples on this host
        for pop_name, _ in viewitems(self.Populations):
            self.recs_dict[pop_name] = { 'Soma': [], 'Axon hillock': [], 'Apical dendrite': [], 'Basal dendrite': [] } 

        # used to calculate model construction times and run time
        self.mkcellstime = 0
        self.mkstimtime = 0
        self.connectcellstime = 0
        self.connectgjstime = 0

        self.simtime = None
        self.lfp = {}

        self.edge_count = defaultdict(dict)
        self.syns_set = defaultdict(set)

        # stimulus cell templates
        if len(self.template_paths) > 0:
            find_template(self, 'StimCell', path=self.template_paths)
            find_template(self, 'VecStimCell', path=self.template_paths)


    def parse_arena_domain(self, config):
        vertices = config['vertices']
        simplices = config['simplices']

        return DomainConfig(vertices, simplices)

    def parse_arena_trajectory(self, config):
        velocity = float(config['run velocity'])
        path_config = config['path']

        path_x = []
        path_y = []
        for v in path_config:
            path_x.append(v[0])
            path_y.append(v[1])

        path = np.column_stack((np.asarray(path_x, dtype=np.float32),
                                np.asarray(path_y, dtype=np.float32)))
        
        return TrajectoryConfig(velocity, path)
            
    def parse_input_config(self):
        features_type_dict = self.modelConfig['Definitions']['Input Features']
        input_dict = self.modelConfig['Input']
        input_config = {}

        for k,v in viewitems(input_dict):
            if k == 'Feature Distribution':
                feature_type_dict = {}
                for (pop,dvals) in viewitems(v):
                    pop_feature_type_dict = {}
                    for (feature_type_name,feature_type_fraction) in viewitems(dvals):
                        pop_feature_type_dict[int(self.feature_types[feature_type_name])] = float(feature_type_fraction)                                                                      
                    feature_type_dict[pop] = pop_feature_type_dict
                input_config['Feature Distribution'] = feature_type_dict
            elif k == 'Peak Rate':
                peak_rate_dict = {}
                for (pop,dvals) in viewitems(v):
                    pop_peak_rate_dict = {}
                    for (feature_type_name,peak_rate) in viewitems(dvals):
                        pop_peak_rate_dict[int(self.feature_types[feature_type_name])] = float(peak_rate)
                    peak_rate_dict[pop] = pop_peak_rate_dict
                input_config['Peak Rate'] = peak_rate_dict
            elif k == 'Arena':
                input_config['Arena'] = {}
                for arena_id, arena_val in viewitems(v):
                    arena_properties = {}
                    arena_domain = None
                    arena_trajectories = {}
                    for kk, vv in viewitems(arena_val):
                        if kk == 'Domain':
                            arena_domain = self.parse_arena_domain(vv)
                        elif kk == 'Trajectory':
                            for name, trajectory_config in viewitems(vv):
                                trajectory = self.parse_arena_trajectory(trajectory_config)
                                arena_trajectories[name] = trajectory
                        else:
                            arena_properties[kk] = vv
                    input_config['Arena'][arena_id] = ArenaConfig(arena_id, arena_domain,
                                                                  arena_trajectories, arena_properties)
            else:
                input_config[k] = v
                
        self.input_config = input_config
            
    def parse_netclamp_config(self):
        """

        :return:
        """
        netclamp_config_dict = self.modelConfig['Network Clamp']
        input_generator_dict = netclamp_config_dict['Input Generator']
        weight_generator_dict = netclamp_config_dict['Weight Generator']
        template_param_rules_dict = netclamp_config_dict['Template Parameter Rules']
        opt_param_rules_dict = netclamp_config_dict['Synaptic Optimization']
        
        template_params = {}
        for (template_name, params) in viewitems(template_param_rules_dict):
            template_params[template_name] = params

        self.netclamp_config = NetclampConfig(template_params, input_generator_dict, weight_generator_dict, opt_param_rules_dict)

    def parse_origin_coords(self):
        origin_spec = self.geometry['Parametric Surface']['Origin']

        coords = {}
        for key in ['U','V','L']:
            spec = origin_spec[key]
            if isinstance(spec, float):
                coords[key] = lambda x: spec
            elif spec == 'median':
                coords[key] = lambda x: np.median(x)
            elif spec == 'mean':
                coords[key] = lambda x: np.mean(x)
            elif spec == 'min':
                coords[key] = lambda x: np.min(x)
            elif spec == 'max':
                coords[key] = lambda x: np.max(x)
            else:
                raise ValueError
        self.geometry['Parametric Surface']['Origin'] = coords

    def parse_definitions(self):
        defs               = self.modelConfig['Definitions']
        self.Populations   = defs['Populations']
        self.SWC_Types     = defs['SWC Types']
        self.Synapse_Types = defs['Synapse Types']
        self.layers        = defs['Layers']
        self.feature_types = defs['Input Features']

    def parse_globals(self):
        self.globals       = self.modelConfig['Global Parameters']
        
    def parse_connection_config(self):
        """

        :return:
        """
        connection_config = self.modelConfig['Connection Generator']
        
        self.connection_velocity = connection_config['Connection Velocity']

        syn_mech_names  = connection_config['Synapse Mechanisms']
        syn_param_rules = connection_config['Synapse Parameter Rules']

        self.synapse_attributes = SynapseAttributes(self, syn_mech_names, syn_param_rules)

        extent_config = connection_config['Axon Extent']
        self.connection_extents = {}

        for population in extent_config.keys():

            pop_connection_extents = {}
            for layer_name in extent_config[population].keys():

                if layer_name == 'default':
                    pop_connection_extents[layer_name] = \
                      { 'width': extent_config[population][layer_name]['width'], \
                        'offset': extent_config[population][layer_name]['offset'] } 
                else:
                    layer_index = self.layers[layer_name]
                    pop_connection_extents[layer_index] = \
                      { 'width': extent_config[population][layer_name]['width'], \
                        'offset': extent_config[population][layer_name]['offset'] } 
            
            self.connection_extents[population] = pop_connection_extents

        
        synapse_config = connection_config['Synapses']
        connection_dict = {}
        
        for (key_postsyn, val_syntypes) in viewitems(synapse_config):
            connection_dict[key_postsyn] = {}
            
            for (key_presyn, syn_dict) in viewitems(val_syntypes):
                val_type         = syn_dict['type']
                val_synsections  = syn_dict['sections']
                val_synlayers    = syn_dict['layers']
                val_proportions  = syn_dict['proportions']
                if 'contacts' in syn_dict:
                    val_contacts     = syn_dict['contacts']
                else:
                    val_contacts = 1
                val_mechparams   = None
                val_swctype_mechparams    = None
                if 'mechanisms' in syn_dict:
                    val_mechparams   = syn_dict['mechanisms']
                else:
                    val_swctype_mechparams = syn_dict['swctype mechanisms']

                res_type = self.Synapse_Types[val_type]
                res_synsections = []
                res_synlayers = []
                res_mechparams = {}

                for name in val_synsections:
                    res_synsections.append(self.SWC_Types[name])
                for name in val_synlayers:
                    res_synlayers.append(self.layers[name])
                if val_swctype_mechparams is not None:
                    for swc_type in val_swctype_mechparams:
                        swc_type_index = self.SWC_Types[swc_type]
                        res_mechparams[swc_type_index] = val_swctype_mechparams[swc_type]
                else:
                    res_mechparams['default'] = val_mechparams
                        
                connection_dict[key_postsyn][key_presyn] = \
                    SynapseConfig(res_type, res_synsections, res_synlayers, val_proportions, val_contacts, \
                                  res_mechparams)

            config_dict = defaultdict(lambda: 0.0)
            for (key_presyn, conn_config) in viewitems(connection_dict[key_postsyn]):
                for (s,l,p) in zip(conn_config.sections, conn_config.layers, conn_config.proportions):
                    config_dict[(conn_config.type, s, l)] += p
                                              
            for (k,v) in viewitems(config_dict):
                try:
                    assert(np.isclose(v, 1.0))
                except Exception as e:
                    logger.error('Connection configuration: probabilities for %s do not sum to 1: %s = %f' %
                                 (key_postsyn, str(k), v))
                    raise e
                    
        self.connection_config = connection_dict

    def parse_gapjunction_config(self):
        """

        :return:
        """
        connection_config = self.modelConfig['Connection Generator']
        if 'Gap Junctions' in connection_config:
            gj_config = connection_config['Gap Junctions']

            gj_sections = gj_config['Locations']
            sections = {}
            for pop_a, pop_dict in viewitems(gj_sections):
                for pop_b, sec_names in viewitems(pop_dict):
                    pair = (pop_a, pop_b)
                    sec_idxs = []
                    for sec_name in sec_names:
                        sec_idxs.append(self.SWC_Types[sec_name])
                    sections[pair] = sec_idxs

            gj_connection_probs = gj_config['Connection Probabilities']
            connection_probs = {}
            for pop_a, pop_dict in viewitems(gj_connection_probs):
                for pop_b, prob in viewitems(pop_dict):
                    pair = (pop_a, pop_b)
                    connection_probs[pair] = float(prob)

            connection_weights_x = []
            connection_weights_y = []
            gj_connection_weights = gj_config['Connection Weights']
            for x in sorted(gj_connection_weights.keys()):
                connection_weights_x.append(x)
                connection_weights_y.append(gj_connection_weights[x])

            connection_params = np.polyfit(np.asarray(connection_weights_x), \
                                           np.asarray(connection_weights_y), \
                                           3)
            connection_bounds = [np.min(connection_weights_x), \
                                 np.max(connection_weights_x)]
            
            gj_coupling_coeffs = gj_config['Coupling Coefficients']
            coupling_coeffs = {}
            for pop_a, pop_dict in viewitems(gj_coupling_coeffs):
                for pop_b, coeff in viewitems(pop_dict):
                    pair = (pop_a, pop_b)
                    coupling_coeffs[pair] = float(coeff)

            gj_coupling_weights = gj_config['Coupling Weights']
            coupling_weights_x = []
            coupling_weights_y = []
            for x in sorted(gj_coupling_weights.keys()):
                coupling_weights_x.append(x)
                coupling_weights_y.append(gj_coupling_weights[x])

            coupling_params = np.polyfit(np.asarray(coupling_weights_x), \
                                         np.asarray(coupling_weights_y), \
                                         3)
            coupling_bounds = [np.min(coupling_weights_x), \
                               np.max(coupling_weights_x)]
            coupling_params = coupling_params
            coupling_bounds = coupling_bounds

            self.gapjunctions = {}
            for pair, sec_idxs in viewitems(sections):
                self.gapjunctions[pair] = GapjunctionConfig(sec_idxs, \
                                                            connection_probs[pair], \
                                                            connection_params, \
                                                            connection_bounds, \
                                                            coupling_coeffs[pair], \
                                                            coupling_params, \
                                                            coupling_bounds)
        else:
            self.gapjunctions = None
        
    def load_celltypes(self):
        """

        :return:
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        celltypes = self.celltypes
        typenames = sorted(celltypes.keys())

        if rank == 0:
            self.logger.info('env.data_file_path = %s' % str(self.data_file_path))
        self.comm.Barrier()

        (population_ranges, _) = read_population_ranges(self.data_file_path, self.comm)
        if rank == 0:
            self.logger.info('population_ranges = %s' % str(population_ranges))
        
        for k in typenames:
            celltypes[k]['start'] = population_ranges[k][0]
            celltypes[k]['num'] = population_ranges[k][1]
            if 'mechanism file' in celltypes[k]:
                celltypes[k]['mech_file_path'] = '%s/%s' % (self.config_prefix, celltypes[k]['mechanism file'])

        population_names  = read_population_names(self.data_file_path, self.comm)
        if rank == 0:
            self.logger.info('population_names = %s' % str(population_names))
        self.cellAttributeInfo = read_cell_attribute_info(self.data_file_path, population_names, comm=self.comm)

        if rank == 0:
            self.logger.info('attribute info: %s'  % str(self.cellAttributeInfo))

    def load_cell_template(self, popName):
        """

        :param popName: str
        """
        rank = self.comm.Get_rank()
        if not (popName in self.celltypes):
            raise KeyError('Env.load_cell_templates: unrecognized cell population: %s' % popName)
        templateName = self.celltypes[popName]['template']
        if 'template file' in self.celltypes[popName]:
            template_file = self.celltypes[popName]['template file']
        else:
            template_file = None
        find_template(self, templateName, template_file=template_file, path=self.template_paths)
        assert(hasattr(h, templateName))
        template_class = getattr(h, templateName)
        return template_class
