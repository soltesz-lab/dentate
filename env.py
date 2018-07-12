import numpy as np
from dentate.utils import *
from dentate.neuron_utils import *
from neuroh5.io import read_projection_names, read_population_ranges, read_population_names, read_cell_attribute_info
from dentate.synapses import SynapseAttributes

ConnectionConfig = namedtuple('ConnectionConfig',
                                 ['type',
                                  'sections',
                                  'layers',
                                  'proportions',
                                  'mechanisms'])

GapjunctionConfig = namedtuple('GapjunctionConfig',
                                 ['sections',
                                  'connection_probabilities',
                                  'connection_parameters',
                                  'coupling_coefficients',
                                  'coupling_parameters'])


class Env:
    """
    Network model configuration.
    """
    def __init__(self, comm=None, configFile=None, templatePaths=None, hoclibPath=None, datasetPrefix=None,
                 configPrefix=None, resultsPath=None, resultsId=None, nodeRankFile=None, IOsize=0, vrecordFraction=0,
                 coredat=False, tstop=0, v_init=-65, stimulus_onset=0.0, max_walltime_hrs=0, results_write_time=0,
                 dt=0.025, ldbal=False, lptbal=False, verbose=False, **kwargs):
        """
        :param comm: :class:'MPI.COMM_WORLD'
        :param configFile: str; model configuration file name
        :param templatePaths: str; colon-separated list of paths to directories containing hoc cell templates
        :param hoclibPath: str; path to directory containing required hoc libraries
        :param datasetPrefix: str; path to directory containing required neuroh5 data files
        :param configPrefix: str; path to directory containing network and cell mechanism config files
        :param resultsPath: str; path to directory to export output files
        :param resultsId: str; label for neuroh5 namespaces to write spike and voltage trace data
        :param nodeRankFile: str; name of file specifying assignment of node gids to MPI ranks
        :param IOsize: int; the number of MPI ranks to be used for I/O operations
        :param vrecordFraction: float; fraction of cells to record intracellular voltage from
        :param coredat: bool; Save CoreNEURON data
        :param tstop: int; physical time to simulate (ms)
        :param v_init: float; initialization membrane potential (mV)
        :param stimulus_onset: float; starting time of stimulus (ms)
        :param max_walltime_hrs: float; maximum wall time (hours)
        :param results_write_time: float; time to write out results at end of simulation
        :param dt: float; simulation time step
        :param ldbal: bool; estimate load balance based on cell complexity
        :param lptbal: bool; calculate load balance with LPT algorithm
        :param verbose: bool; print verbose diagnostic messages while constructing the network
        """
        self.SWC_Types = {}
        self.Synapse_Types = {}

        self.gidlist = []
        self.cells = []
        self.biophys_cells = defaultdict(dict)

        self.comm = comm

        self.colsep = ' '  # column separator for text data files
        self.bufsize = 100000  # buffer size for text data files

        # print verbose diagnostic messages
        self.verbose = verbose
        config_logging(verbose)
        self.logger = get_root_logger()
        
        # Directories for cell templates
        self.templatePaths = []
        if templatePaths is not None:
            self.templatePaths = string.split(templatePaths, ':')

        # The location of required hoc libraries
        self.hoclibPath = hoclibPath

        # The location of all datasets
        self.datasetPrefix = datasetPrefix

        # The path where results files should be written
        self.resultsPath = resultsPath

        # Identifier used to construct results data namespaces
        self.resultsId = resultsId

        # Number of MPI ranks to be used for I/O operations
        self.IOsize = IOsize

        # Initialization voltage
        self.v_init = v_init

        # simulation time [ms]
        self.tstop = tstop

        # stimulus onset time [ms]
        self.stimulus_onset = stimulus_onset

        # maximum wall time in hours
        self.max_walltime_hrs = max_walltime_hrs

        # time to write out results at end of simulation
        self.results_write_time = results_write_time

        # time step
        self.dt = dt

        # used to estimate cell complexity
        self.cxvec = None

        # measure/perform load balancing
        self.optldbal = ldbal
        self.optlptbal = lptbal

        # Fraction of cells to record intracellular voltage from
        self.vrecordFraction = vrecordFraction

        # Save CoreNEURON data
        self.coredat = coredat

        self.nodeRanks = None
        if nodeRankFile:
            with open(nodeRankFile) as fp:
                dval = {}
                lines = fp.readlines()
                for l in lines:
                    a = l.split(' ')
                    dval[int(a[0])] = int(a[1])
                self.nodeRanks = dval

        self.configPrefix = configPrefix

        if configFile is not None:
            if configPrefix is not None:
                configFilePath = self.configPrefix + '/' + configFile
            else:
                configFilePath = configFile
            if not os.path.isfile(configFilePath):
                raise RuntimeError("missing configuration file")
            with open(configFilePath) as fp:
                self.modelConfig = yaml.load(fp, IncludeLoader)
        else:
            raise RuntimeError("missing configuration file")

        defs = self.modelConfig['Definitions']
        self.SWC_Types = defs['SWC Types']
        self.Synapse_Types = defs['Synapse Types']
        self.layers = defs['Layers']
        self.feature_types = defs['Input Features']
        if self.modelConfig.has_key('Geometry'):
            self.geometry = self.modelConfig['Geometry']
        else:
            self.geometry = None

        if self.geometry['Parametric Surface'].has_key('Origin'):
            self.parse_origin_coords()
            
        self.celltypes = self.modelConfig['Cell Types']
        self.cellAttributeInfo = {}

        # The name of this model
        self.modelName = self.modelConfig['Model Name']
        # The dataset to use for constructing the network
        self.datasetName = self.modelConfig['Dataset Name']

        if resultsPath:
            self.resultsFilePath = "%s/%s_results.h5" % (self.resultsPath, self.modelName)
        else:
            self.resultsFilePath = "%s_results.h5" % self.modelName

        if self.modelConfig.has_key('Definitions'):
            self.parse_definitions()

        if self.modelConfig.has_key('Connection Generator'):
            self.parse_connection_config()
            self.parse_gapjunction_config()

        if self.datasetPrefix is not None:
            self.datasetPath = os.path.join(self.datasetPrefix, self.datasetName)
            self.dataFilePath = os.path.join(self.datasetPath, self.modelConfig['Cell Data'])
            self.load_celltypes()
            self.connectivityFilePath = os.path.join(self.datasetPath, self.modelConfig['Connection Data'])
            self.forestFilePath = os.path.join(self.datasetPath, self.modelConfig['Cell Data'])
            if self.modelConfig.has_key('Gap Junction Data'):
                self.gapjunctionsFilePath = os.path.join(self.datasetPath, self.modelConfig['Gap Junction Data'])
            else:
                self.gapjunctionsFilePath = None

        if self.modelConfig.has_key('Input'):
            self.parse_input_config()

        self.projection_dict = defaultdict(list)
        if self.datasetPrefix is not None:
            for (src, dst) in read_projection_names(self.connectivityFilePath, comm=self.comm):
                self.projection_dict[dst].append(src)

        self.lfpConfig = {}
        if self.modelConfig.has_key('LFP'):
            for label, config in self.modelConfig['LFP'].iteritems():
                self.lfpConfig[label] = {'position': tuple(config['position']),
                                         'maxEDist': config['maxEDist'],
                                         'fraction': config['fraction'],
                                         'rho': config['rho'],
                                         'dt': config['dt']}

        self.t_vec = h.Vector()  # Spike time of all cells on this host
        self.id_vec = h.Vector()  # Ids of spike times on this host

        self.v_dict = defaultdict(lambda: {})  # Voltage samples on this host

        # used to calculate model construction times and run time
        self.mkcellstime = 0
        self.mkstimtime = 0
        self.connectcellstime = 0
        self.connectgjstime = 0

        self.simtime = None
        self.lfp = {}

        self.edge_count = defaultdict(dict)
        self.syns_set = defaultdict(set)

        if self.hoclibPath:
            # polymorphic hoc value template
            h.load_file(self.hoclibPath + '/templates/Value.hoc')
            # stimulus cell template
            h.load_file(self.hoclibPath + '/templates/StimCell.hoc')
            h.xopen(self.hoclibPath + '/lib.hoc')

    def parse_input_config(self):
        """

        :return:
        """
        features_type_dict = self.modelConfig['Definitions']['Input Features']
        input_dict = self.modelConfig['Input']
        input_config = {}
        
        for (id,dvals) in input_dict.iteritems():
            config_dict = {}
            config_dict['trajectory'] = dvals['trajectory']
            feature_type_dict = {}
            for (pop,pdvals) in dvals['feature type'].iteritems():
                pop_feature_type_dict = {}
                for (feature_type_name,feature_type_fraction) in pdvals.iteritems():
                    pop_feature_type_dict[int(self.feature_types[feature_type_name])] = float(feature_type_fraction)
                feature_type_dict[pop] = pop_feature_type_dict
            config_dict['feature type'] = feature_type_dict
            input_config[int(id)] = config_dict

        self.inputConfig = input_config


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
        populations_dict = self.modelConfig['Definitions']['Populations']
        self.pop_dict = populations_dict
        syntypes_dict    = self.modelConfig['Definitions']['Synapse Types']
        self.syntypes_dict = syntypes_dict
        swctypes_dict    = self.modelConfig['Definitions']['SWC Types']
        self.swctypes_dict = swctypes_dict
        layers_dict      = self.modelConfig['Definitions']['Layers']
        self.layers_dict = layers_dict
        
    def parse_connection_config(self):
        """

        :return:
        """
        connection_config = self.modelConfig['Connection Generator']
        
        self.connection_velocity = connection_config['Connection Velocity']

        syn_mech_names  = connection_config['Synapse Mechanisms']
        syn_param_rules = connection_config['Synapse Parameter Rules']

        self.synapse_attributes = SynapseAttributes(syn_mech_names, syn_param_rules)

        synapse_config = connection_config['Synapses']
        connection_dict = {}
        
        for (key_postsyn, val_syntypes) in synapse_config.iteritems():
            connection_dict[key_postsyn]  = {}
            
            for (key_presyn, syn_dict) in val_syntypes.iteritems():
                val_type        = syn_dict['type']
                val_synsections = syn_dict['sections']
                val_synlayers   = syn_dict['layers']
                val_proportions = syn_dict['proportions']
                val_synparams   = syn_dict['mechanisms']

                res_type = self.syntypes_dict[val_type]
                res_synsections = []
                res_synlayers = []
                for name in val_synsections:
                    res_synsections.append(self.swctypes_dict[name])
                for name in val_synlayers:
                    res_synlayers.append(self.layers_dict[name])
                
                connection_dict[key_postsyn][key_presyn] = \
                    ConnectionConfig(res_type, \
                                     res_synsections, \
                                     res_synlayers, \
                                     val_proportions, \
                                     val_synparams)


            config_dict = defaultdict(lambda: 0.0)
            for (key_presyn, conn_config) in connection_dict[key_postsyn].iteritems():
                for (s,l,p) in itertools.izip(conn_config.sections, \
                                              conn_config.layers, \
                                              conn_config.proportions):
                    config_dict[(conn_config.type, s, l)] += p
                                              
            for (k,v) in config_dict.iteritems():
                try:
                    assert(v == 1.0)
                except Exception as e:
                    logger.error('Connection configuration: probabilities for %s do not sum to 1: %s = %f' % (key_postsyn, str(k), v))
                    raise e
                    
        self.connection_config = connection_dict

    def parse_gapjunction_config(self):
        """

        :return:
        """
        connection_config = self.modelConfig['Connection Generator']
        if connection_config.has_key('Gap Junctions'):
            gj_config = connection_config['Gap Junctions']

            gj_sections = gj_config['Locations']
            sections = {}
            for pair, sec_names in gj_sections.iteritems():
                sec_idxs = []
                for sec_name in sec_names:
                    sec_idxs.append(self.swctypes_dict[sec_name])
                sections[pair] = sec_idxs

            gj_connection_probs = gj_config['Connection Probabilities']
            connection_probs = {}
            for pair, prob in gj_connection_probs.iteritems():
                connection_probs[pair] = float(prob)

            gj_connection_probs = gj_config['Connection Probabilities']
            connection_probs = {}
            for pair, prob in gj_connection_probs.iteritems():
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

            gj_coupling_coeffs = gj_config['Coupling Coefficients']
            coupling_coeffs = {}
            for pair, coeff in gj_coupling_coeffs.iteritems():
                coupling_coeffs[pair] = float(coeff)

            coupling_weights_x = []
            coupling_weights_y = []
            gj_coupling_weights = gj_config['Coupling Weights']
            for x in sorted(gj_coupling_weights.keys()):
                coupling_weights_x.append(x)
                coupling_weights_y.append(gj_coupling_weights[x])

            coupling_params = np.polyfit(np.asarray(coupling_weights_x), \
                                         np.asarray(coupling_weights_y), \
                                         3)
                
            self.gapjunctions = GapjunctionConfig(sections, \
                                                  connection_probs, \
                                                  connection_params, \
                                                  coupling_coeffs, \
                                                  coupling_params)
        else:
            self.gapjunctions = None
        
        
        
    def load_celltypes(self):
        """

        :return:
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        celltypes = self.celltypes
        typenames = celltypes.keys()
        typenames.sort()

        self.comm.Barrier()

        (population_ranges, _) = read_population_ranges(self.dataFilePath, self.comm)
        if rank == 0 and self.verbose:
            self.logger.info('population_ranges = %s' % str(population_ranges))
        
        for k in typenames:
            celltypes[k]['start'] = population_ranges[k][0]
            celltypes[k]['num'] = population_ranges[k][1]

        population_names  = read_population_names(self.dataFilePath, self.comm)
        if rank == 0 and self.verbose:
            self.logger.info('population_names = %s' % str(population_names))
        self.cellAttributeInfo = read_cell_attribute_info(self.dataFilePath, population_names, comm=self.comm)

        if rank == 0 and self.verbose:
            self.logger.info('attribute info: %s'  % str(self.cellAttributeInfo))

    def load_cell_template(self, popName):
        """

        :param popName: str
        """
        rank = self.comm.Get_rank()
        if not self.celltypes.has_key(popName):
            raise KeyError('Env.load_cell_templates: unrecognized cell population: %s' % popName)
        templateName = self.celltypes[popName]['template']

        h('objref templatePaths, templatePathValue')
        h.templatePaths = h.List()
        for path in self.templatePaths:
            h.templatePathValue = h.Value(1, path)
            h.templatePaths.append(h.templatePathValue)

        if not hasattr(h, templateName):
            if 'templateFile' in self.celltypes[popName]:
                templateFile = self.celltypes[popName]['templateFile']
                templateFilePath = None
                for templatePath in self.templatePaths:
                    if os.path.isfile(templatePath + '/' + templateFile):
                        templateFilePath = templatePath + '/' + templateFile
                        break
                if templateFilePath is None:
                    raise IOError('Env.load_cell_templates: population: %s; template not found: %s' %
                                  (popName, templateFile))
                h.load_file(templateFilePath)
                if rank == 0 and self.verbose:
                    self.logger.info('load_cell_templates: population: %s; template loaded: %s' % \
                                (popName, templateFilePath))
            else:
                h.find_template(self.pc, h.templatePaths, templateName)
