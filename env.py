
from dentate.utils import *
from dentate.neuron_utils import *
from neuroh5.io import read_projection_names, read_population_ranges, read_population_names, read_cell_attribute_info
from dentate.synapses import SynapseAttributes
import logging


ConnectionGenerator = namedtuple('ConnectionGenerator',
                                 ['synapse_types',
                                  'synapse_locations',
                                  'synapse_layers',
                                  'synapse_proportions',
                                  'synapse_parameters'])


class Env:
    """
    Network model configuration.
    """
    def __init__(self, comm=None, configFile=None, templatePaths=None, hoclibPath=None, datasetPrefix=None,
                 resultsPath=None, resultsId=None, nodeRankFile=None, IOsize=0, vrecordFraction=0, coredat=False,
                 tstop=0, v_init=-65, stimulus_onset=0.0, max_walltime_hrs=0, results_write_time=0, dt=0.025,
                 ldbal=False, lptbal=False, verbose=False, **kwargs):
        """
        :param comm: :class:'MPI.COMM_WORLD'
        :param configFile: str; model configuration file
        :param templatePaths: str; colon-separated list of paths to directories containing hoc cell templates
        :param hoclibPath: str; path to directory containing required hoc libraries
        :param datasetPrefix: str; path to directory containing required neuroh5 data files
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

        if configFile is not None:
            with open(configFile) as fp:
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

        if self.modelConfig.has_key('Connection Generator'):
            self.load_connection_generator()

        if self.datasetPrefix is not None:
            self.datasetPath = os.path.join(self.datasetPrefix, self.datasetName)
            self.dataFilePath = os.path.join(self.datasetPath, self.modelConfig['Cell Data'])
            self.load_celltypes()
            self.connectivityFilePath = os.path.join(self.datasetPath, self.modelConfig['Connection Data'])
            self.forestFilePath = os.path.join(self.datasetPath, self.modelConfig['Cell Data'])

        if self.modelConfig.has_key('Input'):
            self.load_input_config()

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

    def load_input_config(self):
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

    def load_connection_generator(self):
        """

        :return:
        """
        populations_dict = self.modelConfig['Definitions']['Populations']
        self.pop_dict = populations_dict
        syntypes_dict    = self.modelConfig['Definitions']['Synapse Types']
        self.syntypes_dict = syntypes_dict
        swctypes_dict    = self.modelConfig['Definitions']['SWC Types']
        layers_dict      = self.modelConfig['Definitions']['Layers']
        synapse_parameters = self.modelConfig['Connection Generator']['Synapse Parameters']
        synapse_types    = self.modelConfig['Connection Generator']['Synapse Types']
        synapse_locs     = self.modelConfig['Connection Generator']['Synapse Locations']
        synapse_layers   = self.modelConfig['Connection Generator']['Synapse Layers']
        synapse_proportions   = self.modelConfig['Connection Generator']['Synapse Proportions']
        self.connection_velocity = self.modelConfig['Connection Generator']['Connection Velocity']
        syn_mech_names = self.modelConfig['Connection Generator']['Synapse Mechanisms']
        syn_param_rules = self.modelConfig['Connection Generator']['Synapse Parameter Rules']
        self.synapse_attributes = SynapseAttributes(syn_mech_names, syn_param_rules)
        connection_generator_dict = {}
        
        for (key_postsyn, val_syntypes) in synapse_types.iteritems():
            connection_generator_dict[key_postsyn]  = {}
            
            for (key_presyn, val_syntypes) in val_syntypes.iteritems():
                val_synlocs     = synapse_locs[key_postsyn][key_presyn]
                val_synlayers   = synapse_layers[key_postsyn][key_presyn]
                val_proportions = synapse_proportions[key_postsyn][key_presyn]
                val_synparams   = synapse_parameters[key_postsyn][key_presyn]
                val_syntypes1  = [syntypes_dict[val_syntype] for val_syntype in val_syntypes]
                val_synlocs1   = [swctypes_dict[val_synloc] for val_synloc in val_synlocs]
                val_synlayers1 = [layers_dict[val_synlayer] for val_synlayer in val_synlayers]
                
                connection_generator_dict[key_postsyn][key_presyn] = \
                    ConnectionGenerator(val_syntypes1, val_synlocs1, val_synlayers1, val_proportions, val_synparams)

        self.connection_generator = connection_generator_dict

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
            logger.info('population_ranges = %s' % str(population_ranges))
        
        for k in typenames:
            celltypes[k]['start'] = population_ranges[k][0]
            celltypes[k]['num'] = population_ranges[k][1]

        population_names  = read_population_names(self.dataFilePath, self.comm)
        if rank == 0 and self.verbose:
            logger.info('population_names = %s' % str(population_names))
        self.cellAttributeInfo = read_cell_attribute_info(self.dataFilePath, population_names, comm=self.comm)

        if rank == 0 and self.verbose:
            logger.info('attribute info: %s'  % str(self.cellAttributeInfo))

    def load_cell_template(self, popName):
        """

        :param popName: str
        """
        rank = self.comm.Get_rank()
        if not self.celltypes.has_key(popName):
            raise KeyError('Env.load_cell_templates: unrecognized cell population: %s' % popName)
        templateName = self.celltypes[popName]['template']
        if not hasattr(h, templateName):
            if 'templateFile' in self.celltypes[popName]:
                templateFile = self.celltypes[popName]['templateFile']
                templateFilePath = None
                for templatePath in self.templatePaths:
                    if os.path.isfile(templatePath + '/' + templateFile):
                        templateFilePath = templatePath + '/' + templateFile
                        break
                if templateFilePath is None:
                    raise IOError('Env.load_cell_templates: population: %s; templateFile not found: %s' %
                                  (popName, templateFile))
                h.load_file(templateFilePath)
                if rank == 0 and self.verbose:
                    logger.info('load_cell_templates: population: %s; templateFile loaded: %s' % \
                                (popName, templateFilePath))
            else:
                h.find_template(self.pc, h.templatePaths, templateName)
