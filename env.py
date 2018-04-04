import sys, os.path, string
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_cell_attributes, read_population_ranges, read_population_names, read_cell_attribute_info
import numpy as np
from collections import namedtuple, defaultdict
import yaml
import dentate
from dentate import utils

ConnectionGenerator = namedtuple('ConnectionGenerator',
                                 ['synapse_types',
                                  'synapse_locations',
                                  'synapse_layers',
                                  'synapse_proportions',
                                  'synapse_kinetics',
                                  'connection_properties'])


class Env:
    """Network model configuration."""

    def load_input_config(self):
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
        
        syntypes_dict    = self.modelConfig['Definitions']['Synapse Types']
        swctypes_dict    = self.modelConfig['Definitions']['SWC Types']
        layers_dict      = self.modelConfig['Definitions']['Layers']
        synapse_kinetics = self.modelConfig['Connection Generator']['Synapse Kinetics']
        synapse_types    = self.modelConfig['Connection Generator']['Synapse Types']
        synapse_locs     = self.modelConfig['Connection Generator']['Synapse Locations']
        synapse_layers   = self.modelConfig['Connection Generator']['Synapse Layers']
        synapse_proportions   = self.modelConfig['Connection Generator']['Synapse Proportions']
        connection_properties = self.modelConfig['Connection Generator']['Connection Properties']

        connection_generator_dict = {}
        
        for (key_postsyn, val_syntypes) in synapse_types.iteritems():
            connection_generator_dict[key_postsyn]  = {}
            
            for (key_presyn, val_syntypes) in val_syntypes.iteritems():
                val_synlocs     = synapse_locs[key_postsyn][key_presyn]
                val_synlayers   = synapse_layers[key_postsyn][key_presyn]
                val_proportions = synapse_proportions[key_postsyn][key_presyn]
                val_synkins     = synapse_kinetics[key_postsyn][key_presyn]
                val_connprops1  = {}
                for (k_mech,v_mech) in connection_properties[key_postsyn][key_presyn].iteritems():
                    v_mech1 = {}
                    for (k,v) in v_mech.iteritems():
                        v1 = v
                        if type(v) is dict:
                            if v.has_key('from file'):
                                with open(v['from file']) as fp:
                                    lst = []
                                    lines = fp.readlines()
                                    for l in lines:
                                        lst.append(float(l))
                            v1 = h.Vector(np.asarray(lst, dtype=np.float32))
                        v_mech1[k] = v1
                    val_connprops1[k_mech] = v_mech1
                            

                val_syntypes1  = [syntypes_dict[val_syntype] for val_syntype in val_syntypes]
                val_synlocs1   = [swctypes_dict[val_synloc] for val_synloc in val_synlocs]
                val_synlayers1 = [layers_dict[val_synlayer] for val_synlayer in val_synlayers]
                
                connection_generator_dict[key_postsyn][key_presyn] = ConnectionGenerator(val_syntypes1,
                                                                                         val_synlocs1,
                                                                                         val_synlayers1,
                                                                                         val_proportions,
                                                                                         val_synkins,
                                                                                         val_connprops1)
                
            
        self.connection_generator = connection_generator_dict

    def load_celltypes(self):
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        celltypes = self.celltypes
        typenames = celltypes.keys()
        typenames.sort()

        self.comm.Barrier()

        datasetPath = os.path.join(self.datasetPrefix,self.datasetName)
        dataFilePath = os.path.join(datasetPath,self.modelConfig['Cell Data'])

        (population_ranges, _) = read_population_ranges(dataFilePath, self.comm)
        if rank == 0:
            print 'population_ranges = ', population_ranges
        
        for k in typenames:
            celltypes[k]['start'] = population_ranges[k][0]
            celltypes[k]['num'] = population_ranges[k][1]

        population_names  = read_population_names(dataFilePath, self.comm)
        if rank == 0:
            print 'population_names = ', population_names
        self.cellAttributeInfo = read_cell_attribute_info(dataFilePath, population_names, self.comm)

        if rank == 0:
            print 'attribute info: ', self.cellAttributeInfo

            
    def __init__(self, comm=None, configFile=None, templatePaths=None, datasetPrefix=None, resultsPath=None, resultsId=None, nodeRankFile=None,
                 IOsize=0, vrecordFraction=0, coredat=False, tstop=0, v_init=-65, max_walltime_hrs=0, results_write_time=0, dt=0.025, ldbal=False, lptbal=False, verbose=False):
        """
        :param configFile: the name of the model configuration file
        :param datasetPrefix: the location of all datasets
        :param resultsPath: the directory in which to write spike raster and voltage trace files
        :param resultsId: identifier that is used to constructs the namespaces in which to spike raster data and voltage trace data are written
        :param nodeRankFile: the name of a file that specifies assignment of node gids to MPI ranks
        :param IOsize: the number of MPI ranks to be used for I/O operations
        :param v_init: initialization membrane potential
        :param tstop: physical time to simulate
        :param max_walltime_hrs:  maximum wall time in hours
        :param results_write_time: time to write out results at end of simulation
        :param dt: simulation time step
        :param vrecordFraction: fraction of cells to record intracellular voltage from
        :param coredat: Save CoreNEURON data
        :param ldbal: estimate load balance based on cell complexity
        :param lptbal: calculate load balance with LPT algorithm
        :param verbose: print verbose diagnostic messages while constructing the network
        """

        self.SWC_Types = {}
        self.Synapse_Types = {}

        self.gidlist = []
        self.cells  = []

        self.comm = comm
        
        self.colsep = ' ' # column separator for text data files
        self.bufsize = 100000 # buffer size for text data files

        # print verbose diagnostic messages while constructing the network
        self.verbose = verbose

        # Directories for cell templates
        self.templatePaths=[]
        if templatePaths is not None:
            self.templatePaths = string.split(templatePaths, ':')

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

        # simulation time
        self.tstop = tstop

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
                self.modelConfig = yaml.load(fp, utils.IncludeLoader)
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
            self.load_celltypes()

        if self.modelConfig.has_key('Input'):
            self.load_input_config()
            
        if self.modelConfig.has_key('LFP'):
            self.lfpConfig = { 'position': tuple(self.modelConfig['LFP']['position']),
                               'maxEDist': self.modelConfig['LFP']['maxEDist'],
                               'fraction': self.modelConfig['LFP']['fraction'],
                               'rho': self.modelConfig['LFP']['rho'],
                               'dt': self.modelConfig['LFP']['dt'] }
        else:
            self.lfpConfig = None
            
        self.t_vec = h.Vector()   # Spike time of all cells on this host
        self.id_vec = h.Vector()  # Ids of spike times on this host

        self.v_dict  = defaultdict(lambda: {})  # Voltage samples on this host
            
        # used to calculate model construction times and run time
        self.mkcellstime = 0
        self.mkstimtime  = 0
        self.connectcellstime = 0
        self.connectgjstime = 0

        self.simtime = None
        self.lfp = None
