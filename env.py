import sys, os.path, string
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_cell_attributes
import numpy as np
from collections import namedtuple
import yaml
import utils

ConnectionGenerator = namedtuple('ConnectionGenerator',
                                 ['synapse_types',
                                  'synapse_locations',
                                  'synapse_layers',
                                  'synapse_proportions',
                                  'synapse_kinetics',
                                  'connection_properties'])


class Env:
    """Network model configuration."""

    def load_connection_generator(self):
        populations_dict = self.modelConfig['Definitions']['Populations']
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
                val_connprops   = connection_properties[key_postsyn][key_presyn]
                
                val_syntypes1  = [syntypes_dict[val_syntype] for val_syntype in val_syntypes]
                val_synlocs1   = [swctypes_dict[val_synloc] for val_synloc in val_synlocs]
                val_synlayers1 = [layers_dict[val_synlayer] for val_synlayer in val_synlayers]
                
                connection_generator_dict[key_postsyn][key_presyn] = ConnectionGenerator(val_syntypes1,
                                                                                         val_synlocs1,
                                                                                         val_synlayers1,
                                                                                         val_proportions,
                                                                                         val_synkins,
                                                                                         val_connprops)
                
            
        self.connection_generator = connection_generator_dict

    def load_celltypes(self):
        # use this communicator for small size I/O operations performed by rank 0
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        if rank == 0:
            color = 1
        else:
            color = 2
        iocomm = self.comm.Split(color, rank)
        offset = 0
        celltypes = self.celltypes
        typenames = celltypes.keys()
        typenames.sort()
        for k in typenames:
            if celltypes[k].has_key('cardinality'):
                num = celltypes[k]['cardinality']
                celltypes[k]['offset'] = offset
                celltypes[k]['num'] = num
                start  = offset
                offset = offset+num
                index  = range(start, offset)
                celltypes[k]['index'] = index
            elif celltypes[k].has_key('indexFile'):
                fpath = os.path.join(self.datasetPrefix,self.datasetName,celltypes[k]['indexFile'])
                if rank == 0:
                    coords = read_cell_attributes(MPI._addressof(iocomm), fpath, k, namespace="Coordinates")
                    index  = coords.keys()
                    index.sort()
                else:
                    index = None
                index = self.comm.bcast(index, root=0)
                celltypes[k]['index'] = index
                celltypes[k]['offset'] = offset
                celltypes[k]['num'] = len(index)
                offset=max(index)+1
        iocomm.Free()
    
    def __init__(self, comm=None, configFile=None, templatePaths=None, datasetPrefix=None, resultsPath=None, nodeRankFile=None,
                 IOsize=0, vrecordFraction=0, coredat=False, tstop=0, v_init=-65, max_walltime_hrs=0, results_write_time=0, dt=0.025, ldbal=False, lptbal=False, verbose=False):
        """
        :param configFile: the name of the model configuration file
        :param datasetPrefix: the location of all datasets
        :param resultsPath: the directory in which to write spike raster and voltage trace files
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
        self.SWC_types = defs['SWC Types']
        self.synapse_types = defs['Synapse Types']
        self.layers = defs['Layers']

        self.celltypes = self.modelConfig['Cell Types']

        # The name of this model
        self.modelName = self.modelConfig['Model Name']
        # The dataset to use for constructing the network
        self.datasetName = self.modelConfig['Dataset Name']

        if self.modelConfig.has_key('Connection Generator'):
            self.load_connection_generator()
        
        if self.datasetPrefix is not None:
            self.load_celltypes()
            
        self.t_vec = h.Vector()   # Spike time of all cells on this host
        self.id_vec = h.Vector()  # Ids of spike times on this host

        # used to calculate model construction times and run time
        self.mkcellstime = 0
        self.mkstimtime  = 0
        self.connectcellstime = 0
        self.connectgjstime = 0
