import sys, os.path, string
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neurotrees.io import read_tree_attributes
import numpy as np
import yaml

class Env:
    """Network model configuration."""

    def load_prjtypes(self):
        projections = self.projections
        prjnames = projections.keys()
        prjnames.sort()
        for k in prjnames:
            projection = projections[k]
            if projection['type'] == 'syn':
                weights = np.fromfile(projection['weightsFile'],sep='\n')
                projection['weights'] = weights

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
                    coords = read_tree_attributes(MPI._addressof(iocomm), fpath, k, namespace="Coordinates")
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
    
    def __init__(self, comm, configFile, templatePaths, datasetPrefix, resultsPath, nodeRankFile,
                 IOsize, vrecordFraction, coredat, tstop, v_init, max_walltime_hrs, results_write_time, dt, ldbal, lptbal, verbose):
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

        self.gidlist = []
        self.cells  = []

        self.comm = comm
        
        self.colsep = ' ' # column separator for text data files
        self.bufsize = 100000 # buffer size for text data files

        # print verbose diagnostic messages while constructing the network
        self.verbose = verbose

        # Directories for cell templates
        self.templatePaths=[]
        if templatePaths:
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

        with open(configFile) as fp:
            self.modelConfig = yaml.load(fp)
            
        # The name of this model
        self.modelName = self.modelConfig['modelname']
        # The dataset to use for constructing the network
        self.datasetName = self.modelConfig['datasetName']

        self.celltypes     = self.modelConfig['celltypes']
        self.synapseOrder  = self.modelConfig['synapses']['order']
        self.connectivityFile = self.modelConfig['connectivity']['connectivityFile']
        self.projections   = self.modelConfig['connectivity']['projections']
        if self.modelConfig['connectivity'].has_key('gapjunctions'):
            self.gapjunctions  = self.modelConfig['connectivity']['gapjunctions']
        else:
            self.gapjunctions  = None
        if self.modelConfig['connectivity'].has_key('gapjunctionsFile'):
            self.gapjunctionsFile = self.modelConfig['connectivity']['gapjunctionsFile']
        else:
            self.gapjunctionsFile = None
        self.load_celltypes()
        self.load_prjtypes()

        self.t_vec = h.Vector()   # Spike time of all cells on this host
        self.id_vec = h.Vector()  # Ids of spike times on this host

        # used to calculate model construction times and run time
        self.mkcellstime = 0
        self.mkstimtime  = 0
        self.connectcellstime = 0
        self.connectgjstime = 0
