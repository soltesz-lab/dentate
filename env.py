import yaml

class Env:
    """Network model configuration."""

    def load_celltypes(self):
        offset = 0
        celltypes = self.celltypes['celltypes']
        for k in celltypes.keys():
            if celltypes[k].has_key('cardinality'):
                num = celltypes[k]['cardinality']
                celltypes[k]['offset'] = offset
                celltypes[k]['num'] = num
                offset = offset+num
            elif celltypes[k].has_key('indexfile'):
                index=[]
                f = open(inputfile)
                lines = f.readlines(self.bufsize)
                while lines:
                    for l in lines:
                        a = l.split(self.colsep)
                        index.append(int(a[0]))
                    lines = f.readlines(self.bufsize)
                f.close()
                celltypes[k]['index'] = index
                celltypes[k]['offset'] = offset
                celltypes[k]['num'] = len(index)
                assert(offset >= min(index))
                offset=offset+len(index)
                
    
    def __init__(self, comm,
                     modelName, datasetPrefix, datasetName, celltypesFileName, connectivityFileName, gapjunctionsFileName, resultsPath,
                     IOsize, vrecordFraction, verbose, coredat, tstop, v_init, max_walltime_hrs, results_write_time, dt):
        """
        :param modelName: the name of this model
        :param datasetPrefix: the location of all datasets
        :param datasetName: the dataset to use for constructing the network
        :param celltypesFileName: the cell type/number configuration file to use for constructing the populations in the network
        :param connectivityFileName: the connectivity configuration file to use for constructing the projections in the network
        :param gapjunctionsFileName: the connectivity configuration file to use for constructing the gap junction connections in the network
        :param resultsPath: the directory in which to write spike raster and voltage trace files
        :param IOsize: the number of MPI ranks to be used for I/O operations
        :param v_init: initialization membrane potential
        :param tstop: physical time to simulate
        :param max_walltime_hrs:  maximum wall time in hours
        :param results_write_time: time to write out results at end of simulation
        :param dt: simulation time step
        :param vrecordFraction: fraction of cells to record intracellular voltage from
        :param coredat: Save CoreNEURON data
        :param verbose: print verbose diagnostic messages while constructing the network
        """

        self.comm = comm
        
        self.colsep = ' ' # column separator for text data files
        self.bufsize = 100000 # buffer size for text data files

        # print verbose diagnostic messages while constructing the network
        self.verbose = verbose

        # The name of this model
        self.modelName = modelName

        # The location of all datasets
        self.datasetPrefix = datasetPrefix

        # The dataset to use for constructing the network
        self.datasetName = datasetName

        # The path where results files should be written
        self.resultsPath = resultsPath

        # The cell type/number configuration file to use for constructing the populations in the network
        self.celltypesFileName=celltypesFileName
        self.celltypesPath='%s/%s/%s' % (datasetPrefix,datasetName,celltypesFileName)

        # The connectivity configuration file to use for constructing the projections in the network
        self.connectivityFileName=connectivityFileName
        self.connectivityPath='%s/%s/%s' % (datasetPrefix,datasetName,connectivityFileName)
        
        # The connectivity configuration file to use for constructing the gap junction connections in the network
        self.gapjunctionsFileName=gapjunctionsFileName
        self.gapjunctionsPath='%s/%s/%s' % (datasetPrefix,datasetName,gapjunctionsFileName)
        
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

        # Fraction of cells to record intracellular voltage from
        self.vrecordFraction = vrecordFraction

        # Save CoreNEURON data
        self.coredat = coredat

        with open(self.celltypesPath) as fp:
            self.celltypes = yaml.load(fp)
        self.load_celltypes()
            
        with open(self.connectivityPath) as fp:
            self.connectivity = yaml.load(fp)

        with open(self.gapjunctionsPath) as fp:
            self.gapjunctions = yaml.load(fp)

        self.cells = []
        self.H5Graph = {}

        self.t_vec = h.Vector()   # Spike time of all cells on this host
        self.id_vec = h.Vector()  # Ids of spike times on this host
