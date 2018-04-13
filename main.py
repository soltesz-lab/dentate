##
##  Dentate Gyrus model initialization script
##

import sys, os, os.path, click, itertools
from collections import defaultdict
from datetime import datetime
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_projection_names, scatter_read_graph, bcast_graph, scatter_read_trees, \
    scatter_read_cell_attributes, write_cell_attributes
import dentate    
from dentate.env import Env
from dentate import lpt, synapses, cells, lfp, simtime
from dentate.neuron_utils import nc_appendsyn, nc_appendsyn_wgtvector, mkgap
import logging
logging.basicConfig()

script_name = 'main.py'
logger = logging.getLogger(script_name)

## Estimate cell complexity. Code by Michael Hines from the discussion thread
## https://www.neuron.yale.edu/phpBB/viewtopic.php?f=31&t=3628
def cx(env):
  rank   = int(env.pc.id())
  lb = h.LoadBalance()
  if os.path.isfile("mcomplex.dat"):
    lb.read_mcomplex() 
  cxvec = h.Vector(len(env.gidlist))
  for i, gid in enumerate(env.gidlist):
    cxvec.x[i] = lb.cell_complexity(env.pc.gid2cell(gid))
  env.cxvec = cxvec
  return cxvec


# for given cxvec on each rank what is the fractional load balance.
def ld_bal(env):
  rank   = int(env.pc.id())
  nhosts = int(env.pc.nhost())
  cxvec  = env.cxvec
  sum_cx = sum(cxvec)
  max_sum_cx = env.pc.allreduce(sum_cx, 2)
  sum_cx = env.pc.allreduce(sum_cx, 1)
  if rank == 0:
    logger.info ("*** expected load balance %.2f" % (sum_cx / nhosts / max_sum_cx))


# Each rank has gidvec, cxvec: gather everything to rank 0, do lpt
# algorithm and write to a balance file.
def lpt_bal(env):
  rank   = int(env.pc.id())
  nhosts = int(env.pc.nhost())

  cxvec  = env.cxvec
  gidvec = env.gidlist
  #gather gidvec, cxvec to rank 0
  src    = [None]*nhosts
  src[0] = zip(cxvec.to_python(), gidvec)
  dest   = env.pc.py_alltoall(src)
  del src

  if rank == 0:
    lb = h.LoadBalance()
    allpairs = sum(dest,[])
    del dest
    parts = lpt.lpt(allpairs, nhosts)
    lpt.statistics(parts)
    part_rank = 0
    with open('parts.%d' % nhosts, 'w') as fp:
      for part in parts:
        for x in part[1]:
          fp.write('%d %d\n' % (x[1],part_rank))
        part_rank = part_rank+1
        
def mkout (env, results_filename):
  import h5py
  datasetPath   = os.path.join(env.datasetPrefix,env.datasetName)
  dataFilePath  = os.path.join(datasetPath,env.modelConfig['Cell Data'])
  dataFile      = h5py.File(dataFilePath,'r')
  resultsFile   = h5py.File(results_filename,'w')
  dataFile.copy('/H5Types',resultsFile)
  dataFile.close()
  resultsFile.close()
        
    
def spikeout (env, output_path, t_vec, id_vec):
    binlst  = []
    typelst = env.celltypes.keys()
    for k in typelst:
        binlst.append(env.celltypes[k]['start'])
        
    binvect  = np.array(binlst)
    sort_idx = np.argsort(binvect,axis=0)
    bins     = binvect[sort_idx][1:]
    types    = [ typelst[i] for i in sort_idx ]
    inds     = np.digitize(id_vec, bins)

    if not str(env.resultsId):
        namespace_id = "Spike Events" 
    else:
        namespace_id = "Spike Events %s" % str(env.resultsId)
    
    for i in range(0,len(types)):
        spkdict  = {}
        sinds    = np.where(inds == i)
        if len(sinds) > 0:
            ids      = id_vec[sinds]
            ts       = t_vec[sinds]
            for j in range(0,len(ids)):
                id = ids[j]
                t  = ts[j]
                if spkdict.has_key(id):
                    spkdict[id]['t'].append(t)
                else:
                    spkdict[id]= {'t': [t]}
            for j in spkdict.keys():
                spkdict[j]['t'] = np.array(spkdict[j]['t'], dtype=np.float32)
        pop_name = types[i]
        write_cell_attributes(output_path, pop_name, spkdict, namespace=namespace_id, comm=env.comm)
        del(spkdict)

def vout (env, output_path, t_vec, v_dict):

    if not str(env.resultsId):
        namespace_id = "Intracellular Voltage" 
    else:
        namespace_id = "Intracellular Voltage %s" % str(env.resultsId)

    for pop_name, gid_v_dict in v_dict.iteritems():
        
        attr_dict  = { gid : { 'v': np.array(vs, dtype=np.float32), 't' : t_vec }  for (gid, vs) in gid_v_dict.iteritems() }

        write_cell_attributes(output_path, pop_name, attr_dict, namespace=namespace_id, comm=env.comm)
        

def lfpout (env, output_path, lfp):

    if not str(env.resultsId):
        namespace_id = "Local Field Potential" 
    else:
        namespace_id = "Local Field Potential %s" % str(env.resultsId)

    import h5py
    output = h5py.File(output_path)

    grp = output.create_group(namespace_id)

    grp['t'] = np.asarray(lfp.t, dtype=np.float32)
    grp['v'] = np.asarray(lfp.meanlfp, dtype=np.float32)

    output.close()
        

def connectcells(env):
    datasetPath = os.path.join(env.datasetPrefix,env.datasetName)
    connectivityFilePath = os.path.join(datasetPath,env.modelConfig['Connection Data'])
    forestFilePath = os.path.join(datasetPath,env.modelConfig['Cell Data'])

    if env.verbose:
      if env.pc.id() == 0:
        logger.info('*** Connectivity file path is %s' % connectivityFilePath)

    prj_dict = defaultdict(list)
    for (src,dst) in read_projection_names(connectivityFilePath,comm=env.comm):
      prj_dict[dst].append(src)

    if env.verbose:
      if env.pc.id() == 0:
        logger.info('*** Reading projections: ')
      
    for (postsyn_name, presyn_names) in prj_dict.iteritems():

      synapse_config = env.celltypes[postsyn_name]['synapses']
      if synapse_config.has_key('spines'):
        spines = synapse_config['spines']
      else:
        spines = False

      if synapse_config.has_key('unique'):
        unique = synapse_config['unique']
      else:
        unique = False
      
      if synapse_config.has_key('weights'):
        has_weights = synapse_config['weights']
      else:
        has_weights = False

      if synapse_config.has_key('weights namespace'):
        weights_namespace = synapse_config['weights namespace']
      else:
        weights_namespace = 'Weights'
      
      if env.verbose:
          if int(env.pc.id()) == 0:
            logger.info('*** Reading synapse attributes of population %s' % (postsyn_name))

      if has_weights:
        cell_attr_namespaces = ['Synapse Attributes', weights_namespace]
      else:
        cell_attr_namespaces = ['Synapse Attributes']
        
            
      if env.nodeRanks is None:
        cell_attributes_dict = scatter_read_cell_attributes(forestFilePath, postsyn_name, 
                                                            namespaces=cell_attr_namespaces, comm=env.comm, 
                                                            io_size=env.IOsize)
      else:
        cell_attributes_dict = scatter_read_cell_attributes(forestFilePath, postsyn_name, 
                                                            namespaces=cell_attr_namespaces, comm=env.comm, 
                                                            node_rank_map=env.nodeRanks,
                                                            io_size=env.IOsize)
      cell_synapses_dict = { k : v for (k,v) in cell_attributes_dict['Synapse Attributes'] }
      if cell_attributes_dict.has_key(weights_namespace):
        has_weights = True
        cell_weights_dict = { k : v for (k,v) in cell_attributes_dict[weights_namespace] }
        if env.verbose:
          if env.pc.id() == 0:
            logger.info('*** Found synaptic weights for population %s' % (postsyn_name))
      else:
        has_weights = False
        cell_weights_dict = None
      del cell_attributes_dict


      for presyn_name in presyn_names:

        edge_count = 0

        if env.verbose:
          if env.pc.id() == 0:
            logger.info('*** Connecting %s -> %s' % (presyn_name, postsyn_name))

        if env.nodeRanks is None:
          (graph, a) = scatter_read_graph(connectivityFilePath,comm=env.comm,io_size=env.IOsize,
                                          projections=[(presyn_name, postsyn_name)],
                                          namespaces=['Synapses','Connections'])
        else:
          (graph, a) = scatter_read_graph(connectivityFilePath,comm=env.comm,io_size=env.IOsize,
                                          node_rank_map=env.nodeRanks,
                                          projections=[(presyn_name, postsyn_name)],
                                          namespaces=['Synapses','Connections'])
        
        edge_iter = graph[postsyn_name][presyn_name]


        connection_dict = env.connection_generator[postsyn_name][presyn_name].connection_properties
        kinetics_dict = env.connection_generator[postsyn_name][presyn_name].synapse_kinetics

        syn_id_attr_index = a[postsyn_name][presyn_name]['Synapses']['syn_id']
        distance_attr_index = a[postsyn_name][presyn_name]['Connections']['distance']

        for (postsyn_gid, edges) in edge_iter:

          postsyn_cell   = env.pc.gid2cell(postsyn_gid)
          cell_syn_dict  = cell_synapses_dict[postsyn_gid]

          if has_weights:
            cell_wgt_dict = cell_weights_dict[postsyn_gid]
            syn_wgt_dict = { int(syn_id): float(weight) for (syn_id, weight) in 
                               itertools.izip(np.nditer(cell_wgt_dict['syn_id']),
                                              np.nditer(cell_wgt_dict['weight'])) }
          else:
            syn_wgt_dict = None

          presyn_gids    = edges[0]
          edge_syn_ids   = edges[1]['Synapses'][syn_id_attr_index]
          edge_dists     = edges[1]['Connections'][distance_attr_index]

          cell_syn_types    = cell_syn_dict['syn_types']
          cell_swc_types    = cell_syn_dict['swc_types']
          cell_syn_locs     = cell_syn_dict['syn_locs']
          cell_syn_sections = cell_syn_dict['syn_secs']

          edge_syn_ps_dict  = synapses.mksyns(postsyn_gid,
                                              postsyn_cell,
                                              edge_syn_ids,
                                              cell_syn_types,
                                              cell_swc_types,
                                              cell_syn_locs,
                                              cell_syn_sections,
                                              kinetics_dict, env,
                                              add_synapse=synapses.add_unique_synapse if unique else synapses.add_shared_synapse,
                                              spines=spines)

          if env.verbose:
            if int(env.pc.id()) == 0:
              if edge_count == 0:
                for sec in list(postsyn_cell.all):
                  h.psection(sec=sec)

          wgt_count = 0
          for (presyn_gid, edge_syn_id, distance) in itertools.izip(presyn_gids, edge_syn_ids, edge_dists):
            syn_ps_dict = edge_syn_ps_dict[edge_syn_id]
            for (syn_mech, syn_ps) in syn_ps_dict.iteritems():
              connection_syn_mech_config = connection_dict[syn_mech]
              if has_weights and syn_wgt_dict.has_key(edge_syn_id):
                wgt_count += 1
                weight = float(syn_wgt_dict[edge_syn_id]) * connection_syn_mech_config['weight']
              else:
                weight = connection_syn_mech_config['weight']
              delay  = (distance / connection_syn_mech_config['velocity']) + 0.1
              if type(weight) is float:
                nc_appendsyn (env.pc, h.nclist, presyn_gid, postsyn_gid, syn_ps, weight, delay)
              else:
                nc_appendsyn_wgtvector (env.pc, h.nclist, presyn_gid, postsyn_gid, syn_ps, weight, delay)
          if env.verbose:
            if int(env.pc.id()) == 0:
              if edge_count == 0:
                logger.info('*** Found %i synaptic weights for gid %i' % (wgt_count, postsyn_gid))

          edge_count += len(presyn_gids)
          


def connectgjs(env):
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    datasetPath = os.path.join(env.datasetPrefix,env.datasetName)
    
    gapjunctions = env.gapjunctions
    if env.gapjunctionsFile is None:
        gapjunctionsFilePath = None
    else:
        gapjunctionsFilePath = os.path.join(datasetPath,env.gapjunctionsFile)

    if gapjunctions is not None:

        h('objref gjlist')
        h.gjlist = h.List()
        datasetPath = os.path.join(env.datasetPrefix,env.datasetName)
        (graph, a) = bcast_graph(gapjunctionsFilePath,attributes=True,comm=env.comm)

        ggid = 2e6
        for name in gapjunctions.keys():
            if env.verbose:
                if env.pc.id() == 0:
                    logger.info("*** Creating gap junctions %s" % name)
            prj = graph[name]
            attrmap = a[name]
            weight_attr_idx = attrmap['Weight']+1
            dstbranch_attr_idx = attrmap['Destination Branch']+1
            dstsec_attr_idx = attrmap['Destination Section']+1
            srcbranch_attr_idx = attrmap['Source Branch']+1
            srcsec_attr_idx = attrmap['Source Section']+1
            for destination in sorted(prj.keys()):
                edges = prj[destination]
                sources      = edges[0]
                weights      = edges[weight_attr_idx]
                dstbranches  = edges[dstbranch_attr_idx]
                dstsecs      = edges[dstsec_attr_idx]
                srcbranches  = edges[srcbranch_attr_idx]
                srcsecs      = edges[srcsec_attr_idx]
                for i in range(0,len(sources)):
                    source    = sources[i]
                    srcbranch = srcbranches[i]
                    srcsec    = srcsecs[i]
                    dstbranch = dstbranches[i]
                    dstsec    = dstsecs[i]
                    weight    = weights[i]
                    if env.pc.gid_exists(source):
                        mkgap(env.pc, h.gjlist, source, srcbranch, srcsec, ggid, ggid+1, weight)
                    if env.pc.gid_exists(destination):
                        mkgap(env.pc, h.gjlist, destination, dstbranch, dstsec, ggid+1, ggid, weight)
                    ggid = ggid+2

            del graph[name]


def mkcells(env):

    h('objref templatePaths, templatePathValue')

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    v_sample_seed = int(env.modelConfig['Random Seeds']['Intracellular Voltage Sample'])
    ranstream_v_sample = np.random.RandomState()
    ranstream_v_sample.seed(v_sample_seed)
    
    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePathValue = h.Value(1,path)
        h.templatePaths.append(h.templatePathValue)
    popNames = env.celltypes.keys()
    popNames.sort()
    for popName in popNames:
        templateName = env.celltypes[popName]['template']
        h.find_template(env.pc, h.templatePaths, templateName)

    dataFilePath = os.path.join(datasetPath,env.modelConfig['Cell Data'])

    for popName in popNames:

        if env.verbose:
            if env.pc.id() == 0:
                logger.info("*** Creating population %s" % popName)

        templateName = env.celltypes[popName]['template']
        templateClass = eval('h.%s' % templateName)

        if env.celltypes[popName].has_key('synapses'):
            synapses = env.celltypes[popName]['synapses']
        else:
            synapses = {}


        v_sample_set = set([])
        env.v_dict[popName] = {}
        
        for gid in xrange(env.celltypes[popName]['start'], env.celltypes[popName]['start']+env.celltypes[popName]['num']):
          if ranstream_v_sample.uniform() <= env.vrecordFraction:
            v_sample_set.add(gid)

        if env.cellAttributeInfo.has_key(popName) and env.cellAttributeInfo[popName].has_key('Trees'):
            if env.verbose:
                if env.pc.id() == 0:
                    logger.info("*** Reading trees for population %s" % popName)

            if env.nodeRanks is None:
                (trees, forestSize) = scatter_read_trees(dataFilePath, popName, comm=env.comm, io_size=env.IOsize)
            else:
                (trees, forestSize) = scatter_read_trees(dataFilePath, popName, comm=env.comm, io_size=env.IOsize,
                                                         node_rank_map=env.nodeRanks)
            if env.verbose:
                if env.pc.id() == 0:
                    logger.info("*** Done reading trees for population %s" % popName)

            h.numCells = 0
            i=0
            for (gid, tree) in trees:
                if env.verbose:
                    if env.pc.id() == 0:
                        logger.info("*** Creating gid %i" % gid)
            
                verboseflag = 0
                model_cell = cells.make_neurotree_cell(templateClass, neurotree_dict=tree, gid=gid, local_id=i, dataset_path=datasetPath)
                if env.verbose:
                    if (rank == 0) and (i == 0):
                        for sec in list(model_cell.all):
                            h.psection(sec=sec)
                env.gidlist.append(gid)
                env.cells.append(model_cell)
                env.pc.set_gid2node(gid, int(env.pc.id()))
                ## Tell the ParallelContext that this cell is a spike source
                ## for all other hosts. NetCon is temporary.
                nc = model_cell.connect2target(h.nil)
                env.pc.cell(gid, nc, 1)
                ## Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                ## Record voltages from a subset of cells
                if gid in v_sample_set:
                    v_vec = h.Vector()
                    soma = list(model_cell.soma)[0]
                    v_vec.record(soma(0.5)._ref_v)
                    env.v_dict[popName][gid] = v_vec 
                i = i+1
                h.numCells = h.numCells+1
            if env.verbose:
                if env.pc.id() == 0:
                    logger.info("*** Created %i cells" % i)
                    
        elif env.cellAttributeInfo.has_key(popName) and env.cellAttributeInfo[popName].has_key('Coordinates'):
            if env.verbose:
                if env.pc.id() == 0:
                    logger.info("*** Reading coordinates for population %s" % popName)
            
            if env.nodeRanks is None:
                cell_attributes_dict = scatter_read_cell_attributes(dataFilePath, popName, 
                                                                    namespaces=['Coordinates'],
                                                                    comm=env.comm, io_size=env.IOsize)
            else:
                cell_attributes_dict = scatter_read_cell_attributes(dataFilePath, popName, 
                                                                    namespaces=['Coordinates'],
                                                                    node_rank_map=env.nodeRanks,
                                                                    comm=env.comm, io_size=env.IOsize)
            if env.verbose:
                if env.pc.id() == 0:
                    logger.info("*** Done reading coordinates for population %s" % popName)
                    
            coords = cell_attributes_dict['Coordinates']

            h.numCells = 0
            i=0
            for (gid, cell_coords_dict) in coords:
                if env.verbose:
                    if env.pc.id() == 0:
                        logger.info("*** Creating gid %i" % gid)
            
                verboseflag = 0
                model_cell = cells.make_cell(templateClass, gid=gid, local_id=i, dataset_path=datasetPath)

                cell_x = cell_coords_dict['X Coordinate'][0]
                cell_y = cell_coords_dict['Y Coordinate'][0]
                cell_z = cell_coords_dict['Z Coordinate'][0]
                model_cell.position(cell_x, cell_y, cell_z)
                
                env.gidlist.append(gid)
                env.cells.append(model_cell)
                env.pc.set_gid2node(gid, int(env.pc.id()))
                ## Tell the ParallelContext that this cell is a spike source
                ## for all other hosts. NetCon is temporary.
                nc = model_cell.connect2target(h.nil)
                env.pc.cell(gid, nc, 1)
                ## Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                i = i+1
                h.numCells = h.numCells+1
        h.define_shape()
            
             
def mkstim(env):

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)
    
    inputFilePath = os.path.join(datasetPath,env.modelConfig['Cell Data'])

    popNames = env.celltypes.keys()
    popNames.sort()
    for popName in popNames:
        if env.celltypes[popName].has_key('Vector Stimulus'):
            vecstim_namespace = env.celltypes[popName]['Vector Stimulus']

            if env.nodeRanks is None:
              cell_attributes_dict = scatter_read_cell_attributes(inputFilePath, popName, 
                                                                  namespaces=[vecstim_namespace],
                                                                  comm=env.comm, io_size=env.IOsize)
            else:
              cell_attributes_dict = scatter_read_cell_attributes(inputFilePath, popName, 
                                                                  namespaces=[vecstim_namespace],
                                                                  node_rank_map=env.nodeRanks,
                                                                  comm=env.comm, io_size=env.IOsize)
            cell_vecstim = cell_attributes_dict[vecstim_namespace]
            for (gid, vecstim_dict) in cell_vecstim:
              if env.verbose:
                if rank == 0:
                  logger.info("*** Stimulus onset is %g ms" % env.stimulus_onset)
                if len(vecstim_dict['spiketrain']) > 0:
                  logger.info( "*** Spike train for gid %i is of length %i (first spike at %g ms)" % (gid, len(vecstim_dict['spiketrain']),vecstim_dict['spiketrain'][0]))
                else:
                  logger.info("*** Spike train for gid %i is of length %i" % (gid, len(vecstim_dict['spiketrain'])))

              vecstim_dict['spiketrain'] += env.stimulus_onset
              cell = env.pc.gid2cell(gid)
              cell.play(h.Vector(vecstim_dict['spiketrain']))


def init(env):
    h.load_file("nrngui.hoc")
    h.load_file("loadbal.hoc")
    h('objref fi_status, fi_checksimtime, pc, nclist, nc, nil')
    h('strdef datasetPath')
    h('numCells = 0')
    h('totalNumCells = 0')
    h('max_walltime_hrs = 0')
    h('mkcellstime = 0')
    h('mkstimtime = 0')
    h('connectcellstime = 0')
    h('connectgjstime = 0')
    h('results_write_time = 0')
    h.nclist = h.List()
    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)
    h.datasetPath = datasetPath
    ##  new ParallelContext object
    h.pc   = h.ParallelContext()
    env.pc = h.pc
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    ## polymorphic value template
    h.load_file("./templates/Value.hoc")
    ## randomstream template
    h.load_file("./templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file("./templates/StimCell.hoc")
    h.xopen("./lib.hoc")
    h.dt = env.dt
    h.tstop = env.tstop
    if env.optldbal or env.optlptbal:
        lb = h.LoadBalance()
        if not os.path.isfile("mcomplex.dat"):
            lb.ExperimentalMechComplex()

    if (env.pc.id() == 0):
      mkout (env, env.resultsFilePath)
    if (env.pc.id() == 0):
        logger.info("*** Creating cells...")
    env.pc.barrier()
    h.startsw()
    mkcells(env)
    env.mkcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        logger.info("*** Cells created in %g seconds" % env.mkcellstime)
    logger.info("*** Rank %i created %i cells" % (env.pc.id(), len(env.cells)))
    h.startsw()
    mkstim(env)
    env.mkstimtime = h.stopsw()
    if (env.pc.id() == 0):
        logger.info("*** Stimuli created in %g seconds" % env.mkstimtime)
    env.pc.barrier()
    h.startsw()
    connectcells(env)
    env.connectcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        logger.info("*** Connections created in %g seconds" % env.connectcellstime)
    logger.info("*** Rank %i created %i connections" % (env.pc.id(), int(h.nclist.count())))
    h.startsw()
    #connectgjs(env)
    env.connectgjstime = h.stopsw()
    if (env.pc.id() == 0):
        logger.info("*** Gap junctions created in %g seconds" % env.connectgjstime)
    env.pc.setup_transfer()
    env.pc.set_maxstep(10.0)
    h.max_walltime_hrs   = env.max_walltime_hrs
    h.mkcellstime        = env.mkcellstime
    h.mkstimtime         = env.mkstimtime
    h.connectcellstime   = env.connectcellstime
    h.connectgjstime     = env.connectgjstime
    h.results_write_time = env.results_write_time
    env.simtime          = simtime.SimTimeEvent(env.pc, env.max_walltime_hrs, env.results_write_time)
    env.lfp              = lfp.LFP(env.pc, env.celltypes, env.lfpConfig['position'], \
                                   rho=env.lfpConfig['rho'], dt_lfp=env.lfpConfig['dt'], \
                                   fdst=env.lfpConfig['fraction'], maxEDist=env.lfpConfig['maxEDist'], \
                                   seed=int(env.modelConfig['Random Seeds']['Local Field Potential']))
    h.v_init = env.v_init
    h.stdinit()
    h.finitialize(env.v_init)
    env.pc.barrier()
    if env.optldbal or env.optlptbal:
        cx(env)
        ld_bal(env)
        if env.optlptbal:
            lpt_bal(env)

# Run the simulation
def run (env):
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    if (rank == 0):
        logger.info("*** Running simulation")

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if (rank == 0):
        logger.info("*** Simulation completed")
    del(env.cells)
    env.pc.barrier()
    if (rank == 0):
        logger.info("*** Writing spike data")
    spikeout(env, env.resultsFilePath, np.array(env.t_vec, dtype=np.float32), np.array(env.id_vec, dtype=np.uint32))
    if env.vrecordFraction > 0.:
      if (rank == 0):
        logger.info("*** Writing intracellular trace data")
      t_vec = np.arange(0, h.tstop+h.dt, h.dt, dtype=np.float32)
      vout(env, env.resultsFilePath, t_vec, env.v_dict)
    env.pc.barrier()
    if (rank == 0):
        logger.info("*** Writing local field potential data")
        lfpout(env, env.resultsFilePath, env.lfp)

    comptime = env.pc.step_time()
    cwtime   = comptime + env.pc.step_wait()
    maxcw    = env.pc.allreduce(cwtime, 2)
    avgcomp  = env.pc.allreduce(comptime, 1)/nhosts
    maxcomp  = env.pc.allreduce(comptime, 2)

    if (env.pc.id() == 0):
        logger.info("Execution time summary for host %i:" % (int(env.pc.id())))
        logger.info("  created cells in %g seconds" % env.mkcellstime)
        logger.info("  connected cells in %g seconds" % env.connectcellstime)
        logger.info("  created gap junctions in %g seconds" % env.connectgjstime)
        logger.info("  ran simulation in %g seconds" % comptime)
        if (maxcw > 0):
            logger.info("  load balance = %g" % (avgcomp/maxcw))

    env.pc.runworker()
    env.pc.done()
    h.quit()

    
@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-paths", type=str)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-id", type=str, required=False, default='')
@click.option("--node-rank-file", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=1)
@click.option("--coredat", is_flag=True)
@click.option("--vrecord-fraction", type=float, default=0.001)
@click.option("--tstop", type=int, default=1)
@click.option("--v-init", type=float, default=-75.0)
@click.option("--stimulus-onset", type=float, default=1.0)
@click.option("--max-walltime-hours", type=float, default=1.0)
@click.option("--results-write-time", type=float, default=360.0)
@click.option("--dt", type=float, default=0.025)
@click.option("--ldbal", is_flag=True)
@click.option("--lptbal", is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def main(config_file, template_paths, dataset_prefix, results_path, results_id, node_rank_file, io_size, coredat,
         vrecord_fraction, tstop, v_init, stimulus_onset, max_walltime_hours, results_write_time, dt, ldbal, lptbal, verbose):
    """

    :param config_file:
    :param template_paths:
    :param dataset_prefix:
    :param results_path:
    :param results_id:
    :param node_rank_file:
    :param io_size:
    :param coredat:
    :param vrecord_fraction:
    :param tstop:
    :param v_init:
    :param stimulus_onset:
    :param max_walltime_hours:
    :param results_write_time:
    :param dt:
    :param ldbal:
    :param lptbal:
    :param verbose:
    """
    if verbose:
        logger.setLevel(logging.INFO)
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    np.seterr(all='raise')
    env = Env(comm, config_file, 
              template_paths, dataset_prefix, results_path, results_id,
              node_rank_file, io_size,
              vrecord_fraction, coredat, tstop, v_init, stimulus_onset,
              max_walltime_hours, results_write_time,
              dt, ldbal, lptbal, verbose)


    init(env)
    run(env)


if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index(script_name)+1):])

