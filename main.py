##
##  Dentate Gyrus model initialization script
##

import sys, os
import os.path
import click
import itertools
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neurograph.io import scatter_graph, bcast_graph
from neurotrees.io import scatter_read_trees
from env import Env
import lpt

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
    print ("*** expected load balance %.2f" % (sum_cx / nhosts / max_sum_cx))

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

    
    
def connectprj(env, graph, prjname, prjvalue):
    prjType    = prjvalue['type']
    indexType  = prjvalue['index']
    prj        = graph[prjname]

    if (prjType == 'long.+trans.dist'):
        wgtval = prjvalue['weight']
        if isinstance(wgtval,list):
            wgtlst = h.List()
            for val in wgtval:
                hval = h.Value(0, val)
                wgtlst.append(hval)
            h.syn_weight = h.Value(2,wgtlst)
        else:
            h.syn_weight = h.Value(0,wgtval)
        idxval = prjvalue['synIndex']
        if isinstance(idxval,list):
            idxlst = h.List()
            for val in idxval:
                hval = h.Value(0, val)
                idxlst.append(hval)
            h.syn_index = h.Value(2,idxlst)
        else:
            h.syn_index = h.Value(0,idxval)
        velocity = prjvalue['velocity']
        for destination in prj:
            edges  = prj[destination]
            sources = edges[0]
            ldists  = edges[1]
            tdists  = edges[2]
            if indexType == 'absolute':
                for i in range(0,len(sources)):
                        source   = sources[i]
                        distance = ldists[i] + tdists[i]
                        delay    = (distance / velocity) + 1.0
                        h.nc_appendsyn(env.pc, h.nclist, source, destination, h.syn_index, h.syn_weight, delay)
            else:
                raise RuntimeError ("Unsupported index type %s of projection %s" % (indexType, prjname))
    elif (prjType == 'dist'):
        wgtval = prjvalue['weight']
        if isinstance(wgtval,list):
            wgtlst = h.List()
            for val in wgtval:
                hval = h.Value(0, val)
                wgtlst.append(hval)
            h.syn_weight = h.Value(2,wgtlst)
        else:
            h.syn_weight = h.Value(0,wgtval)
        idxval = prjvalue['synIndex']
        if isinstance(idxval,list):
            idxlst = h.List()
            for val in idxval:
                hval = h.Value(0, val)
                idxlst.append(hval)
            h.syn_index = h.Value(2,idxlst)
        else:
            h.syn_index = h.Value(0,idxval)
        velocity = prjvalue['velocity']
        for destination in prj:
            edges  = prj[destination]
            sources = edges[0]
            dists  = edges[1]
            if indexType == 'absolute':
                for i in range(0,len(sources)):
                        source   = sources[i]
                        distance = dists[i]
                        delay    = (distance / velocity) + 1.0
                        h.nc_appendsyn(env.pc, h.nclist, source, destination, h.syn_index, h.syn_weight, delay)
            else:
                raise RuntimeError ("Unsupported index type %s of projection %s" % (indexType, prjname))
    elif (prjType == 'syn'):
        wgtvector = prjvalue['weights']
        h.syn_weight_vector = h.Vector()
        h.syn_weight_vector.from_python(wgtvector)
        h.syn_type = h.Value(0,prjvalue['synType'])
        velocity = prjvalue['velocity']
        for destination in prj:
            edges  = prj[destination]
            sources = edges[0]
            synidxs = edges[1]
            if indexType == 'absolute':
                for i in range(0,len(sources)):
                    source   = sources[i]
                    h.syn_index = h.Value(0,synidxs[i])
                    delay = 1.0
                    h.nc_appendsyn_wgtvector(env.pc, h.nclist, source, destination, h.syn_type, h.syn_index, h.syn_weight_vector, delay)
            else:
                raise RuntimeError ("Unsupported index type %s of projection %s" % (indexType, prjname))
    else:
        raise RuntimeError ("Unsupported projection type %s of projection %s" % (prjType, prjname))
                       
    del graph[prjname]

def connectcells(env):
    h('objref syn_type, syn_index, syn_weight, syn_weight_vector')
    projections = env.projections
    if env.verbose:
        if env.pc.id() == 0:
            print projections
    datasetPath = os.path.join(env.datasetPrefix,env.datasetName)
    connectivityFilePath = os.path.join(datasetPath,env.connectivityFile)
    if env.nodeRanks is None:
        (graph, a) = scatter_graph(MPI._addressof(env.comm),connectivityFilePath,env.IOsize,attributes=True)
    else:
        (graph, a) = scatter_graph(MPI._addressof(env.comm),connectivityFilePath,env.IOsize,node_rank_map=env.nodeRanks,attributes=True)
    for name in projections.keys():
        if env.verbose:
            if env.pc.id() == 0:
                print "*** Creating projection %s" % name
        connectprj(env, graph, name, projections[name])
    
def mksyn1(cell,synapses,env):
    for synkey in synapses.keys():
        synval = synapses[synkey]
        synorder = env.synapseOrder[synkey]
        location = synval['location']
        t_rise   = synval['t_rise']
        t_decay  = synval['t_decay']
        e_rev    = synval['e_rev']
        if location == 'soma':
            cell.soma.push()
            h.syn = h.Exp2Syn(0.5)
            h.syn.tau1 = t_rise
            h.syn.tau2 = t_decay
            h.syn.e    = e_rev
    	    cell.syns.o(synorder).append(h.syn)
            h.pop_section()
        elif location.has_key('dendrite'):
            for dendindex in location['dendrite']:
                if isinstance(location['compartment'],list):
                    compartments=location['compartment']
                else:
                    compartments=[location['compartment']]
                for secindex in compartments:
                    cell.dendrites[dendindex][secindex].push()
                    h.syn = h.Exp2Syn(0.5)
                    h.syn.tau1 = t_rise
                    h.syn.tau2 = t_decay
                    h.syn.e    = e_rev
                    cell.syns.o(synorder).append(h.syn)
                    h.pop_section()

def mksyn2(cell,syn_ids,syn_types,swc_types,syn_locs,syn_sections,synapses,env):
    for (syn_id,syn_type,swc_type,syn_loc,syn_section) in itertools.izip(syn_ids,syn_types,swc_types,syn_locs,syn_sections):
        if (syn_type == 0) & (swc_type == 5):
            cell.alldendritesList[syn_section].root.push()
            h.syn      = h.Exp2Syn(syn_loc)
            h.syn.tau1 = synapses[syn_type]['t_rise']
            h.syn.tau2 = synapses[syn_type]['t_decay']
            h.syn.e    = synapses[syn_type]['e_rev']
            cell.allsyns.o(syn_type).append(h.syn)
            h.pop_section()
            cell.alldendritesList[syn_section].sec(syn_loc).count_spines += 1

def mksyn3(cell,syn_ids,syn_types,syn_locs,syn_sections,synapses,env):
    for (syn_id,syn_type,syn_loc,syn_section) in itertools.izip(syn_ids,syn_types,syn_locs,syn_sections):
        cell.alldendritesList[syn_section].root.push()
        h.syn      = h.Exp2Syn(syn_loc)
        h.syn.tau1 = synapses[syn_type]['t_rise']
        h.syn.tau2 = synapses[syn_type]['t_decay']
        h.syn.e    = synapses[syn_type]['e_rev']
        cell.allsyns.o(syn_type).append(h.syn)
        h.pop_section()
        cell.alldendritesList[syn_section].sec(syn_loc).count_spines += 1

    

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
        if env.verbose:
            if env.pc.id() == 0:
                print gapjunctions
        datasetPath = os.path.join(env.datasetPrefix,env.datasetName)
        (graph, a) = bcast_graph(MPI._addressof(env.comm),gapjunctionsFilePath,attributes=True)

        ggid = 2e6
        for name in gapjunctions.keys():
            if env.verbose:
                if env.pc.id() == 0:
                    print "*** Creating gap junctions %s" % name
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
                        h.mkgap(env.pc, h.gjlist, source, srcbranch, srcsec, ggid, ggid+1, weight)
                    if env.pc.gid_exists(destination):
                        h.mkgap(env.pc, h.gjlist, destination, dstbranch, dstsec, ggid+1, ggid, weight)
                    ggid = ggid+2

            del graph[name]
 
def mkcells(env):

    h('objref templatePaths, templatePathValue, cell, syn, syn_ids, syn_types, swc_types, syn_locs, syn_sections')

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePathValue = h.Value(1,path)
        h.templatePaths.append(h.templatePathValue)
    popNames = env.celltypes.keys()
    popNames.sort()
    for popName in popNames:
        templateName = env.celltypes[popName]['templateName']
        h.find_template(env.pc, h.templatePaths, templateName)

    for popName in popNames:

        if env.verbose:
            if env.pc.id() == 0:
                print "*** Creating population %s" % popName
        
        templateName = env.celltypes[popName]['templateName']
        if env.celltypes[popName].has_key('synapses'):
            synapses = env.celltypes[popName]['synapses']
        else:
            synapses = {}

        i=0
        if env.celltypes[popName]['templateType'] == 'single':
            numCells  = env.celltypes[popName]['num']
            h.numCells = numCells
            h.totalNumCells = h.totalNumCells + numCells
            index  = env.celltypes[popName]['index']

            mygidlist = []
            if env.nodeRanks is None:
                for x in index:
                    if (x % nhosts) == rank:
                        mygidlist.append(x)
            else:
                for x in index:
                    if env.nodeRanks[x] == rank:
                        mygidlist.append(x)

            if env.verbose:
                print "Population %s, rank %d: " % (popName, env.pc.id())
                print mygidlist
        
            env.gidlist.extend(mygidlist)
            for gid in mygidlist:
                hstmt = 'cell = new %s(%d, %d, "%s")' % (templateName, i, gid, datasetPath)
                h(hstmt)

                mksyn1(h.cell,synapses,env)
                
                env.cells.append(h.cell)
                env.pc.set_gid2node(gid, int(env.pc.id()))
                
                ## Tell the ParallelContext that this cell is a spike source
                ## for all other hosts. NetCon is temporary.
                if (h.cell.is_art()):
                    nc = h.NetCon(h.cell.pp, h.nil)
                else:
                    nc = h.cell.connect2target(h.nil)
                env.pc.cell(gid, nc, 1)
                nc = None
                ## Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                i = i+1
                
        elif env.celltypes[popName]['templateType'] == 'forest':
            h('objref vx, vy, vz, vradius, vsection, vlayer, vsection, vsrc, vdst, secnodes')
            h('gid = fid = node = 0')
            inputFilePath = os.path.join(datasetPath,env.celltypes[popName]['forestFile'])
            if env.nodeRanks is None:
                (trees, forestSize) = scatter_read_trees(MPI._addressof(env.comm), inputFilePath, popName, env.IOsize,
                                                        attributes=True, namespace='Synapse_Attributes')
            else:
                (trees, forestSize) = scatter_read_trees(MPI._addressof(env.comm), inputFilePath, popName, env.IOsize,
                                                        attributes=True, namespace='Synapse_Attributes',
                                                        node_rank_map=env.nodeRanks)
            if env.celltypes[popName].has_key('synapses'):
                synapses = env.celltypes[popName]['synapses']
            else:
                synapses = {}
            mygidlist = trees.keys()
            numCells = len(mygidlist)
            h.numCells = numCells
            i=0
            for gid in mygidlist:
                tree       = trees[gid]
                h.gid      = gid
                h.fid      = int(i%forestSize)
                h.vx       = tree['x']
                h.vy       = tree['y']
                h.vz       = tree['z']
                h.vradius  = tree['radius']
                h.vlayer   = tree['layer']
                h.vsection = tree['section']
                h.secnodes = tree['section_topology']['nodes']
                h.vsrc     = tree['section_topology']['src']
                h.vdst     = tree['section_topology']['dst']
                ## syn_id syn_type syn_locs section layer
                h.syn_ids      = tree['Synapse_Attributes']['syn_id']
                h.syn_types    = tree['Synapse_Attributes']['syn_type']
                if tree.has_key('Synapse_Attributes.swc_type'):
                    h.swc_types    = tree['Synapse_Attributes']['swc_type']
                h.syn_locs     = tree['Synapse_Attributes']['syn_locs']
                h.syn_sections = tree['Synapse_Attributes']['section']
                verboseflag = 0
                hstmt = 'cell = new %s(fid, gid, numCells, "", 0, vlayer, vsrc, vdst, secnodes, vx, vy, vz, vradius, %d)' % (templateName, verboseflag)
                h(hstmt)
                if h.swc_types is None:
                    mksyn3(h.cell,h.syn_ids,h.syn_types,h.syn_locs,h.syn_sections,synapses,env)
                else:
                    mksyn2(h.cell,h.syn_ids,h.syn_types,h.swc_types,h.syn_locs,h.syn_sections,synapses,env)
                h.cell.correct_for_spines()
                env.gidlist.append(gid)
                env.cells.append(h.cell)
                env.pc.set_gid2node(gid, int(env.pc.id()))
                ## Tell the ParallelContext that this cell is a spike source
                ## for all other hosts. NetCon is temporary.
                nc = h.cell.connect2target(h.nil)
                env.pc.cell(gid, nc, 1)
                ## Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                i = i+1
        else:
             error ("Unsupported template type %s" % (env.celltypes[popName]['templateType']))

def mkstim(env):

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)
    
    h('objref vecstim_index')

    popNames = env.celltypes.keys()
    popNames.sort()
    for popName in popNames:
        if env.celltypes[popName].has_key('vectorStimulus'):
            vecstim   = env.celltypes[popName]['vectorStimulus']
            spikeFile = os.path.join(datasetPath, vecstim['spikeFile'])
            indexFile = os.path.join(datasetPath, vecstim['indexFile'])
            if vecstim.has_key('activeFile'):
                activeFile = os.path.join(datasetPath, vecstim['activeFile'])
            else:
                activeFile = ""
            index     = env.celltypes[popName]['index']
            h.vecstim_index = h.Vector()
            for x in index:
                h.vecstim_index.append(x)
            h.loadVectorStimulation(env.pc, h.vecstim_index, indexFile, spikeFile, activeFile)
            h.vecstim_index.resize(0)
            

            
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
    h.startsw()
    mkcells(env)
    env.mkcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        print "*** Cells created in %g seconds" % env.mkcellstime
    h.startsw()
    mkstim(env)
    env.mkstimtime = h.stopsw()
    if (env.pc.id() == 0):
        print "*** Stimuli created in %g seconds" % env.mkstimtime
    h.startsw()
    connectcells(env)
    env.connectcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        print "*** Cells connected in %g seconds" % env.connectcellstime
    h.startsw()
    connectgjs(env)
    env.connectgjstime = h.stopsw()
    if (env.pc.id() == 0):
        print "*** Gap junctions created in %g seconds" % env.connectgjstime
    env.pc.setup_transfer()
    env.pc.set_maxstep(10.0)
    h.max_walltime_hrs   = env.max_walltime_hrs
    h.mkcellstime        = env.mkcellstime
    h.mkstimtime         = env.mkstimtime
    h.connectcellstime   = env.connectcellstime
    h.connectgjstime     = env.connectgjstime
    h.results_write_time = env.results_write_time
    h.fi_checksimtime    = h.FInitializeHandler("checksimtime(pc)")
    if (env.pc.id() == 0):
        print "dt = %g" % h.dt
        print "tstop = %g" % h.tstop
        h.fi_status = h.FInitializeHandler("simstatus()")
    h.stdinit()
    env.pc.barrier()
    if env.optldbal or env.optlptbal:
        cx(env)
        ld_bal(env)
        if env.optlptbal:
            lpt_bal(env)

# Run the simulation
def run (env):
    h('objref vcnts, t_vec_all, id_vec_all')

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if (rank == 0):
        print "*** Simulation completed"
    h.vcnts      = h.Vector(nhosts)
    h.vcnts.x[0]   = env.t_vec.size()
    h.t_vec_all  = h.Vector()
    h.id_vec_all = h.Vector()
    env.pc.alltoall(env.t_vec, h.vcnts, h.t_vec_all)
    env.pc.alltoall(env.id_vec, h.vcnts, h.id_vec_all)
    
    if (rank == 0):
        h.spikeout("%s/%s_spikeout.dat" % (env.resultsPath, env.modelName),h.t_vec_all,h.id_vec_all)
    #if (env.vrecordFraction > 0):
    #    h.vrecordout("%s/%s_vrecord_%d.dat" % (env.resultsPath, env.modelName, env.pc.id(), env.indicesVrecord))

    comptime = env.pc.step_time()
    cwtime   = comptime + env.pc.step_wait()
    maxcw    = env.pc.allreduce(cwtime, 2)
    avgcomp  = env.pc.allreduce(comptime, 1)/nhosts
    maxcomp  = env.pc.allreduce(comptime, 2)

    if (env.pc.id() == 0):
        print "Execution time summary for host 0:"
        print "  created cells in %g seconds" % env.mkcellstime
        print "  connected cells in %g seconds" % env.connectcellstime
        print "  created gap junctions in %g seconds" % env.connectgjstime
        print "  ran simulation in %g seconds" % comptime
        if (maxcw > 0):
            print "  load balance = %g" % (avgcomp/maxcw)

    env.pc.runworker()
    env.pc.done()
    h.quit()

    
@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-paths", type=str)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--node-rank-file", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=1)
@click.option("--coredat", is_flag=True)
@click.option("--vrecord-fraction", type=float, default=0.0)
@click.option("--tstop", type=int, default=1)
@click.option("--v-init", type=float, default=-75.0)
@click.option("--max-walltime-hours", type=float, default=1.0)
@click.option("--results-write-time", type=float, default=30.0)
@click.option("--dt", type=float, default=0.025)
@click.option("--ldbal", is_flag=True)
@click.option("--lptbal", is_flag=True)
@click.option('--verbose', is_flag=True)
def main(config_file, template_paths, dataset_prefix, results_path, node_rank_file, io_size, coredat, vrecord_fraction, tstop, v_init, max_walltime_hours, results_write_time, dt, ldbal, lptbal, verbose):
    np.seterr(all='raise')
    env = Env(MPI.COMM_WORLD, config_file, template_paths, dataset_prefix, results_path, node_rank_file, io_size, vrecord_fraction, coredat, tstop, v_init, max_walltime_hours, results_write_time, dt, ldbal, lptbal, verbose)
    init(env)
    run(env)

if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index("main.py")+1):])

