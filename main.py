##
##  Dentate Gyrus model initialization script
##

import sys, os
import click
import itertools
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neurograph.io import scatter_graph
from neurotrees.io import scatter_read_trees
from env import Env

        
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
            h.synWeight = h.Value(2,wgtlst)
        else:
            h.synWeight = h.Value(0,wgtval)
        idxval = prjvalue['synIndex']
        if isinstance(idxval,list):
            idxlst = h.List()
            for val in idxval:
                hval = h.Value(0, val)
                idxlst.append(hval)
            h.synIndex = h.Value(2,idxlst)
        else:
            h.synIndex = h.Value(0,idxval)
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
                    delay = (distance / velocity) + 1.0
                    h.nc_appendsyn(env.pc, h.nclist, source, destination, h.synIndex, h.synWeight, delay)
            else:
                raise RuntimeError ("Unsupported index type %s of projection %s" % (indexType, prjname))
    elif (prjType == 'syn'):
        wgtvector = prjvalue['weights']
        h.synWeightVector = h.Vector()
        h.synWeightVector.from_python(wgtvector)
        h.synType = h.Value(0,prjvalue['synType'])
        velocity = prjvalue['velocity']
        for destination in prj:
            edges  = prj[destination]
            sources = edges[0]
            synidxs = edges[1]
            if indexType == 'absolute':
                for i in range(0,len(sources)):
                    source   = sources[i]
                    h.synIndex = h.Value(0,synidxs[i])
                    delay = 1.0
                    h.nc_appendsyn_wgtvector(env.pc, h.nclist, source, destination, h.synType, h.synIndex, h.synWeightVector, delay)
            else:
                raise RuntimeError ("Unsupported index type %s of projection %s" % (indexType, prjname))
    else:
        raise RuntimeError ("Unsupported projection type %s of projection %s" % (prjType, prjname))
                       
    del graph[prjname]

def connectcells(env):
    h('objref synType, synIndex, synWeight, synWeightVector')
    projections = env.projections
    if env.verbose:
        if env.pc.id() == 0:
            print projections
    datasetPath = os.path.join(env.datasetPrefix,env.datasetName)
    connectivityFilePath = os.path.join(datasetPath,env.connectivityFile)
    graph = scatter_graph(MPI._addressof(env.comm),connectivityFilePath,env.IOsize)
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
        if swc_type == 5:
            cell.alldendritesList[syn_section].root.push()
            h.syn      = h.Exp2Syn(syn_loc)
            h.syn.tau1 = synapses[syn_type]['t_rise']
            h.syn.tau2 = synapses[syn_type]['t_decay']
            h.syn.e    = synapses[syn_type]['e_rev']
            cell.allsyns.o(syn_type).append(h.syn)
            h.pop_section()

def mksyn3(cell,syn_ids,syn_types,syn_locs,syn_sections,synapses,env):
    for (syn_id,syn_type,syn_loc,syn_section) in itertools.izip(syn_ids,syn_types,syn_locs,syn_sections):
        cell.alldendritesList[syn_section].root.push()
        h.syn      = h.Exp2Syn(syn_loc)
        h.syn.tau1 = synapses[syn_type]['t_rise']
        h.syn.tau2 = synapses[syn_type]['t_decay']
        h.syn.e    = synapses[syn_type]['e_rev']
        cell.allsyns.o(syn_type).append(h.syn)
        h.pop_section()

    
def mkcells(env):

    h('objref templatePaths, templatePathValue, cell, syn, syn_ids, syn_types, swc_types, syn_locs, syn_sections')
    h('numCells = 0')

    hostid = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    h('strdef datasetPath')
    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)
    h.datasetPath = datasetPath

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePathValue = h.Value(1,path)
        h.templatePaths.append(h.templatePathValue)
    popNames = env.celltypes.keys()
    popNames.sort()
    for popName in popNames:
        templateName = env.celltypes[popName]['templateName']
        h.find_template(h.templatePaths, templateName)

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
            index  = env.celltypes[popName]['index']

            mygidlist = []
            for x in index:
                if (x % nhosts) == hostid:
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
                ## Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                i = i+1
                
        elif env.celltypes[popName]['templateType'] == 'forest':
            h('objref vx, vy, vz, vradius, vsection, vlayer, vsection, vsrc, vdst, secnodes')
            h('gid = fid = node = 0')
            inputFilePath = os.path.join(datasetPath,env.celltypes[popName]['forestFile'])
            (trees, forestSize) = scatter_read_trees(MPI._addressof(env.comm), inputFilePath, popName, env.IOsize,
                                                     attributes=True, namespace='Synapse_Attributes')
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
                h.syn_ids      = tree['Synapse_Attributes.syn_id']
                h.syn_types    = tree['Synapse_Attributes.syn_type']
                if tree.has_key('Synapse_Attributes.swc_type'):
                    h.swc_types    = tree['Synapse_Attributes.swc_type']
                h.syn_locs     = tree['Synapse_Attributes.syn_locs']
                h.syn_sections = tree['Synapse_Attributes.section']
                verboseflag = 0
                hstmt = 'cell = new %s(fid, gid, numCells, "", 0, vlayer, vsrc, vdst, secnodes, vx, vy, vz, vradius, %d)' % (templateName, verboseflag)
                h(hstmt)
                if h.swc_types == h.nil:
                    mksyn3(h.cell,h.syn_ids,h.syn_types,h.syn_locs,h.syn_sections,synapses,env)
                else:
                    mksyn2(h.cell,h.syn_ids,h.syn_types,h.swc_types,h.syn_locs,h.syn_sections,synapses,env)
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

                    
def init(env):

    h.load_file("nrngui.hoc")
    h('objref pc, nclist, nc, nil')
    h.nclist = h.List()
    ##  new ParallelContext object
    h.pc = h.ParallelContext()
    env.pc = h.pc
    ## polymorphic value template
    h.load_file("./templates/Value.hoc")
    ## randomstream template
    h.load_file("./templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file("./templates/StimCell.hoc")
    h.xopen("./lib.hoc")
    h.startsw()
    mkcells(env)
    env.mkcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        print "*** Cells created in %g seconds" % env.mkcellstime
    h.startsw()
    if not env.cells_only:
        connectcells(env)
    env.connectcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        print "*** Cells connected in %g seconds" % env.connectcellstime

# Run the simulation
def run (env):
    env.pc.psolve(env.tstop)
    h.spikeout("%s/%s_spikeout_%d.dat" % (env.resultsPath, env.modelName, env.pc.id()),env.t_vec,env.id_vec)
    #if (env.vrecordFraction > 0):
    #    h.vrecordout("%s/%s_vrecord_%d.dat" % (env.resultsPath, env.modelName, env.pc.id(), env.indicesVrecord))

    comptime = env.pc.step_time
    avgcomp  = env.pc.allreduce(comptime, 1)/env.pc.nhost()
    maxcomp  = env.pc.allreduce(comptime, 2)

    if (env.pc.id() == 0):
        print "Execution time summary for host 0:"
        print "  created cells in %g seconds" % env.mkcellstime
        print "  connected cells in %g seconds\n" % connectcellstime
        #print "  created gap junctions in %g seconds\n" % connectgjstime
        print "  ran simulation in %g seconds\n" % env.comptime
        if (maxcomp > 0):
            print "  load balance = %g\n" % (avgcomp/maxcomp)

    env.pc.done()
    h.quit()

    
@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-paths", type=str)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--io-size", type=int, default=1)
@click.option("--coredat", is_flag=True)
@click.option("--vrecord-fraction", type=float, default=0.0)
@click.option("--tstop", type=int, default=1)
@click.option("--v-init", type=float, default=-75.0)
@click.option("--max-walltime-hours", type=float, default=1.0)
@click.option("--results-write-time", type=float, default=30.0)
@click.option("--dt", type=float, default=0.025)
@click.option("--cells-only", is_flag=True)
@click.option('--verbose', is_flag=True)
def main(config_file, template_paths, dataset_prefix, results_path, io_size, coredat, vrecord_fraction, tstop, v_init, max_walltime_hours, results_write_time, cells_only, dt, verbose):
    env = Env(MPI.COMM_WORLD, config_file, template_paths, dataset_prefix, results_path, io_size, vrecord_fraction, coredat, tstop, v_init, max_walltime_hours, results_write_time, dt, cells_only, verbose)
    init(env)
    run(env)

if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index("main.py")+1):])
