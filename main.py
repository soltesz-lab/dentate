##
##  Dentate Gyrus model initialization script
##

import sys, os
import click
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
#from neurograph.io import scatter_graph
from neurotrees.io import scatter_read_trees
from env import Env

def prj_offsets(order):
    if (order == 2):
        return lambda (presynapticOffset,postsynapticOffset,dst,sources): (sources+presynapticOffset, dst)
    elif (order == 1):
        # absolute numbering of pre/post synaptic cells
        return lambda (presynapticOffset,postsynapticOffset,dst,sources): (dst, sources)
    elif (order == 0):
        # relative numbering of pre/post synaptic cells -- add the respective offsets
        return lambda (presynapticOffset,postsynapticOffset,sources,dst): (sources+presynapticOffset, dst+postsynapticOffset)
    else:
        error ("unknown connectivity order type")

        
def connect_h5prj(pnm, connectivityType, synType, presynapticSize, presynapticOffset, postsynapticSize, postsynapticOffset, prjname):
    
    order   = connectivityType.getPropertyScalar("order")
    synType = connectivityType.getPropertyObject("synType")
    wdType  = connectivityType.getPropertyScalar("wdType")

    h5prj = H5Graph[prjname]

    f_offsets = prj_offsets(order)
    
    if (len(h5prj.keys()) > 0):

        if (wdType == 1):
            for dst in h5prj:
                edges   = h5prj[dst]
                sources = edges[0]
                weights = edges[1]
                delays  = edges[2]
                dst1,sources1 = f_offsets(presynapticOffset,postsynapticOffset,dst,sources)
                for i in range(0,len(sources)):
                    h.nc_appendsyn(pnm, sources1[i], dst1, synType, weights[i], delays[i])

        elif (wdType == 2):
            weight = connectivityType.getPropertyObject("standardWeight")
            velocity = connectivityType.getPropertyScalar("standardVelocity")
               
            for dst in h5prj:

                edges   = h5prj[dst]
                sources = edges[0]
                ldist   = edges[1]
                tdist   = edges[2]
                
                dist  = ldist + tdist
                delay = (dist / velocity) + 1.0
                    
                dst1,sources1 = f_offsets(presynapticOffset,postsynapticOffset,dst,sources)
                
                for i in range(0,len(sources)):
                    h.nc_appendsyn(pnm, sources1[i], dst1, synType, weights[i], delays[i])
               
        elif (wdType == 3):
            
            weights = connectivityType.getPropertyObject("weightHistogram")
            wscale = connectivityType.getPropertyScalar("weightHistogramScale")
            w.v.mul(wscale)
            velocity = connectivityType.getPropertyScalar("standardVelocity")
            
            for dst in h5prj:

                edges     = h5prj[dst]
                sources   = edges[0]
                distances = edges[1]
                layers    = edges[2]
                sections  = edges[3]
                nodes     = edges[4]

                delays = (distances / velocity) + 1.0

                dst1,sources1 = f_offsets(presynapticOffset,postsynapticOffset,dst,sources)

                for i in range(0,len(sources)):
                    h.nc_appendsyn_lsn(pnm, sources1[i], dst1, synType, weights[i], delays[i], layers[i], sections[i], nodes[i])

        elif (wdType == 4):
               
            w = connectivityType.getPropertyObject("standardWeight")
            velocity = connectivityType.getPropertyScalar("standardVelocity")
               
            for dst in h5prj:
                
                edges     = h5prj[dst]
                sources   = edges[0]
                distances = edges[1]
                
                delays = (distances / velocity) + 1.0
                    
                dst1,sources1 = f_offsets(presynapticOffset,postsynapticOffset,dst,sources)
                
                for i in range(0,len(sources)):
                    h.nc_appendsyn(pnm, sources1[i], dst1, synType, weights[i], delays[i])
                       
    del H5Graph[prjname]

def mksyn(cell,synapses,env):
    for synkey in synapses.keys():
        synval = synapses[synkey]
        synorder = env.synapseOrder[synkey]
        location = synval['location']
        t_rise   = synval['t_rise']
        t_decay  = synval['t_decay']
        e_rev    = synval['e_rev']
        if location == 'soma':
            cell.soma.push()
            h('syn = new Exp2Syn(0.5)')
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
                    h('syn = new Exp2Syn(0.5)')
                    h.syn.tau1 = t_rise
                    h.syn.tau2 = t_decay
                    h.syn.e    = e_rev
    		    cell.syns.o(synorder).append(h.syn)
                    h.pop_section()

    
def mkcells(env):

    h('objref templatePaths, templatePathValue, cell, syn, syn_type, syn_locs, syn_sections, nc, nil')
    h('numCells = 0')

    h('strdef datasetPath')
    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)
    h.datasetPath = datasetPath

    h('templatePaths = new List()')
    for path in env.templatePaths:
        h('templatePathValue = new Value(1,"%s")' % path)
        h.templatePaths.append(h.templatePathValue)
    popNames = env.celltypes.keys()
    popNames.sort()
    for popName in popNames:

        if env.verbose:
            print "*** Creating population %s" % popName
        
        templateName = env.celltypes[popName]['templateName']
        if env.celltypes[popName].has_key('synapses'):
            synapses = env.celltypes[popName]['synapses']
        else:
            synapses = {}

        h.find_template(h.templatePaths, templateName)
        ## Round-robin counting.
        ## Each host as an id from 0 to pc.nhost() - 1.
        i=0
        if env.celltypes[popName]['templateType'] == 'single':
            numCells  = env.celltypes[popName]['num']
            h.numCells = numCells
            index = env.celltypes[popName]['index']
            mygidlist = []
            for i in range(int(env.pc.id()), numCells, int(env.pc.nhost())):
                mygidlist.append(index[i])
            env.gidlist.extend(mygidlist)
            for gid in mygidlist:
                hstmt = 'cell = new %s(%d, %d, "%s")' % (templateName, i, gid, datasetPath)
                
                if env.verbose:
                    print hstmt
                h(hstmt)

                mksyn(h.cell,synapses,env)
                
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
                h.syn_type = tree['Synapse_Attributes.syn_type']
                h.syn_locs = tree['Synapse_Attributes.syn_locs']
                h.syn_sections = tree['Synapse_Attributes.section']
                verboseflag = 0
                if env.verbose:
                    verboseflag = 1
                hstmt = 'cell = new %s(fid, gid, numCells, "", 0, vlayer, vsection, vsrc, vdst, secnodes, vx, vy, vz, vradius, %d)' % (templateName, verboseflag)
                if env.verbose: print hstmt
                h(hstmt)
                env.gidlist.append(gid)
                i = i+1
        else:
             error ("Unsupported template type %s" % (env.celltypes[popName]['templateType']))

                    
def init(env):

    h.load_file("nrngui.hoc")
    h('objref pc')
    ##  new ParallelContext object
    h.pc = h.ParallelContext()
    env.pc = h.pc
    ## polymorphic value template
    h.load_file("./templates/Value.hoc")
    ## randomstream template
    h.load_file("./templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file("./templates/StimCell.hoc")
    h.load_file("./lib.hoc")
    h.startsw()
    mkcells(env)
    env.mkcellstime = h.stopsw()
    print "*** Cells created in %g seconds" % env.mkcellstime

# Run the simulation
def run (env):
    env.pc.psolve(env.tstop)
    h.spikeout("%s/%s_spikeout_%d.dat" % (env.resultsPath, env.modelName, env.pc.id()),env.t_vec,env.idvec)
    #if (env.vrecordFraction > 0):
    #    h.vrecordout("%s/%s_vrecord_%d.dat" % (env.resultsPath, env.modelName, env.pc.id(),env.indicesVrecord))

    comptime = env.pc.step_time
    avgcomp  = env.pc.allreduce(comptime, 1)/int(env.pc.nhost())
    maxcomp  = env.pc.allreduce(comptime, 2)

    if (env.pc.id() == 0):
        print "Execution time summary for host 0:"
        print "  created cells in %g seconds" % env.mkcellstime
        #print "  connected cells in %g seconds\n" % connectcellstime
        #print "  created gap junctions in %g seconds\n" % connectgjstime
        print "  ran simulation in %g seconds\n" % env.comptime
        if (maxcomp > 0):
            print "  load balance = %g\n" % (avgcomp/maxcomp)

    env.pc.done()
    h.quit()

    
@click.command()
@click.option("--model-name", required=True)
@click.option("--template-paths", type=str)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--dataset-name", required=True)
@click.option("--celltypes-filename", default="celltypes.yaml")
@click.option("--connectivity-filename", default="connectivity.yaml")
@click.option("--gapjunctions-filename", default="gapjunctions.yaml")
@click.option("--results-path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--io-size", type=int, default=1)
@click.option("--coredat", is_flag=True)
@click.option("--vrecord-fraction", type=float, default=0.0)
@click.option("--tstop", type=int, default=1)
@click.option("--v-init", type=float, default=-75.0)
@click.option("--max-walltime-hours", type=float, default=1.0)
@click.option("--results-write-time", type=float, default=30.0)
@click.option("--dt", type=float, default=0.025)
@click.option('--verbose', is_flag=True)
def main(model_name, template_paths, dataset_prefix, dataset_name, celltypes_filename, connectivity_filename, gapjunctions_filename, results_path,
         io_size, coredat, vrecord_fraction, tstop, v_init, max_walltime_hours, results_write_time, dt, verbose):
    env = Env(MPI.COMM_WORLD,
              model_name, template_paths, dataset_prefix, dataset_name, celltypes_filename, connectivity_filename, gapjunctions_filename, results_path,
              io_size, vrecord_fraction, coredat, tstop, v_init, max_walltime_hours, results_write_time, dt, verbose)
    init(env)
    run(env)

if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index("main.py")+1):])
