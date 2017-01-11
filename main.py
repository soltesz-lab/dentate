##
##  Dentate Gyrus model initialization script
##

import sys, os
import click
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neurograph.io import scatter_graph
from neurotrees.io import scatter_read_trees


def prj_offsets(order) =
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

def create_cells(self,env):

    h('objref cell')
    h('strdef datasetPath')
    for popName in env.celltypes.keys():

        templateName = env.celltypes[popName]['templateName']
        numCells     = env.celltypes[popName]['num']
        offset       = env.celltypes[popName]['offset']
        datasetPath  = env.datasetPath

        h('numCells') = numCells
        h('datasetPath') = datasetPath
        
        ## Round-robin counting.
        ## Each host as an id from 0 to pc.nhost() - 1.
        for i in range(int(env.pc.id()), numCells, int(env.pc.nhost())):
            env.gidlist.append(offset+i)

        i=0
        if env.celltypes[popName]['templateName'] == 'single':
            for gid in env.gidlist:
                hstmt = 'cell = new %s(%d, %d, "%s/%s/")' % (templateName, i, gid, env.datasetPath, k)
                if env.verbose:
                    print hstmt
                cell = h(hstmt)
                
                env.cells.append(cell)
                
                env.pc.set_gid2node(i, int(env.pc.id()))
                
                ## Tell the ParallelContext that this cell is a spike source
                ## for all other hosts. NetCon is temporary.
                nc = cell.connect2target(None)
                env.pc.cell(gid, nc) # Associate the cell with this host and gid
                
                ## Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                i = i+1
                
        elif env.celltypes[popName]['templateType'] == 'forest':
            h('objref vx, vy, vz, vradius, vsection, vlayer, vnodes, vsrc, vdst')
            h('gid = fid ndendpts = node = 0')
            inputFile = env.celltypes[popName]['forestFile']
            (trees, forestSize) = scatter_read_trees(MPI._addressof(env.comm), inputFile, popName, env.IOsize)
            i=0
            for (gid, tree) in trees:
                h.gid      = gid
                h.fid      = int(i%forestSize)
                h.ndendpts = tree['x'].size
                h.vx       = tree['x']
                h.vy       = tree['y']
                h.vz       = tree['z']
                h.vradius  = tree['radius']
                h.vlayer   = tree['layer']
                h.vnodes   = tree['section_topology']['nodes']
                h.vsrc     = tree['section_topology']['src']
                h.vdst     = tree['section_topology']['dst']
                hstmt = 'cell = new %s(fid, gid, numCells, "", 0, ndendpts, vx, vy, vz, vradius, vlayer, vnodes, vsrc, vdst)' % (templateName)
                if env.verbose: print hstmt
                h(hstmt)
                i = i+1
        else:
             error "Unsupported template type %s" % env.celltypes[popName]['templateType']
    
                    
def init(env):

    ##  new ParallelContext object
    env.pc = h.ParallelContext()
    
    ## randomstream template
    h.load_file("./templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file("./templates/StimCell.hoc")
    ## helper functions
    h.load_file("lib.hoc")
    create_cells(env)

    

# Run the simulation
def run ():
    h.load_file("run.hoc")
    h.pc.done()
    h.quit()

    
@cli.command()
@click.option("--model-name", required=True)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--dataset-name", required=True)
@click.option("--celltypes-filename", default="celltypes.yaml")
@click.option("--connectivity-filename", default="connectivity.yaml")
@click.option("--gapjunctions-filename", default="gapjunctions.yaml")
@click.option("--results-path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--io-size", type=int, default=64)
@click.option("--coredat", is_flag=True)
@click.option("--vrecord-fraction", type=float, default=0.0)
@click.option("--tstop", , type=int, default=1)
@click.option("--v-init", , type=float, default=-75.0)
@click.option("--max-walltime-hours", , type=float, default=1.0)
@click.option("--results-write-time", , type=float, default=30.0)
@click.option("--dt", , type=float, default=0.025)
@click.option('--verbose', is_flag=True)
def main(model_name, dataset_prefix, dataset_name, celltypes_filename, connectivity_filename, gapjunctions_filename, results_path,
             io_size, vrecord_fraction, coredat, tstop, v_init, max_walltime_hrs, results_write_time, dt, verbose):
    env = Env(MPI.COMM_WORLD,
              model_name, dataset_prefix, dataset_name, celltypes_filename, connectivity_filename, gapjunctions_filename, results_path,
              io_size, vrecord_fraction, coredat, tstop, v_init, max_walltime_hrs, results_write_time, dt, verbose)
    init(env)
    run()


