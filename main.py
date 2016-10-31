##
##  Dentate Gyrus model initialization script
##

import sys, os
import click
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neurograph.reader import scatter_graph

H5Graph = {}

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

                    
connectprj = lambda (): connecth5prj()

def init(nrnpath, parameters, results_path, io_size=64, verbose=False):

    comm = MPI.COMM_WORLD

    h('strdef parameters')
    h('parameters = "%s"', parameters)
    h('strdef resultsPath')
    h('resultsPath = "%s"', results_path)

    ## define and set model variables
    h.load_file("env.hoc")
    ## randomstream template
    h.load_file("./templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file("./templates/StimCell.hoc")
    ## helper functions
    h.load_file("lib.hoc")
    if h['H5Connectivity']:
            H5Graph = scatter_graph(MPI._addressof(comm), h['H5ConnectivityPath'], io_size)
    h.load_file("init.hoc")
    

# Run the simulation
def run ():
    h.load_file("run.hoc")
    h.pc.done()
    h.quit()

@cli.command()
@click.argument("--parameters", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("--results-path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("--io-size", type=int, default=64)
@click.option('--verbose', is_flag=True)
def main(parameters, results_path, verbose):
    init(parameters, results_path, io_size=io_size, verbose=verbose)
    run()


