from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
import sys, os, gc, itertools, math, click, logging
from collections import defaultdict
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, read_projection_names, bcast_cell_attributes, NeuroH5ProjectionGen
import numpy as np
from dentate import utils
from dentate.utils import *


sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


def update_bins(bins, binsize, x):
    i = math.floor(old_div(x, binsize))
    if i in bins:
        bins[i] += 1
    else:
        bins[i] = 1
        
def finalize_bins(bins, binsize):
    imin = int(min(bins.keys()))
    imax = int(max(bins.keys()))
    a = [0] * (imax - imin + 1)
    b = [binsize * k for k in range(imin, imax + 1)]
    for i in range(imin, imax + 1):
        if i in bins:
            a[i - imin] = bins[i]
    return np.asarray(a), np.asarray(b)

def merge_bins(bins1, bins2, datatype):
    for i, count in viewitems(bins2):
        bins1[i] += count
    return bins1

def add_bins(bins1, bins2, datatype):
    for item in bins2:
        if item in bins1:
            bins1[item] += bins2[item]
        else:
            bins1[item] = bins2[item]
    return bins1

@click.command()
@click.option("--connectivity-path", '-p', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", '-o', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--coords-path", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--distances-namespace", type=str, default='Arc Distances')
@click.option("--destination", '-d', required=True, type=str)
@click.option("--bin-size", type=int, default=20)
@click.option("--cache-size", type=int, default=50)
@click.option("--verbose", "-v", is_flag=True)
def main(connectivity_path, output_path, coords_path, distances_namespace, destination, bin_size, cache_size, verbose):
    """
    Measures vertex distribution with respect to septo-temporal distance

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination: 
    :param source: 

    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
        
    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    if rank == 0:
        logger.info('reading %s distances...' % destination)
    destination_soma_distances = bcast_cell_attributes(coords_path, destination, namespace=distances_namespace, comm=comm, root=0)
    

    destination_soma_distance_U = {}
    destination_soma_distance_V = {}
    for k,v in destination_soma_distances:
        destination_soma_distance_U[k] = v['U Distance'][0]
        destination_soma_distance_V[k] = v['V Distance'][0]

    del(destination_soma_distances)

    sources = []
    for (src, dst) in read_projection_names(connectivity_path):
        if dst == destination:
            sources.append(src)

    source_soma_distances = {}
    for s in sources:
        if rank == 0:
            logger.info('reading %s distances...' % s)
        source_soma_distances[s] = bcast_cell_attributes(coords_path, s, namespace=distances_namespace, comm=comm, root=0)

    
    source_soma_distance_U = {}
    source_soma_distance_V = {}
    for s in sources:
        this_source_soma_distance_U = {}
        this_source_soma_distance_V = {}
        for k,v in source_soma_distances[s]:
            this_source_soma_distance_U[k] = v['U Distance'][0]
            this_source_soma_distance_V[k] = v['V Distance'][0]
        source_soma_distance_U[s] = this_source_soma_distance_U
        source_soma_distance_V[s] = this_source_soma_distance_V
    del(source_soma_distances)

    logger.info('reading connections %s -> %s...' % (str(sources), destination))
    gg = [ NeuroH5ProjectionGen (connectivity_path, source, destination, cache_size=cache_size, comm=comm) for source in sources ]

    dist_bins = defaultdict(dict)
    dist_u_bins = defaultdict(dict)
    dist_v_bins = defaultdict(dict)
    
    for prj_gen_tuple in utils.zip_longest(*gg):
        destination_gid = prj_gen_tuple[0][0]
        if not all([prj_gen_elt[0] == destination_gid for prj_gen_elt in prj_gen_tuple]):
            raise Exception('destination %s: destination_gid %i not matched across multiple projection generators: %s' %
                            (destination, destination_gid, [prj_gen_elt[0] for prj_gen_elt in prj_gen_tuple]))

        if destination_gid is not None:
            logger.info('reading connections of gid %i' % destination_gid)
            for (source, (this_destination_gid,rest)) in zip(sources, prj_gen_tuple):
                this_source_soma_distance_U = source_soma_distance_U[source]
                this_source_soma_distance_V = source_soma_distance_V[source]
                this_dist_bins = dist_bins[source]
                this_dist_u_bins = dist_u_bins[source]
                this_dist_v_bins = dist_v_bins[source]
                (source_indexes, attr_dict) = rest
                dst_U = destination_soma_distance_U[destination_gid]
                dst_V = destination_soma_distance_V[destination_gid]
                for source_gid in source_indexes:
                    dist_u = dst_U - this_source_soma_distance_U[source_gid]
                    dist_v = dst_V - this_source_soma_distance_V[source_gid]
                    dist = abs(dist_u) + abs(dist_v)
                
                    update_bins(this_dist_bins, bin_size, dist)
                    update_bins(this_dist_u_bins, bin_size, dist_u)
                    update_bins(this_dist_v_bins, bin_size, dist_v)
    comm.barrier()

    logger.info('merging distance dictionaries...')
    add_bins_op = MPI.Op.Create(add_bins, commute=True)
    for source in sources:
        dist_bins[source] = comm.reduce(dist_bins[source], op=add_bins_op, root=0)
        dist_u_bins[source] = comm.reduce(dist_u_bins[source], op=add_bins_op, root=0)
        dist_v_bins[source] = comm.reduce(dist_v_bins[source], op=add_bins_op, root=0)
                
    comm.barrier()
    
    if rank == 0:
        color = 1
    else:
        color = 0

    ## comm0 includes only rank 0
    comm0 = comm.Split(color, 0)

    if rank == 0:
        if output_path is None:
            output_path = connectivity_path
        logger.info('writing output to %s...' % output_path)

        #f = h5py.File(output_path, 'a', driver='mpio', comm=comm0)
        #if 'Nodes' in f:
        #    nodes_grp = f['Nodes']
        #else:
        #    nodes_grp = f.create_group('Nodes')
        #grp = nodes_grp.create_group('Connectivity Distance Histogram')
        #dst_grp = grp.create_group(destination)
        for source in sources:
            dist_histoCount, dist_bin_edges = finalize_bins(dist_bins[source], bin_size)
            dist_u_histoCount, dist_u_bin_edges = finalize_bins(dist_u_bins[source], bin_size)
            dist_v_histoCount, dist_v_bin_edges = finalize_bins(dist_v_bins[source], bin_size)
            np.savetxt('%s Distance U Bin Count.dat' % source, dist_u_histoCount)
            np.savetxt('%s Distance U Bin Edges.dat' % source, dist_u_bin_edges)
            np.savetxt('%s Distance V Bin Count.dat' % source, dist_v_histoCount)
            np.savetxt('%s Distance V Bin Edges.dat' % source, dist_v_bin_edges)
            np.savetxt('%s Distance Bin Count.dat' % source, dist_histoCount)
            np.savetxt('%s Distance Bin Edges.dat' % source, dist_bin_edges)
        #f.close()
    comm.barrier()
            

            
if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
