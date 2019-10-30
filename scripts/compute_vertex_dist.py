
import os, sys, gc, math
import click
from mpi4py import MPI
import numpy as np
import dentate
from dentate import utils, graph
import h5py

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook

script_name = os.path.basename(__file__)

@click.command()
@click.option("--connectivity-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-t', type=str, default='Arc Distances')
@click.option("--destination", '-d', type=str)
@click.option("--source", '-s', type=str, multiple=True)
@click.option("--bin-size", type=float, default=20.0)
@click.option("--cache-size", type=int, default=100)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(connectivity_path, coords_path, distances_namespace, destination, source, bin_size, cache_size, verbose):

    utils.config_logging(verbose)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    vertex_distribution_dict = graph.vertex_distribution(connectivity_path, coords_path,
                                                         distances_namespace,
                                                         destination, source,
                                                         bin_size, cache_size, comm=comm)

    if rank == 0:
        print(vertex_distribution_dict)
        f = h5py.File(connectivity_path, 'r+')
        
        for dst, src_dict in utils.viewitems(vertex_distribution_dict['Total distance']):
            grp = f.create_group('Vertex Distribution/Total distance/%s' % dst)
            for src, bins in utils.viewitems(src_dict):
                grp[src] = np.asarray(bins, dtype=np.float32)
        for dst, src_dict in utils.viewitems(vertex_distribution_dict['U distance']):
            grp = f.create_group('Vertex Distribution/U distance/%s' % dst)
            for src, bins in utils.viewitems(src_dict):
                grp[src] = np.asarray(bins, dtype=np.float32)
        for dst, src_dict in utils.viewitems(vertex_distribution_dict['V distance']):
            grp = f.create_group('Vertex Distribution/V distance/%s' % dst)
            for src, bins in utils.viewitems(src_dict):
                grp[src] = np.asarray(bins, dtype=np.float32)
        
        f.close()

    comm.Barrier()
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
