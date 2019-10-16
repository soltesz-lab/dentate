
import os, sys, gc, math
import click
import dentate
from dentate import utils, graph
from mpi4py import MPI
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

    vertex_distribution_dict = graph.vertex_distribution(connectivity_path, coords_path,
                                                         distances_namespace,
                                                         destination, source,
                                                         bin_size, cache_size, comm=comm)

    if rank == 0:
        f = h5py.File(connectivity_path, 'a')

        for dst, src_dict in utils.viewitems(vertex_distribution_dict['Total distance']):
            for src, bins in utils.viewitems(src_dict):
                f['Vertex Distribution']['Total distance'][dst][src] = np.asarray(bins, dtype=np.uint32)
        for dst, src_dict in utils.viewitems(vertex_distribution_dict['U distance']):
            for src, bins in utils.viewitems(src_dict):
                f['Vertex Distribution']['U distance'][dst][src] = np.asarray(bins, dtype=np.uint32)
        for dst, src_dict in utils.viewitems(vertex_distribution_dict['V distance']):
            for src, bins in utils.viewitems(src_dict):
                f['Vertex Distribution']['V distance'][dst][src] = np.asarray(bins, dtype=np.uint32)
        
        f.close()

    comm.Barrier()
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
