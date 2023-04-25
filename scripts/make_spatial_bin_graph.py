
import os, sys, pickle, base64
import click
import dentate
from dentate import plot, utils, geometry, graph
from dentate.env import Env
from mpi4py import MPI
import h5py

script_name = os.path.basename(__file__)

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook

@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--connectivity-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--output-path", '-o', required=True, type=click.Path(exists=False))
@click.option("--vertex-metrics-namespace", type=str, default='Vertex Metrics')
@click.option("--distances-namespace", '-t', type=str, default='Arc Distances')
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--bin-size", type=float, default=50.)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, config_prefix, connectivity_path, coords_path, output_path, vertex_metrics_namespace, distances_namespace, destination, sources, bin_size, verbose):
        
    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)

    env = Env(config=config, config_prefix=config_prefix)

    layer_extents = env.geometry['Parametric Surface']['Layer Extents']
    (extent_u, extent_v, extent_l) = geometry.get_total_extents(layer_extents)

    extents = geometry.measure_distance_extents(env)
    
    comm=MPI.COMM_WORLD
    
    graph_dict = graph.spatial_bin_graph(connectivity_path, coords_path, distances_namespace,
                                         destination, sources, extents,
                                         bin_size=bin_size, comm=comm)

    if comm.rank == 0:
        graph.save_spatial_bin_graph(output_path, graph_dict)

    comm.barrier()
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(script_name), sys.argv)+1):])
