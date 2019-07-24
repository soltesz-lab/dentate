
import os, sys, pickle, base64
import click
import dentate
from dentate import plot
from dentate import utils
from mpi4py import MPI
import h5py

script_name = os.path.basename(__file__)

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

    env = Env(config_file=config, config_prefix=config_prefix)

    (u_bin_graph, v_bin_graph) = spatial_bin_graph(connectivity_path, coords_path, distances_namespace,
                                                   destination, sources, extents, bin_size=bin_size, comm=MPI.COMM_WORLD)

    u_bin_graph_pkl = pickle.dumps(u_bin_graph)
    u_bin_graph_pkl_str = base64.b64encode(u_bin_graph_pkl) 
    v_bin_graph_pkl = pickle.dumps(v_bin_graph)
    v_bin_graph_pkl_str = base64.b64encode(v_bin_graph_pkl) 

    f = h5py.File(output_path)
    grp = f.create_group(dataset_path)
    grp['U graph.pkl'] = u_bin_graph_pkl_str
    grp['V graph.pkl'] = v_bin_graph_pkl_str
    f.close()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(script_name), sys.argv)+1):])
