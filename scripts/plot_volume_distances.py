import sys, click, logging
from mpi4py import MPI
import numpy as np
import dentate
from dentate import plot, utils
from dentate.geometry import make_volume, get_volume_distances
from dentate.env import Env

script_name = 'plot_volume_distances.py'

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--resolution", type=(int,int,int), default=(33,33,10))
@click.option("--resample", type=int, default=7)
@click.option("--alpha-radius", type=float, default=120.)
@click.option("--graph-type", type=str, default='scatter')
@click.option("--verbose", "-v", is_flag=True)
def main(config, resolution, resample, alpha_radius, graph_type, verbose):
    
    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)

    env = Env(config_file=config)

    layers = env.layers
    rotate = env.geometry['Parametric Surface']['Rotation']
    min_u = float('inf')
    max_u = 0.0
    min_v = float('inf')
    max_v = 0.0
    min_l = float('inf')
    max_l = 0.0
    for layer in list(layers.keys()):
        min_extent = env.geometry['Parametric Surface']['Minimum Extent'][layer]
        max_extent = env.geometry['Parametric Surface']['Maximum Extent'][layer]
        min_u = min(min_extent[0], min_u)
        max_u = max(max_extent[0], max_u)
        min_v = min(min_extent[1], min_v)
        max_v = max(max_extent[1], max_v)
        min_l = min(min_extent[2], min_l)
        max_l = max(max_extent[2], max_l)
        
    logger.info('Creating volume: min_l = %f max_l = %f...' % (min_l, max_l))
    ip_volume = make_volume((min_u, max_u), \
                            (min_v, max_v), \
                            (min_l, max_l), \
                            resolution=resolution, \
                            rotate=rotate)
    logger.info('Computing volume distances...')
    
    vol_dist = get_volume_distances(ip_volume, res=resample, alpha_radius=alpha_radius)
    (obs_uv, dist_u, dist_v) = vol_dist

    dist_dict = {}
    for i in range(0, len(dist_u)):
        dist_dict[i] = { 'U Distance': np.asarray([dist_u[i]], dtype=np.float32), \
                         'V Distance': np.asarray([dist_v[i]], dtype=np.float32) }
    
    plot.plot_positions ("DG Volume", iter(dist_dict.items()), verbose=verbose, saveFig=True, graphType=graph_type)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
