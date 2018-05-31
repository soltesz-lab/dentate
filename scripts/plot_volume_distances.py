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
@click.option("--resolution", type=(int,int), default=(20,10))
@click.option("--resample", type=int, default=2)
@click.option("--alpha-radius", type=float, default=100.)
@click.option("--verbose", "-v", is_flag=True)
def main(config, resolution, resample, alpha_radius, verbose):
    
    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)

    env = Env(configFile=config)

    layers = env.layers
    rotate = env.geometry['Parametric Surface']['Rotation']
    min_l = float('inf')
    max_l = 0.0
    for layer in layers.keys():
        min_extent = env.geometry['Parametric Surface']['Minimum Extent'][layer]
        max_extent = env.geometry['Parametric Surface']['Maximum Extent'][layer]
        min_l = min(min_extent[2], min_l)
        max_l = max(max_extent[2], max_l)
        
    logger.info('Creating volume: min_l = %f max_l = %f...' % (min_l, max_l))
    ip_volume = make_volume(min_l, max_l, \
                            ures=resolution[0], \
                            vres=resolution[0], \
                            lres=resolution[1], \
                            rotate=rotate)
    logger.info('Computing volume distances...')
    
    vol_dist = get_volume_distances(ip_volume, res=resample, alpha_radius=alpha_radius)
    (dist_u, obs_dist_u, dist_v, obs_dist_v) = vol_dist

    dist_dict = {}
    for i in xrange(0, len(dist_u)):
        dist_dict[i] = { 'U Distance': np.asarray([dist_u[i]], dtype=np.float32), \
                         'V Distance': np.asarray([dist_v[i]], dtype=np.float32) }
    
    plot.plot_positions ("DG Volume", dist_dict, verbose=verbose, saveFig=True)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
