import sys, itertools, click
import h5py
import numpy as np
from dentate import utils, env
from env import Env

script_name="make_h5types.py"

grp_h5types      = 'H5Types'
grp_projections  = 'Projections'
grp_populations  = 'Populations'

path_population_labels = '/%s/Population labels' % grp_h5types
path_population_range = '/%s/Population range' % grp_h5types

grp_population_projections = 'Population projections'
grp_valid_population_projections = 'Valid population projections'
path_population_projections = '/%s/Population projections' % grp_h5types


def h5_get_group (h, groupname):
    if groupname in h.keys():
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g

def h5_get_dataset (g, dsetname, **kwargs):
    if dsetname in g.keys():
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset

def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize+len(data),)
    dset.resize(newshape)
    dset[dsize:] = data
    return dset


@click.command()
@click.option("--config", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", default='dentate_h5types.h5', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--gap-junctions', is_flag=True)
def main(config, output_path, gap_junctions):

    env = Env(configFile=config)

    populations = []
    for pop_name, pop_idx in env.pop_dict.items():
        layer_counts = env.geometry['Cell Layer Counts'][pop_name]
        pop_count = 0
        for layer_name, layer_count in layer_counts.items():
            pop_count += layer_count
        populations.append((pop_name, pop_idx, pop_count))
    populations.sort(key=lambda x: x[1])

    projections = []
    if gap_junctions:
        for post, connection_dict in env.gapjunctions.items():
            for pre, _ in connection_dict.items():
                projections.append((env.pop_dict[pre], env.pop_dict[post]))
    else:
        for post, connection_dict in env.connection_config.items():
            for pre, _ in connection_dict.items():
                projections.append((env.pop_dict[pre], env.pop_dict[post]))
    
    # create an HDF5 enumerated type for the population label
    mapping = { name: idx for name, idx in env.pop_dict.items() }
    dt_population_labels = h5py.special_dtype(enum=(np.uint16, mapping))

    with h5py.File(output_path, "a", libver="latest") as h5:


        h5[path_population_labels] = dt_population_labels

        dt_populations = np.dtype([("Start", np.uint64), ("Count", np.uint32),
                                   ("Population", h5[path_population_labels].dtype)])
        h5[path_population_range]  = dt_populations
        
        # create an HDF5 compound type for population ranges
        dt = h5[path_population_range].dtype

        g = h5_get_group (h5, grp_h5types)

        dset = h5_get_dataset(g, grp_populations, maxshape=(len(populations),), dtype=dt)
        dset.resize((len(populations),))
        a = np.zeros(len(populations), dtype=dt)

        start = 0
        for name, idx, count in populations:
            a[idx]["Start"] = start
            a[idx]["Count"] = count
            a[idx]["Population"] = idx
            start += count

        dset[:] = a

        dt_projections = np.dtype([("Source", h5[path_population_labels].dtype),
                                    ("Destination", h5[path_population_labels].dtype)])

        h5[path_population_projections] = dt_projections

        dt = h5[path_population_projections]
        dset = h5_get_dataset(g, grp_valid_population_projections,
                              maxshape=(len(projections),), dtype=dt)
        dset.resize((len(projections),))
        a = np.zeros(len(projections), dtype=dt)
        idx = 0
        for i, prj in enumerate(projections):
            src, dst = prj
            a[i]["Source"] = int(src)
            a[i]["Destination"] = int(dst)

        dset[:] = a




if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
