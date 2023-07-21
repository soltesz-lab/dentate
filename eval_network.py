#!/usr/bin/env python
"""
Dentate Gyrus model script for evaluation of synaptic parameters
"""

import os, sys, logging, datetime, gc, pprint
from functools import partial
import click
import numpy as np
from collections import defaultdict, namedtuple
from mpi4py import MPI
import h5py
from neuroh5.io import scatter_read_cell_attribute_selection, read_cell_attribute_info, append_cell_attributes
import dentate
from dentate import network, network_clamp, synapses, spikedata, stimulus, utils, io_utils, optimization
from dentate.env import Env
from dentate.utils import read_from_yaml, write_to_yaml, list_find, viewitems, get_module_logger, config_logging
from dentate.synapses import (SynParam, syn_param_from_dict)
from dentate.optimization import (optimization_params, 
                                  update_network_params, network_features)
from dentate.stimulus import rate_maps_from_features

ParamSpec = namedtuple("ParamSpec", ['param_names',  'param_tuples', ])


logger = get_module_logger(__name__)

def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook

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

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--config-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), \
              help='path to evaluation configuration file')
@click.option("--params-id", required=False, type=int, help='index of parameter set')
@click.option("--n-samples", required=False, type=int, 
              help='number of samples per parameter')
@click.option("--target-features-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing target rate maps used for rate optimization')
@click.option("--target-features-namespace", type=str, required=False, default='Input Spikes',
              help='namespace containing target rate maps used for rate optimization')
@click.option("--output-file-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='results')
@click.option("--output-file-name", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--verbose", '-v', is_flag=True)
def main(config_path, params_id, n_samples, target_features_path, target_features_namespace, output_file_dir, output_file_name, verbose):

    config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    if params_id is None:
        if n_samples is not None:
            logger.info("Generating parameter lattice ...")
            generate_param_lattice(config_path, n_samples, output_file_dir, output_file_name, verbose)
            return

    network_args = click.get_current_context().args
    network_config = {}
    for arg in network_args:
        kv = arg.split("=")
        if len(kv) > 1:
            k,v = kv
            network_config[k.replace('--', '').replace('-', '_')] = v
        else:
            k = kv[0]
            network_config[k.replace('--', '').replace('-', '_')] = True

    run_ts = datetime.datetime.today().strftime('%Y%m%d_%H%M')

    if output_file_name is None:
        output_file_name=f"network_features_{run_ts}.h5"
    output_path = f'{output_file_dir}/{output_file_name}'

    eval_config = read_from_yaml(config_path, include_loader=utils.IncludeLoader)
    eval_config['run_ts'] = run_ts
    if target_features_path is not None:
        eval_config['target_features_path'] = target_features_path
    if target_features_namespace is not None:
        eval_config['target_features_namespace'] = target_features_namespace

    network_param_spec_src = eval_config['param_spec']
    network_param_values = eval_config.get('param_values', {})

    feature_names = eval_config['feature_names']
    target_populations = eval_config['target_populations']

    network_config.update(eval_config.get('kwargs', {}))

    if params_id is None:
        network_config['results_file_id'] = f"DG_eval_network_{eval_config['run_ts']}"
    else:
        network_config['results_file_id'] = f"DG_eval_network_{params_id}_{eval_config['run_ts']}"
        

    env = init_network(comm=MPI.COMM_WORLD, kwargs=network_config)
    gc.collect()

    t_start = 50.
    t_stop = env.tstop
    time_range = (t_start, t_stop)

    target_trj_rate_map_dict = {}
    target_features_arena = env.arena_id
    target_features_trajectory = env.trajectory_id
    for pop_name in target_populations:
        if ('%s snr' % pop_name) not in feature_names:
            continue
        my_cell_index_set = set(env.biophys_cells[pop_name].keys())
        trj_rate_maps = {}
        trj_rate_maps = rate_maps_from_features(env, pop_name,
                                                cell_index_set=list(my_cell_index_set),
                                                input_features_path=target_features_path, 
                                                input_features_namespace=target_features_namespace,
                                                time_range=time_range)
        target_trj_rate_map_dict[pop_name] = trj_rate_maps


    network_param_spec = make_param_spec(target_populations, network_param_spec_src)

    def from_param_list(x):
        result = []
        for pop_param in x:
            this_population, source, sec_type, syn_name, param_path, param_val = pop_param
            param_tuple = SynParam(this_population, source, sec_type, syn_name, param_path, None)
            result.append((param_tuple, param_val))

        return result

    def from_param_dict(x):
        result = []
        for pop_name, param_specs in viewitems(x):
            keyfun = lambda kv: str(kv[0])
            for source, source_dict in sorted(viewitems(param_specs), key=keyfun):
                for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                    for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                        for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                            if isinstance(param_rst, dict):
                                for const_name, const_value in sorted(viewitems(param_rst)):
                                    param_path = (param_fst, const_name)
                                    param_tuple = SynParam(pop_name, source, sec_type, syn_name, param_path, const_value)
                                    result.append(param_tuple, const_value)
                            else:
                                param_name = param_fst
                                param_value = param_rst
                                param_tuple = SynParam(pop_name, source, sec_type, syn_name, param_name, param_value)
                                result.append(param_tuple, param_value)
        return result

    eval_network(env, network_config,
                 from_param_list, from_param_dict,
                 network_param_spec, network_param_values, params_id,
                 target_trj_rate_map_dict, t_start, t_stop, 
                 target_populations, output_path)



def generate_param_lattice(config_path, n_samples, output_file_dir, output_file_name, maxiter=5, verbose=False):
    from dmosopt import sampling

    logger = utils.get_script_logger(os.path.basename(__file__))

    output_path = None
    if output_file_name is not None:
        output_path = f'{output_file_dir}/{output_file_name}'
    eval_config = read_from_yaml(config_path, include_loader=utils.IncludeLoader)
    network_param_spec_src = eval_config['param_spec']

    target_populations = eval_config['target_populations']
    network_param_spec = make_param_spec(target_populations, network_param_spec_src)
    param_tuples = network_param_spec.param_tuples
    param_names = network_param_spec.param_names
    n_params = len(param_tuples)

    n_init = n_params * n_samples
    Xinit = sampling.glp(n_init, n_params, maxiter=maxiter)

    ub = []
    lb = []
    for param_name, param_tuple in zip(param_names, param_tuples):
        param_range = param_tuple.param_range
        ub.append(param_range[1])
        lb.append(param_range[0])

    ub = np.asarray(ub)
    lb = np.asarray(lb)

    for i in range(n_init):
        Xinit[i,:] = Xinit[i,:] * (ub - lb) + lb

    output_dict = {}
    for i in range(Xinit.shape[0]):
        output_dict[i] = list([float(x) for x in Xinit[i, :]])

    if output_path is not None:    
        write_to_yaml(output_path, output_dict)
    else:
        pprint.pprint(output_dict)



def make_param_spec(pop_names, param_dict):

    """Constructs a flat list representation of synaptic parameters."""
    
    param_names = []
    param_initial_dict = {}
    param_tuples = []

    for pop_name in pop_names:
        param_specs = param_dict[pop_name]
        keyfun = lambda kv: str(kv[0])
        for source, source_dict in sorted(viewitems(param_specs), key=keyfun):
            for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                    for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                        if isinstance(param_rst, dict):
                            for const_name, const_range in sorted(viewitems(param_rst)):
                                param_path = (param_fst, const_name)
                                param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_path, const_range))
                                param_key = '%s.%s.%s.%s.%s.%s' % (pop_name, str(source), sec_type, syn_name, param_fst, const_name)
                                param_names.append(param_key)
                        else:
                            param_name = param_fst
                            param_range = param_rst
                            param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_name, param_range))
                            param_key = '%s.%s.%s.%s.%s' % (pop_name, source, sec_type, syn_name, param_name)
                            param_names.append(param_key)
                                
    return ParamSpec(param_names=param_names, param_tuples=param_tuples)

    
def init_network(comm, kwargs):
    np.seterr(all='raise')
    env = Env(comm=comm, **kwargs)
    network.init(env)
    env.comm.barrier()
    return env


def eval_network(env, network_config, from_param_list, from_param_dict, network_params, network_param_values, params_id, target_trj_rate_map_dict, t_start, t_stop, target_populations, output_path):

    param_tuple_values = None
    if params_id is not None:
        x = network_param_values[params_id]
        if isinstance(x, list):
            param_tuple_values = from_param_list(x)
        elif isinstance(x, dict):
            param_tuple_values = from_param_dict(x)
        else:
            raise RuntimeError(f"eval_network: invalid input parameters argument {x}")
    
        if env.comm.rank == 0:
            logger.info("*** Updating network parameters ...")
            logger.info(pprint.pformat(param_tuple_values))
        update_network_params(env, param_tuple_values)

    env.checkpoint_clear_data = False
    env.checkpoint_interval = None
    env.tstop = t_stop
    network.run(env, output=network_config.get('output_results', False), shutdown=False)

    for pop_name in target_trj_rate_map_dict:
        append_cell_attributes(env.results_file_path, pop_name, target_trj_rate_map_dict[pop_name], 
                               namespace='Target Trajectory Rate Map', comm=env.comm, io_size=env.io_size)

    local_features = network_features(env, target_trj_rate_map_dict, t_start, t_stop, target_populations)
    return collect_network_features(env, local_features, target_populations, output_path, params_id, param_tuple_values)



def collect_network_features(env, local_features, target_populations, output_path, params_id, param_tuple_values):

    all_features = env.comm.gather(local_features, root=0)
    
    collected_features = {}
    if env.comm.rank == 0:
        for pop_name in target_populations:

            pop_features_dicts = [ features_dict[pop_name] for features_dict in all_features ]

            sum_mean_rate = 0.
            sum_snr = 0.
            n_total = 0
            n_active = 0
            n_target_rate_map = 0
            for pop_feature_dict in pop_features_dicts:

                n_active_local = pop_feature_dict['n_active']
                n_total_local = pop_feature_dict['n_total']
                n_target_rate_map_local = pop_feature_dict['n_target_rate_map']
                sum_mean_rate_local = pop_feature_dict['sum_mean_rate']
                sum_snr_local = pop_feature_dict['sum_snr']

                n_total += n_total_local
                n_active += n_active_local
                n_target_rate_map += n_target_rate_map_local
                sum_mean_rate += sum_mean_rate_local

                if sum_snr_local is not None:
                    sum_snr += sum_snr_local

            if n_active > 0:
                mean_rate = sum_mean_rate / n_active
            else:
                mean_rate = 0.

            if n_total > 0:
                fraction_active = n_active / n_total
            else:
                fraction_active = 0.

            mean_snr = None
            if n_target_rate_map > 0:
                mean_snr = sum_snr / n_target_rate_map

            logger.info(f'population {pop_name}: n_active = {n_active} n_total = {n_total} mean rate = {mean_rate}')
            logger.info(f'population {pop_name}: n_target_rate_map = {n_target_rate_map} snr: sum = {sum_snr} mean = {mean_snr}')

            collected_features['%s fraction active' % pop_name] = fraction_active
            collected_features['%s firing rate' % pop_name] = mean_rate
            if mean_snr is not None:
                collected_features['%s snr' % pop_name] = mean_snr

        output_file = h5py.File(output_path, "a")
        network_grp = h5_get_group(output_file, 'DG_eval_network')
        param_grp = h5_get_group(network_grp, f'{params_id}')
        if param_tuple_values is not None:
            dset = h5_get_dataset(param_grp, 'param_values', maxshape=(len(param_tuple_values),), dtype=np.float32)
            param_vals = np.asarray([v[1] for v in param_tuple_values], dtype=np.float32)
            dset.resize((len(param_vals),))
            dset[:] = param_vals
        feature_grp = h5_get_group(param_grp, 'feature_values')
        for feature_name in sorted(collected_features):
            feature_val = collected_features[feature_name]
            dset = h5_get_dataset(feature_grp, feature_name, maxshape=(1,), dtype=np.float32)
            dset.resize((1,))
            dset[0] = feature_val
        output_file.close()

    env.comm.barrier()

    return collected_features


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
