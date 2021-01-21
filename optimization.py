#!/usr/bin/env python

"""
Routines and objective functions related to optimization of synaptic parameters.
"""

__author__ = 'See AUTHORS.md'

import os, sys, logging
import click
import numpy as np
from mpi4py import MPI
from collections import defaultdict, namedtuple
import dentate
from dentate import synapses, spikedata, stimulus, utils
from dentate.env import Env
from dentate.utils import viewitems
from neuroh5.io import scatter_read_cell_attribute_selection, read_cell_attribute_info


SynParam = namedtuple('SynParam',
                      ['population',
                       'source',
                       'sec_type',
                       'syn_name',
                       'param_path',
                       'param_range'])


def syn_param_from_dict(d):
    return SynParam(*[d[key] for key in SynParam._fields])


OptConfig = namedtuple("OptConfig",
                       ['param_bounds', 
                        'param_names', 
                        'param_initial_dict', 
                        'param_tuples', 
                        'opt_targets'])


def optimization_params(optimization_config, pop_names, param_config_name, param_type='synaptic'):

    """Constructs a flat list representation of synaptic optimization parameters based on network clamp optimization configuration."""
    
    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_tuples = []
    opt_targets = {}

    for pop_name in pop_names:
        if param_type == 'synaptic':
            if pop_name in optimization_config['synaptic']:
                opt_params = optimization_config['synaptic'][pop_name]
                param_ranges = opt_params['Parameter ranges'][param_config_name]
            else:
                raise RuntimeError(
                    "optimization_params: population %s does not have optimization configuration" % pop_name)
            for target_name, target_val in viewitems(opt_params['Targets']):
                opt_targets['%s %s' % (pop_name, target_name)] = target_val
            keyfun = lambda kv: str(kv[0])
            for source, source_dict in sorted(viewitems(param_ranges), key=keyfun):
                for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                    for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                        for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                            if isinstance(param_rst, dict):
                                for const_name, const_range in sorted(viewitems(param_rst)):
                                    param_path = (param_fst, const_name)
                                    param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_path, const_range))
                                    param_key = '%s.%s.%s.%s.%s.%s' % (pop_name, str(source), sec_type, syn_name, param_fst, const_name)
                                    param_initial_value = (const_range[1] - const_range[0]) / 2.0
                                    param_initial_dict[param_key] = param_initial_value
                                    param_bounds[param_key] = const_range
                                    param_names.append(param_key)
                            else:
                                param_name = param_fst
                                param_range = param_rst
                                param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_name, param_range))
                                param_key = '%s.%s.%s.%s.%s' % (pop_name, source, sec_type, syn_name, param_name)
                                param_initial_value = (param_range[1] - param_range[0]) / 2.0
                                param_initial_dict[param_key] = param_initial_value
                                param_bounds[param_key] = param_range
                                param_names.append(param_key)
                                
        else:
            raise RuntimeError("optimization_params: unknown parameter type %s" % param_type)

    return OptConfig(param_bounds=param_bounds, 
                     param_names=param_names, 
                     param_initial_dict=param_initial_dict, 
                     param_tuples=param_tuples, 
                     opt_targets=opt_targets)


def update_network_params(env, param_tuple_values):
    
    for param_tuple, param_value in param_tuple_values:

        pop_name = param_tuple.population
        biophys_cell_dict = env.biophys_cells[pop_name]
        synapse_config = env.celltypes[pop_name]['synapses']
        weights_dict = synapse_config.get('weights', {})

        for gid in biophys_cell_dict:

            biophys_cell = biophys_cell_dict[gid]
            is_reduced = False
            if hasattr(biophys_cell, 'is_reduced'):
                is_reduced = biophys_cell.is_reduced

            source = param_tuple.source
            sec_type = param_tuple.sec_type
            syn_name = param_tuple.syn_name
            param_path = param_tuple.param_path
            
            if isinstance(param_path, list) or isinstance(param_path, tuple):
                p, s = param_path
            else:
                p, s = param_path, None

            sources = None
            if isinstance(source, list) or isinstance(source, tuple):
                sources = source
            else:
                if source is not None:
                    sources = [source]

            if isinstance(sec_type, list) or isinstance(sec_type, tuple):
                sec_types = sec_type
            else:
                sec_types = [sec_type]
            for this_sec_type in sec_types:
                synapses.modify_syn_param(biophys_cell, env, this_sec_type, syn_name,
                                              param_name=p, 
                                              value={s: param_value} if (s is not None) else param_value,
                                              filters={'sources': sources} if sources is not None else None,
                                              origin=None if is_reduced else 'soma', 
                                              update_targets=True)


def rate_maps_from_features (env, pop_name, input_features_path, input_features_namespace, cell_index_set,
                             arena_id=None, trajectory_id=None, time_range=None, n_trials=1):
    
    """Initializes presynaptic spike sources from a file with input selectivity features represented as firing rates."""
        
    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    if arena_id is None:
        arena_id = env.arena_id
    if trajectory_id is None:
        trajectory_id = env.trajectory_id

    spatial_resolution = float(env.stimulus_config['Spatial Resolution'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    
    this_input_features_namespace = '%s %s' % (input_features_namespace, arena_id)
    
    input_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                 'Module ID', 'Grid Spacing', 'Grid Orientation',
                                 'Field Width Concentration Factor', 
                                 'X Offset', 'Y Offset']
    
    selectivity_type_names = { i: n for n, i in viewitems(env.selectivity_types) }

    arena = env.stimulus_config['Arena'][arena_id]
    arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=spatial_resolution)
    
    trajectory = arena.trajectories[trajectory_id]
    equilibration_duration = float(env.stimulus_config.get('Equilibration Duration', 0.))

    t, x, y, d = stimulus.generate_linear_trajectory(trajectory, temporal_resolution=temporal_resolution,
                                                     equilibration_duration=equilibration_duration)
    if time_range is not None:
        t_range_inds = np.where((t < time_range[1]) & (t >= time_range[0]))[0] 
        t = t[t_range_inds]
        x = x[t_range_inds]
        y = y[t_range_inds]
        d = d[t_range_inds]

    input_rate_map_dict = {}
    pop_index = int(env.Populations[pop_name])

    input_features_iter = scatter_read_cell_attribute_selection(input_features_path, pop_name,
                                                                selection=cell_index_set,
                                                                namespace=this_input_features_namespace,
                                                                mask=set(input_features_attr_names), 
                                                                comm=env.comm, io_size=env.io_size)
    for gid, selectivity_attr_dict in input_features_iter:

        this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
        input_cell_config = stimulus.get_input_cell_config(selectivity_type=this_selectivity_type,
                                                           selectivity_type_names=selectivity_type_names,
                                                           selectivity_attr_dict=selectivity_attr_dict)
        if input_cell_config.num_fields > 0:
            rate_map = input_cell_config.get_rate_map(x=x, y=y)
            rate_map[np.isclose(rate_map, 0., atol=1e-3, rtol=1e-3)] = 0.
            input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict



def network_features(env, target_trj_rate_map_dict, t_start, t_stop, target_populations):

    features_dict = dict()
 
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    time_bins  = np.arange(t_start, t_stop, temporal_resolution)

    pop_spike_dict = spikedata.get_env_spike_dict(env, include_artificial=False)
    

    for pop_name in target_populations:

        has_target_trj_rate_map = pop_name in target_trj_rate_map_dict

        n_active = 0
        sum_mean_rate = 0.
        spike_density_dict = spikedata.spike_density_estimate (pop_name, pop_spike_dict[pop_name], time_bins)
        for gid, dens_dict in utils.viewitems(spike_density_dict):
            mean_rate = np.mean(dens_dict['rate'])
            sum_mean_rate += mean_rate
            if mean_rate > 0.:
                n_active += 1
        
        n_total = len(env.cells[pop_name]) - len(env.artificial_cells[pop_name])

        n_target_trj_rate_map = 0
        sum_target_rate_dist_residual = None
        if has_target_trj_rate_map:
            pop_target_trj_rate_map_dict = target_trj_rate_map_dict[pop_name]
            n_target_rate_map = len(pop_target_trj_rate_map_dict)
            target_rate_dist_residuals = []
            for gid in pop_target_trj_rate_map_dict:
                target_trj_rate_map = pop_target_trj_rate_map_dict[gid]
                rate_map_len = len(target_trj_rate_map)
                if gid in spike_density_dict:
                    residual = np.mean(target_trj_rate_map - spike_density_dict[gid]['rate'][:rate_map_len])
                else:
                    residual = np.mean(target_trj_rate_map)
                target_rate_dist_residuals.append(residual)
            sum_target_rate_dist_residual = np.sum(target_rate_dist_residuals)
    
        pop_features_dict = {}
        pop_features_dict['n_total'] = n_total
        pop_features_dict['n_active'] = n_active
        pop_features_dict['n_target_rate_map'] = n_target_rate_map
        pop_features_dict['sum_mean_rate'] = sum_mean_rate
        pop_features_dict['sum_target_rate_dist_residual'] = sum_target_rate_dist_residual

        features_dict[pop_name] = pop_features_dict

    return features_dict
