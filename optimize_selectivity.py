"""
Routines for selectivity optimization via Network Clamp.
"""
import os, sys, copy, uuid, pprint, time, gc
from enum import Enum, IntEnum, unique
from collections import defaultdict, namedtuple
from neuroh5.io import read_cell_attribute_info
from mpi4py import MPI
import numpy as np
import click
from dentate import io_utils, spikedata, synapses, stimulus, cell_clamp, optimization
from dentate.cells import h, make_input_cell, register_cell, record_cell, report_topology, is_cell_registered, load_biophys_cell_dicts
from dentate.env import Env
from dentate.neuron_utils import h, configure_hoc_env
from dentate.utils import is_interactive, is_iterable, Context, list_find, list_index, range, str, viewitems, zip_longest, get_module_logger, config_logging, EnumChoice
from dentate.utils import write_to_yaml, read_from_yaml, get_trial_time_indices, get_trial_time_ranges, get_low_pass_filtered_trace, contiguous_ranges, generate_results_file_id
from dentate.network_clamp import init, run_with
from dentate.stimulus import rate_maps_from_features
from dentate.optimization import ProblemRegime, TrialRegime, update_network_params, optimization_params, selectivity_optimization_params, opt_eval_fun
from dmosopt import dmosopt

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

def init_selectivity_objfun(config_file, population, cell_index_set, arena_id, trajectory_id,
                            n_trials, trial_regime, problem_regime,
                            generate_weights, t_max, t_min,
                            template_paths, dataset_prefix, config_prefix, results_path,
                            spike_events_path, spike_events_namespace, spike_events_t,
                            input_features_path, input_features_namespaces,
                            param_type, param_config_name, selectivity_config_name, recording_profile, 
                            state_variable, state_filter, state_baseline,
                            target_features_path, target_features_namespace,
                            target_features_arena, target_features_trajectory,   
                            use_coreneuron, cooperative_init, dt, worker, **kwargs):
    
    params = dict(locals())
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env, bcast_template=True)

    my_cell_index_set = init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
                             spike_events_path, spike_events_namespace=spike_events_namespace, 
                             spike_train_attr_name=spike_events_t,
                             input_features_path=input_features_path,
                             input_features_namespaces=input_features_namespaces,
                             generate_weights_pops=set(generate_weights), 
                             t_min=t_min, t_max=t_max, cooperative_init=cooperative_init,
                             worker=worker)

    time_step = float(env.stimulus_config['Temporal Resolution'])
    equilibration_duration = float(env.stimulus_config.get('Equilibration Duration', 0.))
    
    target_rate_vector_dict = rate_maps_from_features (env, population,
                                                       cell_index_set=my_cell_index_set, 
                                                       input_features_path=target_features_path,
                                                       input_features_namespace=target_features_namespace, 
                                                       time_range=[0., t_max], 
                                                       arena_id=arena_id)


    logger.info(f'target_rate_vector_dict = {target_rate_vector_dict}')
    for gid, target_rate_vector in viewitems(target_rate_vector_dict):
        target_rate_vector[np.isclose(target_rate_vector, 0., atol=1e-3, rtol=1e-3)] = 0.

    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(input_features_path if input_features_path is not None else spike_events_path, 
                                                          target_features_arena, target_features_trajectory)
    time_range = (0., min(np.max(trj_t), t_max))
    time_bins = np.arange(time_range[0], time_range[1]+time_step, time_step)
    state_time_bins = np.arange(time_range[0], time_range[1], time_step)[:-1]

    def range_inds(rs):
        l = list(rs)
        if len(l) > 0:
            a = np.concatenate(l)
        else:
            a = None
        return a

    def time_ranges(rs):
        if len(rs) > 0:
            a = tuple( ( (time_bins[r[0]], time_bins[r[1]-1]) for r in rs ) )
        else:
            a = None
        return a
        
    
    infld_idxs_dict = { gid: np.where(target_rate_vector > 1e-4)[0] 
                        for gid, target_rate_vector in viewitems(target_rate_vector_dict) }
    peak_pctile_dict = { gid: np.percentile(target_rate_vector_dict[gid][infld_idxs], 80)
                         for gid, infld_idxs in viewitems(infld_idxs_dict) }
    trough_pctile_dict = { gid: np.percentile(target_rate_vector_dict[gid][infld_idxs], 20)
                           for gid, infld_idxs in viewitems(infld_idxs_dict) }
    outfld_idxs_dict = { gid: range_inds(contiguous_ranges(target_rate_vector < 1e-4, return_indices=True))
                        for gid, target_rate_vector in viewitems(target_rate_vector_dict) }

    peak_idxs_dict = { gid: range_inds(contiguous_ranges(target_rate_vector >= peak_pctile_dict[gid], return_indices=True)) 
                       for gid, target_rate_vector in viewitems(target_rate_vector_dict) }
    trough_idxs_dict = { gid: range_inds(contiguous_ranges(np.logical_and(target_rate_vector > 0., target_rate_vector <= trough_pctile_dict[gid]), return_indices=True))
                         for gid, target_rate_vector in viewitems(target_rate_vector_dict) }

    outfld_ranges_dict = { gid: time_ranges(contiguous_ranges(target_rate_vector <= 0.) ) 
                           for gid, target_rate_vector in viewitems(target_rate_vector_dict) }
    infld_ranges_dict = { gid: time_ranges(contiguous_ranges(target_rate_vector > 0) ) 
                          for gid, target_rate_vector in viewitems(target_rate_vector_dict) }

    peak_ranges_dict = { gid: time_ranges(contiguous_ranges(target_rate_vector >= peak_pctile_dict[gid]))
                         for gid, target_rate_vector in viewitems(target_rate_vector_dict) }
    trough_ranges_dict = { gid: time_ranges(contiguous_ranges(np.logical_and(target_rate_vector > 0., target_rate_vector <= trough_pctile_dict[gid])))
                         for gid, target_rate_vector in viewitems(target_rate_vector_dict) }

    large_fld_gids = []
    for gid in my_cell_index_set:

        infld_idxs = infld_idxs_dict[gid]

        target_infld_rate_vector = target_rate_vector[infld_idxs]
        target_peak_rate_vector = target_rate_vector[peak_idxs_dict[gid]]
        target_trough_rate_vector = target_rate_vector[trough_idxs_dict[gid]]

        logger.info(f'selectivity objective: target peak/trough rate of gid {gid}: '
                    f'{peak_pctile_dict[gid]:.02f} {trough_pctile_dict[gid]:.02f}')
        logger.info(f'selectivity objective: mean target peak/trough rate of gid {gid}: '
                    f'{np.mean(target_peak_rate_vector):.02f} {np.mean(target_trough_rate_vector):.02f}')
        
    opt_param_config = optimization_params(env.netclamp_config.optimize_parameters, [population], param_config_name, param_type)
    selectivity_opt_param_config = selectivity_optimization_params(env.netclamp_config.optimize_parameters, [population],
                                                                   selectivity_config_name)

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples

    N_objectives = 2
    feature_names = ['mean_peak_rate', 'mean_trough_rate', 
                     'max_infld_rate', 'min_infld_rate', 'mean_infld_rate', 'mean_outfld_rate', 
                     'mean_peak_state', 'mean_trough_state', 'mean_outfld_state']
    feature_dtypes = [(feature_name, np.float32) for feature_name in feature_names]
    feature_dtypes.append(('trial_objs', (np.float32, (N_objectives, n_trials))))
    feature_dtypes.append(('trial_mean_infld_rate', (np.float32, (1, n_trials))))
    feature_dtypes.append(('trial_mean_outfld_rate', (np.float32, (1, n_trials))))

    def from_param_dict(params_dict):
        result = []
        for param_pattern, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_pattern]))

        return result

    def update_run_params(input_param_tuple_vals, update_param_names, update_param_tuples):
        result = []
        updated_set = set([])
        update_param_dict = dict(zip(update_param_names, update_param_tuples))
        for param_pattern, (param_tuple, param_val) in zip(param_names, input_param_tuple_vals):
            if param_pattern in update_param_dict:
                updated_set.add(param_pattern)
                result.append((param_tuple, update_param_dict[param_pattern].param_range))
            else:
                result.append((param_tuple, param_val))
        for update_param_name in update_param_dict:
            if update_param_name not in updated_set:
                result.append((update_param_dict[update_param_name], 
                               update_param_dict[update_param_name].param_range))

        return result
        
    
    def gid_firing_rate_vectors(spkdict, cell_index_set):
        rates_dict = defaultdict(list)
        for i in range(n_trials):
            spkdict1 = {}
            for gid in cell_index_set:
                if gid in spkdict[population]:
                    spkdict1[gid] = spkdict[population][gid][i]
                else:
                    spkdict1[gid] = np.asarray([], dtype=np.float32)
            spike_density_dict = spikedata.spike_density_estimate (population, spkdict1, time_bins)
            for gid in cell_index_set:
                rate_vector = spike_density_dict[gid]['rate']
                rate_vector[np.isclose(rate_vector, 0., atol=1e-3, rtol=1e-3)] = 0.
                rates_dict[gid].append(rate_vector)
                logger.info(f'selectivity objective: trial {i} firing rate min/max of gid {gid}: '
                            f'{np.min(rates_dict[gid]):.02f} / {np.max(rates_dict[gid]):.02f} Hz')

        return rates_dict

    def gid_state_values(spkdict, t_offset, n_trials, t_rec, state_recs_dict):
        t_vec = np.asarray(t_rec.to_python(), dtype=np.float32)
        t_trial_inds = get_trial_time_indices(t_vec, n_trials, t_offset)
        results_dict = {}
        filter_fun = None
        if state_filter == 'lowpass':
            filter_fun = lambda x, t: get_low_pass_filtered_trace(x, t)
        for gid in state_recs_dict:
            state_values = None
            state_recs = state_recs_dict[gid]
            assert(len(state_recs) == 1)
            rec = state_recs[0]
            vec = np.asarray(rec['vec'].to_python(), dtype=np.float32)
            if filter_fun is None:
                data = np.asarray([ vec[t_inds] for t_inds in t_trial_inds ])
            else:
                data = np.asarray([ filter_fun(vec[t_inds], t_vec[t_inds])
                                    for t_inds in t_trial_inds ])

            state_values = []
            max_len = np.max(np.asarray([len(a) for a in data]))
            for state_value_array in data:
                this_len = len(state_value_array)
                if this_len < max_len:
                    a = np.pad(state_value_array, (0, max_len-this_len), 'edge')
                else:
                    a = state_value_array
                state_values.append(a)

            results_dict[gid] = state_values
        return t_vec[t_trial_inds[0]], results_dict


    def trial_snr_residuals(gid, peak_idxs, trough_idxs, infld_idxs, outfld_idxs, 
                            rate_vectors, masked_rate_vectors, target_rate_vector):

        n_trials = len(rate_vectors)
        residual_inflds = []
        trial_inflds = []
        trial_outflds = []

        target_infld = target_rate_vector[infld_idxs]
        target_max_infld = np.max(target_infld)
        target_mean_trough = np.mean(target_rate_vector[trough_idxs])
        logger.info(f'selectivity objective: target max infld/mean trough of gid {gid}: '
                    f'{target_max_infld:.02f} {target_mean_trough:.02f}')
        for trial_i in range(n_trials):

            rate_vector = rate_vectors[trial_i]
            infld_rate_vector = rate_vector[infld_idxs]
            masked_rate_vector = masked_rate_vectors[trial_i]
            if outfld_idxs is None:
                outfld_rate_vector = masked_rate_vectors[trial_i]
            else:
                outfld_rate_vector = rate_vector[outfld_idxs]

            mean_peak = np.mean(rate_vector[peak_idxs])
            mean_trough = np.mean(rate_vector[trough_idxs])
            min_infld = np.min(infld_rate_vector)
            max_infld = np.max(infld_rate_vector)
            mean_infld = np.mean(infld_rate_vector)
            mean_outfld = np.mean(outfld_rate_vector)

            residual_infld = np.abs(np.sum(target_infld - infld_rate_vector))
            logger.info(f'selectivity objective: max infld/mean infld/mean peak/trough/mean outfld/residual_infld of gid {gid} trial {trial_i}: '
                        f'{max_infld:.02f} {mean_infld:.02f} {mean_peak:.02f} {mean_trough:.02f} {mean_outfld:.02f} {residual_infld:.04f}')
            residual_inflds.append(residual_infld)
            trial_inflds.append(mean_infld)
            trial_outflds.append(mean_outfld)

        trial_rate_features = [np.asarray(trial_inflds, dtype=np.float32).reshape((1, n_trials)), 
                               np.asarray(trial_outflds, dtype=np.float32).reshape((1, n_trials))]
        rate_features = [mean_peak, mean_trough, max_infld, min_infld, mean_infld, mean_outfld, ]
        #rate_constr = [ mean_peak if max_infld > 0. else -1. ]
        rate_constr = [ mean_peak - mean_trough if max_infld > 0. else -1. ]
        return (np.asarray(residual_inflds), trial_rate_features, rate_features, rate_constr)

    
    def trial_state_residuals(gid, target_outfld, t_peak_idxs, t_trough_idxs, t_infld_idxs, t_outfld_idxs, state_values, masked_state_values):

        state_value_arrays = np.row_stack(state_values)
        masked_state_value_arrays = None
        if masked_state_values is not None:
            masked_state_value_arrays = np.row_stack(masked_state_values)
        
        residuals_outfld = []
        peak_inflds = []
        trough_inflds = []
        mean_outflds = []
        for i in range(state_value_arrays.shape[0]):
            state_value_array = state_value_arrays[i, :]
            peak_infld = np.mean(state_value_array[t_peak_idxs])
            trough_infld = np.mean(state_value_array[t_trough_idxs])
            mean_infld = np.mean(state_value_array[t_infld_idxs])

            masked_state_value_array = masked_state_value_arrays[i, :]
            mean_masked = np.mean(masked_state_value_array)
            residual_masked = np.mean(masked_state_value_array) - target_outfld

            mean_outfld = mean_masked
            if t_outfld_idxs is not None:
                mean_outfld = np.mean(state_value_array[t_outfld_idxs])
                
            peak_inflds.append(peak_infld)
            trough_inflds.append(trough_infld)
            mean_outflds.append(mean_outfld)
            residuals_outfld.append(residual_masked)
            logger.info(f'selectivity objective: state values of gid {gid}: '
                        f'peak/trough/mean in/mean out/masked: {peak_infld:.02f} / {trough_infld:.02f} / {mean_infld:.02f} / {mean_outfld:.02f} / residual masked: {residual_masked:.04f}')

        state_features = [np.mean(peak_inflds), np.mean(trough_inflds), np.mean(mean_outflds)]
        return (np.asarray(residuals_outfld), state_features)

    
    recording_profile = { 'label': f'optimize_selectivity.{state_variable}',
                          'section quantity': {
                              state_variable: { 'swc types': ['soma'] }
                            }
                        }
    env.recording_profile = recording_profile
    state_recs_dict = {}
    for gid in my_cell_index_set:
        state_recs_dict[gid] = record_cell(env, population, gid, recording_profile=recording_profile)

        
    def eval_problem(cell_param_dict, **kwargs):

        run_params = {population: {gid: from_param_dict(cell_param_dict[gid])
                                   for gid in my_cell_index_set}}
        masked_state_values_dict = {}
        masked_run_params = {population: { gid: update_run_params(run_params[population][gid],
                                                                  selectivity_opt_param_config.mask_param_names,
                                                                  selectivity_opt_param_config.mask_param_tuples)
                                           for gid in my_cell_index_set} }
        spkdict = run_with(env, run_params)
        rates_dict = gid_firing_rate_vectors(spkdict, my_cell_index_set)
        t_s, state_values_dict = gid_state_values(spkdict, equilibration_duration, n_trials, env.t_rec, 
                                                  state_recs_dict)

        masked_spkdict = run_with(env, masked_run_params)
        masked_rates_dict = gid_firing_rate_vectors(masked_spkdict, my_cell_index_set)
        t_s, masked_state_values_dict = gid_state_values(masked_spkdict, equilibration_duration, n_trials, env.t_rec, 
                                                         state_recs_dict)
        
        
        result = {}
        for gid in my_cell_index_set:
            infld_idxs = infld_idxs_dict[gid]
            outfld_idxs = outfld_idxs_dict[gid]
            peak_idxs = peak_idxs_dict[gid]
            trough_idxs = trough_idxs_dict[gid]
            
            target_rate_vector = target_rate_vector_dict[gid]

            peak_ranges = peak_ranges_dict[gid]
            trough_ranges = trough_ranges_dict[gid]
            infld_ranges = infld_ranges_dict[gid]
            outfld_ranges = outfld_ranges_dict[gid]
            
            t_peak_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in peak_ranges ])
            t_trough_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in trough_ranges ])
            t_infld_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in infld_ranges ])
            if outfld_ranges is not None:
                t_outfld_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in outfld_ranges ])
            else:
                t_outfld_idxs = None
            
            masked_state_values = masked_state_values_dict.get(gid, None)
            state_values = state_values_dict[gid]
            rate_vectors = rates_dict[gid]
            masked_rate_vectors = masked_rates_dict[gid]
            
            logger.info(f'selectivity objective: max rates of gid {gid}: '
                        f'{list([np.max(rate_vector) for rate_vector in rate_vectors])}')

            infld_residuals, trial_rate_features, rate_features, rate_constr = \
              trial_snr_residuals(gid, peak_idxs, trough_idxs, infld_idxs, outfld_idxs, 
                                  rate_vectors, masked_rate_vectors, target_rate_vector)
            state_residuals, state_features = trial_state_residuals(gid, state_baseline,
                                                                    t_peak_idxs, t_trough_idxs, t_infld_idxs, t_outfld_idxs,
                                                                    state_values, masked_state_values)
            trial_obj_features = np.row_stack((infld_residuals, state_residuals))
            
            if trial_regime == 'mean':
                mean_infld_residual = np.mean(infld_residuals)
                mean_state_residual = np.mean(state_residuals)
                infld_objective = mean_infld_residual
                state_objective = abs(mean_state_residual)
                logger.info(f'selectivity objective: mean peak/trough/mean infld/mean outfld/mean state residual of gid {gid}: '
                            f'{mean_infld_residual:.04f} {mean_state_residual:.04f}')
            elif trial_regime == 'best':
                min_infld_residual_index = np.argmin(infld_residuals)
                min_infld_residual = infld_residuals[min_infld_index]
                infld_objective = min_infld_residual
                min_state_residual = np.min(np.abs(state_residuals))
                state_objective = min_state_residual
                logger.info(f'selectivity objective: mean peak/trough/max infld/max outfld/min state residual of gid {gid}: '
                            f'{min_infld_residual:.04f} {min_state_residual:.04f}')
            else:
                raise RuntimeError(f'selectivity_rate_objective: unknown trial regime {trial_regime}')

            logger.info(f"rate_features: {rate_features} state_features: {state_features} obj_features: {trial_obj_features}")

            result[gid] = (np.asarray([ infld_objective, state_objective ], 
                                      dtype=np.float32), 
                           np.array([tuple(rate_features+state_features+[trial_obj_features]+trial_rate_features)], 
                                    dtype=np.dtype(feature_dtypes)),
                           np.asarray(rate_constr, dtype=np.float32))
                           
        return result
    
    return opt_eval_fun(problem_regime, my_cell_index_set, eval_problem)


def optimize_run(env, population, param_config_name, selectivity_config_name, init_objfun, problem_regime, nprocs_per_worker=1,
                 n_epochs=10, n_initial=30, initial_maxiter=50, initial_method="slh", optimizer_method="nsga2", surrogate_method='vgp',
                 population_size=200, num_generations=200, resample_fraction=None, mutation_rate=None,
                 param_type='synaptic', init_params={}, results_file=None, cooperative_init=False, 
                 spawn_startup_wait=None, spawn_executable=None, spawn_args=[], verbose=False):

    opt_param_config = optimization_params(env.netclamp_config.optimize_parameters, [population], param_config_name, param_type)

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples
    
    hyperprm_space = { param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
                       for param_pattern, param_tuple in 
                           zip(param_names, param_tuples) }

    if results_file is None:
        if env.results_path is not None:
            file_path = f'{env.results_path}/dmosopt.optimize_selectivity.{env.results_file_id}.h5'
        else:
            file_path = f'dmosopt.optimize_selectivity.{env.results_file_id}.h5'
    else:
        file_path = '%s/%s' % (env.results_path, results_file)
    problem_ids = None
    reduce_fun_name = None
    if ProblemRegime[problem_regime] == ProblemRegime.every:
        reduce_fun_name = "opt_reduce_every"
        problem_ids = init_params.get('cell_index_set', None)
    elif ProblemRegime[problem_regime] == ProblemRegime.mean:
        reduce_fun_name = "opt_reduce_mean"
    elif ProblemRegime[problem_regime] == ProblemRegime.max:
        reduce_fun_name = "opt_reduce_max"
    else:
        raise RuntimeError(f'optimize_run: unknown problem regime {problem_regime}')

    n_trials = init_params.get('n_trials', 1)

    nworkers = env.comm.size-1
    if resample_fraction is None:
        resample_fraction = float(nworkers) / float(population_size)
    if resample_fraction > 1.0:
        resample_fraction = 1.0
    if resample_fraction < 0.1:
        resample_fraction = 0.1

    objective_names = ['residual_infld', 'residual_state']
    feature_names = ['mean_peak_rate', 'mean_trough_rate', 
                     'max_infld_rate', 'min_infld_rate', 'mean_infld_rate', 'mean_outfld_rate', 
                     'mean_peak_state', 'mean_trough_state', 'mean_outfld_state']
    N_objectives = 2
    feature_dtypes = [(feature_name, np.float32) for feature_name in feature_names]
    feature_dtypes.append(('trial_objs', np.float32, (N_objectives, n_trials)))
    feature_dtypes.append(('trial_mean_infld_rate', (np.float32, (1, n_trials))))
    feature_dtypes.append(('trial_mean_outfld_rate', (np.float32, (1, n_trials))))

    constraint_names = ['positive_rate']
    dmosopt_params = {'opt_id': 'dentate.optimize_selectivity',
                      'problem_ids': problem_ids,
                      'obj_fun_init_name': init_objfun, 
                      'obj_fun_init_module': 'dentate.optimize_selectivity',
                      'obj_fun_init_args': init_params,
                      'reduce_fun_name': reduce_fun_name,
                      'reduce_fun_module': 'dentate.optimization',
                      'problem_parameters': {},
                      'space': hyperprm_space,
                      'objective_names': objective_names,
                      'feature_dtypes': feature_dtypes,
                      'constraint_names': constraint_names,
                      'n_initial': n_initial,
                      'n_epochs': n_epochs,
                      'population_size': population_size,
                      'num_generations': num_generations,
                      'resample_fraction': resample_fraction,
                      'mutation_rate': mutation_rate,
                      'initial_maxiter': initial_maxiter,
                      'initial_method': initial_method,
                      'optimizer': optimizer_method,
                      'surrogate_method': surrogate_method,
                      'file_path': file_path,
                      'save': True,
                      'save_eval' : 5,
                      }


    opt_results = dmosopt.run(dmosopt_params, verbose=verbose, collective_mode="sendrecv",
                              spawn_workers=True, nprocs_per_worker=nprocs_per_worker, 
                              spawn_startup_wait=spawn_startup_wait,
                              spawn_executable=spawn_executable, spawn_args=list(spawn_args))
    if opt_results is not None:
        if ProblemRegime[problem_regime] == ProblemRegime.every:
            gid_results_config_dict = {}
            for gid, opt_result in viewitems(opt_results):
                params_dict = dict(opt_result[0])
                result_value = opt_result[1]
                results_config_tuples = []
                for param_pattern, param_tuple in zip(param_names, param_tuples):
                    results_config_tuples.append((param_pattern, params_dict[param_pattern]))
                gid_results_config_dict[int(gid)] = results_config_tuples

            logger.info('Optimized parameters and objective function: '
                        f'{pprint.pformat(gid_results_config_dict)} @'
                        f'{result_value}')
            return gid_results_config_dict
        else:
            params_dict = dict(opt_results[0])
            result_value = opt_results[1]
            results_config_tuples = []
            for param_pattern, param_tuple in zip(param_names, param_tuples):
                results_config_tuples.append((param_pattern, params_dict[param_pattern]))
            logger.info('Optimized parameters and objective function: '
                        f'{pprint.pformat(results_config_tuples)} @'
                        f'{result_value}')
            return results_config_tuples
    else:
        return None

@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--dt",  type=float, help='simulation time step')
@click.option("--gid", '-g', type=int, help='target cell gid')
@click.option("--gid-selection-file", type=click.Path(exists=True, file_okay=True, dir_okay=False), help='file containing target cell gids')
@click.option("--arena-id", '-a', type=str, required=True, help='arena id')
@click.option("--trajectory-id", '-t', type=str, required=True, help='trajectory id')
@click.option("--generate-weights", '-w', required=False, type=str, multiple=True,
              help='generate weights for the given presynaptic population')
@click.option("--t-max", '-t', type=float, default=150.0, help='simulation end time')
@click.option("--t-min", type=float)
@click.option("--nprocs-per-worker", type=int, default=1, help='number of processes per worker')
@click.option("--n-epochs", type=int, default=1)
@click.option("--n-initial", type=int, default=30)
@click.option("--initial-maxiter", type=int, default=50)
@click.option("--initial-method", type=str, default='slh')
@click.option("--optimizer-method", type=str, default='nsga2')
@click.option("--surrogate-method", type=str, default='vgp')
@click.option("--population-size", type=int, default=200)
@click.option("--num-generations", type=int, default=200)
@click.option("--resample-fraction", type=float)
@click.option("--mutation-rate", type=float)
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--param-config-name", type=str, required=True,
              help='parameter configuration name to use for optimization (defined in config file)')
@click.option("--selectivity-config-name", type=str, required=True,
              help='parameter configuration name to use for selectivity-specific config')
@click.option("--param-type", type=str, default='synaptic',
              help='parameter type to use for optimization (synaptic)')
@click.option('--recording-profile', type=str, help='recording profile to use')
@click.option("--results-file", required=False, type=str, help='optimization results file')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--spike-events-path", type=click.Path(), required=False,
              help='path to neuroh5 file containing spike times')
@click.option("--spike-events-namespace", type=str, required=False, default='Spike Events',
              help='namespace containing input spike times')
@click.option("--spike-events-t", required=False, type=str, default='t',
              help='name of variable containing spike times')
@click.option("--input-features-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing input selectivity features')
@click.option("--input-features-namespaces", type=str, multiple=True, required=False, default=['Place Selectivity', 'Grid Selectivity'],
              help='namespace containing input selectivity features')
@click.option("--n-trials", required=False, type=int, default=1,
              help='number of trials for input stimulus')
@click.option("--trial-regime", required=False, type=str, default="mean",
              help='trial aggregation regime (mean or best)')
@click.option("--problem-regime", required=False, type=str, default="every",
              help='problem regime (independently evaluate every problem or mean or max aggregate evaluation)')
@click.option("--target-features-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing target rate maps used for rate optimization')
@click.option("--target-features-namespace", type=str, required=False, default='Input Spikes',
              help='namespace containing target rate maps used for rate optimization')
@click.option("--target-state-variable", type=str, required=False, 
              help='name of state variable used for state optimization')
@click.option("--target-state-filter", type=str, required=False, 
              help='optional filter for state values used for state optimization')
@click.option('--use-coreneuron', is_flag=True, help='enable use of CoreNEURON')
@click.option('--cooperative-init', is_flag=True, help='use a single worker to read model data then send to the remaining workers')
@click.option("--spawn-executable", type=str)
@click.option("--spawn-args", type=str, multiple=True)
@click.option("--spawn-startup-wait", type=int)
def main(config_file, population, dt, gid, gid_selection_file, arena_id, trajectory_id, generate_weights,
         t_max, t_min,  nprocs_per_worker, n_epochs, n_initial, initial_maxiter, initial_method, optimizer_method, surrogate_method,
         population_size, num_generations, resample_fraction, mutation_rate,
         template_paths, dataset_prefix, config_prefix,
         param_config_name, selectivity_config_name, param_type, recording_profile, results_file, results_path, spike_events_path,
         spike_events_namespace, spike_events_t, input_features_path, input_features_namespaces, n_trials,
         trial_regime, problem_regime, target_features_path, target_features_namespace, target_state_variable,
         target_state_filter, use_coreneuron, cooperative_init, spawn_executable, spawn_args, spawn_startup_wait):
    """
    Optimize the input stimulus selectivity of the specified cell in a network clamp configuration.
    """
    init_params = dict(locals())

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    results_file_id = None
    if rank == 0:
        results_file_id = generate_results_file_id(population, gid)
        
    results_file_id = comm.bcast(results_file_id, root=0)
    comm.barrier()
    
    np.seterr(all='raise')
    verbose = True
    cache_queries = True

    config_logging(verbose)

    cell_index_set = set([])
    if gid_selection_file is not None:
        with open(gid_selection_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                gid = int(line)
                cell_index_set.add(gid)
    elif gid is not None:
        cell_index_set.add(gid)
    else:
        comm.barrier()
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
        if rank == 0:
            env = Env(**init_params, comm=comm0)
            attr_info_dict = read_cell_attribute_info(env.data_file_path, populations=[population],
                                                      read_cell_index=True, comm=comm0)
            cell_index = None
            attr_name, attr_cell_index = next(iter(attr_info_dict[population]['Trees']))
            cell_index_set = set(attr_cell_index)
        comm.barrier()
        cell_index_set = comm.bcast(cell_index_set, root=0)
        comm.barrier()
        comm0.Free()
    init_params['cell_index_set'] = cell_index_set
    del(init_params['gid'])

    params = dict(locals())
    env = Env(**params)
    if size == 1:
        configure_hoc_env(env)
        init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
             spike_events_path, spike_events_namespace=spike_events_namespace, 
             spike_train_attr_name=spike_events_t,
             input_features_path=input_features_path,
             input_features_namespaces=input_features_namespaces,
             generate_weights_pops=set(generate_weights), 
             t_min=t_min, t_max=t_max)
        
    if (population in env.netclamp_config.optimize_parameters[param_type]):
        opt_params = env.netclamp_config.optimize_parameters[param_type][population]
    else:
        raise RuntimeError(f'optimize_selectivity: population {population} does not have optimization configuration')

    if target_state_variable is None:
        target_state_variable = 'v'
    
    init_params['target_features_arena'] = arena_id
    init_params['target_features_trajectory'] = trajectory_id
    opt_state_baseline = opt_params['Targets']['state'][target_state_variable]['baseline']
    init_params['state_baseline'] = opt_state_baseline
    init_params['state_variable'] = target_state_variable
    init_params['state_filter'] = target_state_filter
    init_objfun_name = 'init_selectivity_objfun'
        
    best = optimize_run(env, population, param_config_name, selectivity_config_name, init_objfun_name, problem_regime=problem_regime,
                        n_epochs=n_epochs, n_initial=n_initial, initial_maxiter=initial_maxiter, initial_method=initial_method, 
                        optimizer_method=optimizer_method, surrogate_method=surrogate_method, population_size=population_size, 
                        num_generations=num_generations, resample_fraction=resample_fraction, mutation_rate=mutation_rate, 
                        param_type=param_type, init_params=init_params, results_file=results_file, nprocs_per_worker=nprocs_per_worker, 
                        cooperative_init=cooperative_init, spawn_executable=spawn_executable, spawn_args=spawn_args, spawn_startup_wait=spawn_startup_wait, verbose=verbose)
    
    opt_param_config = optimization_params(env.netclamp_config.optimize_parameters, [population], param_config_name, param_type)
    if best is not None:
        if results_path is not None:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            file_path = f'{results_path}/optimize_selectivity.{run_ts}.yaml'
            param_names = opt_param_config.param_names
            param_tuples = opt_param_config.param_tuples

            if ProblemRegime[problem_regime] == ProblemRegime.every:
                results_config_dict = {}
                for gid, prms in viewitems(best):
                    n_res = prms[0][1].shape[0]
                    prms_dict = dict(prms)
                    this_results_config_dict = {}
                    for i in range(n_res):
                        results_param_list = []
                        for param_pattern, param_tuple in zip(param_names, param_tuples):
                            results_param_list.append((param_tuple.population,
                                                       param_tuple.source,
                                                       param_tuple.sec_type,
                                                       param_tuple.syn_name,
                                                       param_tuple.param_path,
                                                       float(prms_dict[param_pattern][i])))
                        this_results_config_dict[i] = results_param_list
                    results_config_dict[gid] = this_results_config_dict
                    
            else:
                prms = best[0]
                n_res = prms[0][1].shape[0]
                prms_dict = dict(prms)
                results_config_dict = {}
                for i in range(n_res):
                    results_param_list = []
                    for param_pattern, param_tuple in zip(param_names, param_tuples):
                        results_param_list.append((param_tuple.population,
                                                   param_tuple.source,
                                                   param_tuple.sec_type,
                                                   param_tuple.syn_name,
                                                   param_tuple.param_path,
                                                   float(prms_dict[param_pattern][i])))
                    results_config_dict[i] = results_param_list

            write_to_yaml(file_path, { population: results_config_dict } )

            
    comm.barrier()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
