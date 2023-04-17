"""
Routines for selectivity optimization via Network Clamp.
"""
import os, sys, copy, uuid, pprint, time, gc
os.environ["DISTWQ_CONTROLLER_RANK"] = "-1"

from enum import Enum, IntEnum, unique
from collections import defaultdict, namedtuple
from neuron import h
from mpi4py import MPI
import numpy as np
import click
from dentate import io_utils, spikedata, synapses, stimulus, cell_clamp, optimization
from dentate.cells import (
    make_input_cell,
    register_cell,
    record_cell,
    report_topology,
    is_cell_registered,
    load_biophys_cell_dicts,
)
from dentate.env import Env
from dentate.neuron_utils import configure_hoc_env
from dentate.utils import (
    is_interactive,
    is_iterable,
    Context,
    list_find,
    list_index,
    range,
    str,
    viewitems,
    zip_longest,
    get_module_logger,
    config_logging,
    EnumChoice,
)
from dentate.utils import (
    write_to_yaml,
    read_from_yaml,
    get_trial_time_indices,
    get_trial_time_ranges,
    get_low_pass_filtered_trace,
    contiguous_ranges,
    generate_results_file_id,
)
from dentate.network_clamp import init, run_with
from dentate.stimulus import rate_maps_from_features
from dentate.optimization import (
    OptResult,
    ProblemRegime,
    TrialRegime,
    update_network_params,
    optimization_params,
    selectivity_optimization_params,
    opt_eval_fun,
)
from neuroh5.io import read_cell_attribute_info
import distgfs

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

def init_controller(subworld_size, use_coreneuron):
    h.nrnmpi_init()
    h('objref pc, cvode')
    h.cvode = h.CVode()
    h.pc = h.ParallelContext()
    h.pc.subworlds(subworld_size)
    if use_coreneuron:
        from neuron import coreneuron
        coreneuron.enable = True
        coreneuron.verbose = 0
        h.cvode.cache_efficient(1)
        h.finitialize(-65)
        h.pc.set_maxstep(10)
        h.pc.psolve(0.1)

def init_selectivity_objfun(
    config,
    population,
    cell_index_set,
    arena_id,
    trajectory_id,
    n_trials,
    trial_regime,
    problem_regime,
    generate_weights,
    t_max,
    t_min,
    template_paths,
    dataset_prefix,
    config_prefix,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    input_features_path,
    input_features_namespaces,
    param_type,
    param_config_name,
    selectivity_config_name,
    target_features_path,
    target_features_namespace,
    target_features_arena,
    target_features_trajectory,
    infld_threshold,
    use_coreneuron,
    cooperative_init,
    nprocs_per_worker,
    dt,
    worker,
    **kwargs,
):

    params = dict(locals())
    params["comm"] = MPI.COMM_WORLD
    if worker is not None:
        params["comm"] = worker.merged_comm

    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env, subworld_size=nprocs_per_worker, bcast_template=True)
    if use_coreneuron:
        h.cvode.cache_efficient(1)
        h.finitialize(-65)
        h.pc.set_maxstep(10)
        h.pc.psolve(0.1)

    my_cell_index_set = init(
        env,
        population,
        cell_index_set,
        arena_id,
        trajectory_id,
        n_trials,
        spike_events_path,
        spike_events_namespace=spike_events_namespace,
        spike_train_attr_name=spike_events_t,
        input_features_path=input_features_path,
        input_features_namespaces=input_features_namespaces,
        generate_weights_pops=set(generate_weights),
        t_min=t_min,
        t_max=t_max,
        cooperative_init=cooperative_init,
        worker=worker,
    )

    time_step = float(env.stimulus_config["Temporal Resolution"])
    equilibration_duration = float(
        env.stimulus_config.get("Equilibration Duration", 0.0)
    )


    target_rate_vector_dict = rate_maps_from_features(
        env,
        population,
        cell_index_set=my_cell_index_set,
        input_features_path=target_features_path,
        input_features_namespace=target_features_namespace,
        time_range=[0.0, t_max],
        arena_id=arena_id,
    )

    logger.info(f"cell_index_set = {cell_index_set}")
    logger.info(f"arena_id = {arena_id}")
    logger.info(f"target_features_path = {target_features_path}")
    logger.info(f"target_features_namespace = {target_features_namespace}")
    logger.info(f"target_rate_vector_dict = {target_rate_vector_dict}")
    
    for gid, target_rate_vector in viewitems(target_rate_vector_dict):
        target_rate_vector[
            np.isclose(target_rate_vector, 0.0, atol=1e-3, rtol=1e-3)
        ] = 0.0

    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(
        input_features_path if input_features_path is not None else spike_events_path,
        target_features_arena,
        target_features_trajectory,
    )
    time_range = (0.0, min(np.max(trj_t), t_max))
    time_bins = np.arange(time_range[0], time_range[1] + time_step, time_step)

    def range_inds(rs):
        l = list(rs)
        if len(l) > 0:
            a = np.concatenate(l)
        else:
            a = None
        return a

    def time_ranges(rs):
        if len(rs) > 0:
            a = tuple(((time_bins[r[0]], time_bins[r[1] - 1]) for r in rs))
        else:
            a = None
        return a

    infld_idxs_dict = {
        gid: np.where(target_rate_vector >= infld_threshold)[0]
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }
    peak_pctile_dict = {
        gid: np.percentile(target_rate_vector_dict[gid][infld_idxs], 80)
        for gid, infld_idxs in viewitems(infld_idxs_dict)
    }
    trough_pctile_dict = {
        gid: np.percentile(target_rate_vector_dict[gid][infld_idxs], 20)
        for gid, infld_idxs in viewitems(infld_idxs_dict)
    }
    outfld_idxs_dict = {
        gid: range_inds(
            contiguous_ranges(target_rate_vector < infld_threshold, return_indices=True)
        )
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }

    peak_idxs_dict = {
        gid: range_inds(
            contiguous_ranges(
                target_rate_vector >= peak_pctile_dict[gid], return_indices=True
            )
        )
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }
    trough_idxs_dict = {
        gid: range_inds(
            contiguous_ranges(
                np.logical_and(
                    target_rate_vector > 0.0,
                    target_rate_vector <= trough_pctile_dict[gid],
                ),
                return_indices=True,
            )
        )
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }

    outfld_ranges_dict = {
        gid: time_ranges(contiguous_ranges(target_rate_vector <= 0.0))
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }
    infld_ranges_dict = {
        gid: time_ranges(contiguous_ranges(target_rate_vector > 0))
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }

    peak_ranges_dict = {
        gid: time_ranges(contiguous_ranges(target_rate_vector >= peak_pctile_dict[gid]))
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }
    trough_ranges_dict = {
        gid: time_ranges(
            contiguous_ranges(
                np.logical_and(
                    target_rate_vector > 0.0,
                    target_rate_vector <= trough_pctile_dict[gid],
                )
            )
        )
        for gid, target_rate_vector in viewitems(target_rate_vector_dict)
    }

    large_fld_gids = []
    for gid in my_cell_index_set:

        infld_idxs = infld_idxs_dict[gid]

        target_infld_rate_vector = target_rate_vector[infld_idxs]
        target_peak_rate_vector = target_rate_vector[peak_idxs_dict[gid]]
        target_trough_rate_vector = target_rate_vector[trough_idxs_dict[gid]]

        logger.info(
            f"selectivity objective: target peak/trough rate of gid {gid}: "
            f"{peak_pctile_dict[gid]:.02f} {trough_pctile_dict[gid]:.02f}"
        )
        logger.info(
            f"selectivity objective: mean target peak/trough rate of gid {gid}: "
            f"{np.mean(target_peak_rate_vector):.02f} {np.mean(target_trough_rate_vector):.02f}"
        )

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        [population],
        param_config_name,
        param_type,
    )
    selectivity_opt_param_config = selectivity_optimization_params(
        env.netclamp_config.optimize_parameters, [population], selectivity_config_name
    )

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples

    feature_names = [
        "mean_peak_rate",
        "mean_trough_rate",
        "max_infld_rate",
        "min_infld_rate",
        "mean_infld_rate",
        "mean_outfld_rate",
    ]

    feature_dtypes = [(feature_name, (np.float32, (1, 1))) for feature_name in feature_names]
    feature_dtypes.append(("trial_mean_infld_rate", (np.float32, (1, n_trials))))
    feature_dtypes.append(("trial_mean_outfld_rate", (np.float32, (1, n_trials))))

    def from_param_dict(params_dict):
        result = []
        for param_pattern, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_pattern]))

        return result

    def update_run_params(
        input_param_tuple_vals, update_param_names, update_param_tuples
    ):
        result = []
        updated_set = set([])
        update_param_dict = dict(zip(update_param_names, update_param_tuples))
        for param_pattern, (param_tuple, param_val) in zip(
            param_names, input_param_tuple_vals
        ):
            if param_pattern in update_param_dict:
                updated_set.add(param_pattern)
                result.append(
                    (param_tuple, update_param_dict[param_pattern].param_range)
                )
            else:
                result.append((param_tuple, param_val))
        for update_param_name in update_param_dict:
            if update_param_name not in updated_set:
                result.append(
                    (
                        update_param_dict[update_param_name],
                        update_param_dict[update_param_name].param_range,
                    )
                )

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
            spike_density_dict = spikedata.spike_density_estimate(
                population, spkdict1, time_bins
            )
            for gid in cell_index_set:
                rate_vector = spike_density_dict[gid]["rate"]
                rate_vector[np.isclose(rate_vector, 0.0, atol=1e-3, rtol=1e-3)] = 0.0
                rates_dict[gid].append(rate_vector)
                logger.info(
                    f"selectivity objective: trial {i} firing rate min/max of gid {gid}: "
                    f"{np.min(rates_dict[gid]):.02f} / {np.max(rates_dict[gid]):.02f} Hz"
                )

        return rates_dict

    def trial_rate_snrs(
        gid,
        peak_idxs,
        trough_idxs,
        infld_idxs,
        outfld_idxs,
        rate_vectors,
        target_rate_vector,
    ):

        n_trials = len(rate_vectors)
        snrs = []
        trial_inflds = []
        trial_outflds = []

        target_var = np.var(target_rate_vector)
        target_infld = target_rate_vector[infld_idxs]
        target_max_infld = np.max(target_infld)
        target_mean_peak = np.mean(target_rate_vector[peak_idxs])
        target_mean_trough = np.mean(target_rate_vector[trough_idxs])
        logger.info(
            f"selectivity objective: gid {gid}: target var/max infld/mean peak/mean trough: "
            f"{target_var:.04f} {target_max_infld:.02f} {target_mean_peak:.02f} {target_mean_trough:.02f}"
        )
        for trial_i in range(n_trials):

            rate_vector = rate_vectors[trial_i]
            infld_rate_vector = rate_vector[infld_idxs]
            outfld_rate_vector = None
            if outfld_idxs is not None:
                outfld_rate_vector = rate_vector[outfld_idxs]
            n = min(len(rate_vector), len(target_rate_vector))
            
            var_delta = np.var(rate_vector[:n] - target_rate_vector[:n])
            mean_peak = np.mean(rate_vector[peak_idxs])
            mean_trough = np.mean(rate_vector[trough_idxs])
            min_infld = np.min(infld_rate_vector)
            max_infld = np.max(infld_rate_vector)
            mean_infld = np.mean(infld_rate_vector)
            mean_outfld = np.nan
            if outfld_rate_vector is not None:
                mean_outfld = np.mean(outfld_rate_vector)

            snr = target_var / var_delta

            logger.info(
                f"selectivity objective: gid {gid} trial {trial_i}: max infld/mean infld/mean peak/trough/mean outfld/snr: "
                f"{max_infld:.02f} {mean_infld:.02f} {mean_peak:.02f} {mean_trough:.02f} {mean_outfld:.02f} {snr:.04f}"
            )

            snrs.append(snr)
            trial_inflds.append(mean_infld)
            trial_outflds.append(mean_outfld)

        trial_rate_features = [
            np.asarray(trial_inflds, dtype=np.float32).reshape((1, n_trials)),
            np.asarray(trial_outflds, dtype=np.float32).reshape((1, n_trials)),
        ]
        rate_features = [
            [mean_peak],
            [mean_trough],
            [max_infld],
            [min_infld],
            [mean_infld],
            [mean_outfld],
        ]
        # rate_constr = [ mean_peak if max_infld > 0. else -1. ]
        # rate_constr = [ mean_peak - mean_trough if max_infld > 0. else -1. ]
        return (np.asarray(snrs), trial_rate_features, rate_features)

    env.recording_profile = None

    def eval_problem(cell_param_dict, **kwargs):

        run_params = {
            population: {
                gid: from_param_dict(cell_param_dict[gid]) for gid in my_cell_index_set
            }
        }
        spkdict = run_with(env, run_params)
        rates_dict = gid_firing_rate_vectors(spkdict, my_cell_index_set)
        t_vec = np.asarray(env.t_rec.to_python(), dtype=np.float32)
        t_trial_inds = get_trial_time_indices(t_vec, n_trials, equilibration_duration)
        t_s = t_vec[t_trial_inds[0]]

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

            t_peak_idxs = np.concatenate(
                [
                    np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0]
                    for r in peak_ranges
                ]
            )
            t_trough_idxs = np.concatenate(
                [
                    np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0]
                    for r in trough_ranges
                ]
            )
            t_infld_idxs = np.concatenate(
                [
                    np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0]
                    for r in infld_ranges
                ]
            )
            if outfld_ranges is not None:
                t_outfld_idxs = np.concatenate(
                    [
                        np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0]
                        for r in outfld_ranges
                    ]
                )
            else:
                t_outfld_idxs = None

            rate_vectors = rates_dict[gid]

            logger.info(
                f"selectivity objective: max rates of gid {gid}: "
                f"{list([np.max(rate_vector) for rate_vector in rate_vectors])}"
            )

            snrs, trial_rate_features, rate_features = trial_rate_snrs(
                gid,
                peak_idxs,
                trough_idxs,
                infld_idxs,
                outfld_idxs,
                rate_vectors,
                target_rate_vector,
            )

            if trial_regime == "mean":
                snr_objective = np.mean(snrs)
            elif trial_regime == "best":
                snr_objective = np.max(snrs)
            else:
                raise RuntimeError(
                    f"selectivity_rate_objective: unknown trial regime {trial_regime}"
                )

            result[gid] = (
                snr_objective,
                np.array(
                    [tuple(rate_features + trial_rate_features)],
                    dtype=np.dtype(feature_dtypes),
                ),
            )

        return result

    return opt_eval_fun(problem_regime, my_cell_index_set, eval_problem, feature_dtypes)


def optimize_run(
    env,
    population,
    param_config_name,
    selectivity_config_name,
    init_objfun,
    problem_regime,
    nprocs_per_worker=1,
    use_coreneuron=False,
    n_iter=10,
    param_type="synaptic",
    init_params={},
    results_file=None,
    n_max_tasks=-1,
    cooperative_init=False,
    spawn_workers=False,
    spawn_startup_wait=None,
    spawn_executable=None,
    spawn_args=[],
    verbose=False,
    return_distgfs_params=False,
):

    objective_names = ["snr"]
    feature_names = [
        "mean_peak_rate",
        "mean_trough_rate",
        "max_infld_rate",
        "min_infld_rate",
        "mean_infld_rate",
        "mean_outfld_rate",
    ]

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        [population],
        param_config_name,
        param_type,
    )

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples

    hyperprm_space = {
        param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
        for param_pattern, param_tuple in zip(param_names, param_tuples)
    }

    if results_file is None:
        if env.results_path is not None:
            file_path = os.path.join(env.results_path, f"distgfs.optimize_selectivity.{env.results_file_id}.h5")
        else:
            file_path = f"distgfs.optimize_selectivity.{env.results_file_id}.h5"
    else:
        file_path = os.path.join(env.results_path, results_file)
    problem_ids = None
    cell_index_set = init_params.get("cell_index_set", None)
    n_trials = init_params.get("n_trials", 1)

    n_problems = 1
    if problem_regime == ProblemRegime.every:
        n_problems = 1
    else:
        n_problems = len(cell_index_set)
    feature_dtypes = [(feature_name, (np.float32, (n_problems, 1))) for feature_name in feature_names]
    feature_dtypes.append(("trial_mean_infld_rate", (np.float32, (n_problems, n_trials))))
    feature_dtypes.append(("trial_mean_outfld_rate", (np.float32, (n_problems, n_trials))))

    
    reduce_fun_name = None
    reduce_fun_args = {}
    if ProblemRegime[problem_regime] == ProblemRegime.every:
        reduce_fun_name = "opt_reduce_every_features"
        problem_ids = cell_index_set
    elif ProblemRegime[problem_regime] == ProblemRegime.mean:
        reduce_fun_name = "opt_reduce_mean_features"
        assert(cell_index_set is not None)
        reduce_fun_args = { "index": cell_index_set,
                            "feature_dtypes": feature_dtypes, }
    elif ProblemRegime[problem_regime] == ProblemRegime.max:
        reduce_fun_name = "opt_reduce_max_features"
        assert(cell_index_set is not None)
        reduce_fun_args = { "index": cell_index_set,
                            "feature_dtypes": feature_dtypes, }
    else:
        raise RuntimeError(f"optimize_run: unknown problem regime {problem_regime}")

    nworkers = env.comm.size - 1
    if n_max_tasks <= 0:
        n_max_tasks = nworkers

    distgfs_params = {
        "opt_id": "dentate.optimize_selectivity",
        "problem_ids": problem_ids,
        "obj_fun_init_name": init_objfun,
        "obj_fun_init_module": "dentate.optimize_selectivity_snr",
        "obj_fun_init_args": init_params,
        "controller_init_fun_module": "dentate.optimize_selectivity_snr",
        "controller_init_fun_name": "init_controller",
        "controller_init_fun_args": {"subworld_size": nprocs_per_worker,
                                     "use_coreneuron": use_coreneuron},
        "reduce_fun_name": reduce_fun_name,
        "reduce_fun_module": "dentate.optimization",
        "reduce_fun_args": reduce_fun_args,
        "problem_parameters": {},
        "space": hyperprm_space,
        "objective_names": objective_names,
        "feature_dtypes": feature_dtypes,
        "n_iter": n_iter,
        "n_max_tasks": n_max_tasks,
        "file_path": file_path,
        "save": True,
        "save_eval": 5,
    }

    opt_results = distgfs.run(
        distgfs_params,
        verbose=verbose,
        collective_mode="sendrecv",
        nprocs_per_worker=nprocs_per_worker,
        spawn_workers=spawn_workers,
        spawn_startup_wait=spawn_startup_wait,
        spawn_executable=spawn_executable,
        spawn_args=list(spawn_args),
    )
    if opt_results is not None:
        if ProblemRegime[problem_regime] == ProblemRegime.every:
            gid_result_dict = {}
            for gid, opt_result in viewitems(opt_results):
                params_dict = dict(opt_result[0])
                result_value = opt_result[1]
                result_param_tuples = []
                for param_pattern, param_tuple in zip(param_names, param_tuples):
                    result_param_tuples.append(
                        (param_pattern, params_dict[param_pattern])
                    )
                gid_result_dict[int(gid)] = OptResult(result_param_tuples,
                                                      {'objective': result_value},
                                                      None)

            logger.info(
                "Optimized parameters and objective function: "
                f"{pprint.pformat(gid_result_dict)}"
            )
            results = gid_result_dict
        else:
            params_dict = dict(opt_results[0])
            result_value = opt_results[1]
            result_param_tuples = []
            for param_pattern, param_tuple in zip(param_names, param_tuples):
                result_param_tuples.append(
                    (param_pattern, params_dict[param_pattern])
                )
            logger.info(
                "Optimized parameters and objective function: "
                f"{pprint.pformat(result_param_tuples)} @"
                f"{result_value}"
            )
            results = {pop_name: OptResult(result_param_tuples,
                                           {'objective': result_value},
                                           None) }
    else:
        results = None

    if return_distgfs_params:
        return results, distgfs_params
    else:
        return results

def main(
    config,
    population,
    dt,
    gid,
    arena_id,
    trajectory_id,
    generate_weights,
    t_max,
    t_min,
    nprocs_per_worker,
    template_paths,
    dataset_prefix,
    config_prefix,
    param_config_name,
    selectivity_config_name,
    param_type,
    param_results_file,
    results_file,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    input_features_path,
    input_features_namespaces,
    n_iter,
    n_trials,
    n_max_tasks,
    trial_regime,
    problem_regime,
    target_features_path,
    target_features_namespace,
    infld_threshold,
    use_coreneuron,
    cooperative_init,
    spawn_workers,
    spawn_executable,
    spawn_args,
    spawn_startup_wait,
    verbose,
):
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

    np.seterr(all="raise")
    cache_queries = True

    config_logging(verbose or (rank == size-1))
    

    cell_index_set = set([])
    if gid is not None:
        cell_index_set.add(gid)
    else:
        comm.barrier()
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
        if rank == 0:
            env = Env(**init_params, comm=comm0)
            attr_info_dict = read_cell_attribute_info(
                env.data_file_path,
                populations=[population],
                read_cell_index=True,
                comm=comm0,
            )
            cell_index = None
            attr_name, attr_cell_index = next(iter(attr_info_dict[population]["Trees"]))
            cell_index_set = set(attr_cell_index)
        comm.barrier()
        cell_index_set = comm.bcast(cell_index_set, root=0)
        comm.barrier()
        comm0.Free()
    init_params["cell_index_set"] = cell_index_set
    del init_params["gid"]

    params = dict(locals())
    env = Env(**params)
    if size == 1:
        configure_hoc_env(env)
        init(
            env,
            population,
            cell_index_set,
            arena_id,
            trajectory_id,
            n_trials,
            spike_events_path,
            spike_events_namespace=spike_events_namespace,
            spike_train_attr_name=spike_events_t,
            input_features_path=input_features_path,
            input_features_namespaces=input_features_namespaces,
            generate_weights_pops=set(generate_weights),
            t_min=t_min,
            t_max=t_max,
        )

    if population in env.netclamp_config.optimize_parameters[param_type]:
        opt_params = env.netclamp_config.optimize_parameters[param_type][population]
    else:
        raise RuntimeError(
            f"optimize_selectivity: population {population} does not have optimization configuration"
        )

    init_params["target_features_arena"] = arena_id
    init_params["target_features_trajectory"] = trajectory_id
    init_objfun_name = "init_selectivity_objfun"

    results_dict, distgfs_params = optimize_run(
        env,
        population,
        param_config_name,
        selectivity_config_name,
        init_objfun_name,
        problem_regime=problem_regime,
        n_iter=n_iter,
        param_type=param_type,
        init_params=init_params,
        results_file=results_file,
        nprocs_per_worker=nprocs_per_worker,
        use_coreneuron=use_coreneuron,
        n_max_tasks=n_max_tasks,
        cooperative_init=cooperative_init,
        spawn_workers=spawn_workers,
        spawn_executable=spawn_executable,
        spawn_args=spawn_args,
        spawn_startup_wait=spawn_startup_wait,
        verbose=verbose or (rank == size-1),
        return_distgfs_params=True,
    )

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        [population],
        param_config_name,
        param_type,
    )
    if results_dict is not None:
        if results_path is not None:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            if param_results_file is None:
                param_results_file = f"optimize_selectivity.{run_ts}.yaml"
            file_path = os.path.join(results_path, param_results_file)
            param_names = opt_param_config.param_names
            param_tuples = opt_param_config.param_tuples
            output_dict = {}
            if ProblemRegime[problem_regime] == ProblemRegime.every:
                for gid, opt_res in viewitems(results_dict):
                    prms_dict = dict(opt_res.parameters)
                    this_results_config_dict = {}
                    results_param_list = []
                    for param_pattern, param_tuple in zip(param_names, param_tuples):
                        results_param_list.append(
                            (
                                param_tuple.population,
                                param_tuple.source,
                                param_tuple.sec_type,
                                param_tuple.syn_name,
                                param_tuple.param_path,
                                float(prms_dict[param_pattern]),
                            )
                        )
                    output_dict[gid] = {0: results_param_list}

            else:
                prms_dict = dict(results_dict.parameters)
                results_param_list = []
                for param_pattern, param_tuple in zip(param_names, param_tuples):
                    results_param_list.append(
                        (
                            param_tuple.population,
                            param_tuple.source,
                            param_tuple.sec_type,
                            param_tuple.syn_name,
                            param_tuple.param_path,
                            float(prms_dict[param_pattern]),
                        )
                    )
                output_dict[0] = results_param_list

            write_to_yaml(file_path, {population: output_dict})

    comm.barrier()
    if results_dict is not None:
        return {population: results_dict}, distgfs_params
    else:
        return None, None

    
@click.command()
@click.option(
    "--config", "-c", required=True, type=str, help="model configuration file name"
)
@click.option(
    "--population",
    "-p",
    required=True,
    type=str,
    default="GC",
    help="target population",
)
@click.option("--dt", type=float, help="simulation time step")
@click.option("--gid", "-g", type=int, help="target cell gid")
@click.option("--arena-id", "-a", type=str, required=True, help="arena id")
@click.option("--trajectory-id", "-t", type=str, required=True, help="trajectory id")
@click.option(
    "--generate-weights",
    "-w",
    required=False,
    type=str,
    multiple=True,
    help="generate weights for the given presynaptic population",
)
@click.option("--t-max", "-t", type=float, default=150.0, help="simulation end time")
@click.option("--t-min", type=float)
@click.option(
    "--nprocs-per-worker", type=int, default=1, help="number of processes per worker"
)
@click.option(
    "--template-paths",
    type=str,
    required=True,
    help="colon-separated list of paths to directories containing hoc cell templates",
)
@click.option(
    "--dataset-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing required neuroh5 data files",
)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
    help="path to directory containing network and cell mechanism config files",
)
@click.option(
    "--param-config-name",
    type=str,
    required=True,
    help="parameter configuration name to use for optimization (defined in config file)",
)
@click.option(
    "--selectivity-config-name",
    type=str,
    required=True,
    help="parameter configuration name to use for selectivity-specific config",
)
@click.option(
    "--param-type",
    type=str,
    default="synaptic",
    help="parameter type to use for optimization (synaptic)",
)
@click.option(
    "--param-results-file", required=False, type=str, help="optimization parameter results yaml file"
)
@click.option(
    "--results-file", required=False, type=str, help="optimization results file"
)
@click.option(
    "--results-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory where output files will be written",
)
@click.option(
    "--spike-events-path",
    type=click.Path(),
    required=False,
    help="path to neuroh5 file containing spike times",
)
@click.option(
    "--spike-events-namespace",
    type=str,
    required=False,
    default="Spike Events",
    help="namespace containing input spike times",
)
@click.option(
    "--spike-events-t",
    required=False,
    type=str,
    default="t",
    help="name of variable containing spike times",
)
@click.option(
    "--input-features-path",
    required=False,
    type=click.Path(),
    help="path to neuroh5 file containing input selectivity features",
)
@click.option(
    "--input-features-namespaces",
    type=str,
    multiple=True,
    required=False,
    default=["Place Selectivity", "Grid Selectivity"],
    help="namespace containing input selectivity features",
)
@click.option(
    "--n-iter",
    required=False,
    type=int,
    default=10,
    help="number of optimization iterations",
)
@click.option(
    "--n-trials",
    required=False,
    type=int,
    default=1,
    help="number of trials for input stimulus",
)
@click.option(
    "--n-max-tasks",
    required=False,
    type=int,
    default=-1,
    help="number of maximum tasks",
)
@click.option(
    "--trial-regime",
    required=False,
    type=str,
    default="mean",
    help="trial aggregation regime (mean or best)",
)
@click.option(
    "--problem-regime",
    required=False,
    type=str,
    default="every",
    help="problem regime (independently evaluate every problem or mean or max aggregate evaluation)",
)
@click.option(
    "--target-features-path",
    required=False,
    type=click.Path(),
    help="path to neuroh5 file containing target rate maps used for rate optimization",
)
@click.option(
    "--target-features-namespace",
    type=str,
    required=False,
    default="Input Spikes",
    help="namespace containing target rate maps used for rate optimization",
)
@click.option(
    "--infld-threshold",
    type=float,
    required=False,
    default=1e-3,
    help="minimum firing rate threshold for in-field calculation",
)
@click.option("--use-coreneuron", is_flag=True, help="enable use of CoreNEURON")
@click.option(
    "--cooperative-init",
    is_flag=True,
    help="use a single worker to read model data then send to the remaining workers",
)
@click.option("--spawn-workers", is_flag=True)
@click.option("--spawn-executable", type=str)
@click.option("--spawn-args", type=str, multiple=True)
@click.option("--spawn-startup-wait", type=int)
@click.option("--verbose", is_flag=True)
def main_cmd(
    config,
    population,
    dt,
    gid,
    arena_id,
    trajectory_id,
    generate_weights,
    t_max,
    t_min,
    nprocs_per_worker,
    template_paths,
    dataset_prefix,
    config_prefix,
    param_config_name,
    selectivity_config_name,
    param_type,
    param_results_file,
    results_file,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    input_features_path,
    input_features_namespaces,
    n_iter,
    n_trials,
    n_max_tasks,
    trial_regime,
    problem_regime,
    target_features_path,
    target_features_namespace,
    infld_threshold,
    use_coreneuron,
    cooperative_init,
    spawn_workers,
    spawn_executable,
    spawn_args,
    spawn_startup_wait,
    verbose,
):
    return main(config,
                population,
                dt,
                gid,
                arena_id,
                trajectory_id,
                generate_weights,
                t_max,
                t_min,
                nprocs_per_worker,
                template_paths,
                dataset_prefix,
                config_prefix,
                param_config_name,
                selectivity_config_name,
                param_type,
                param_results_file,
                results_file,
                results_path,
                spike_events_path,
                spike_events_namespace,
                spike_events_t,
                input_features_path,
                input_features_namespaces,
                n_iter,
                n_trials,
                n_max_tasks,
                trial_regime,
                problem_regime,
                target_features_path,
                target_features_namespace,
                infld_threshold,
                use_coreneuron,
                cooperative_init,
                spawn_workers,
                spawn_executable,
                spawn_args,
                spawn_startup_wait,
                verbose,
                )

if __name__ == "__main__":
    main_cmd(
        args=sys.argv[
            (
                list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ]
    )
