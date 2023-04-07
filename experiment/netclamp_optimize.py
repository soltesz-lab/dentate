import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Tuple, Optional, Union, List, Dict, Set
from dentate import network_clamp, utils, task
from mpi4py import MPI
from simple_slurm import Slurm
from commandlib import Command
import click

def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


@dataclass
class NetClampOptimize:
    run_id: Optional[Any] = None
    comm: Optional[MPI.Intracomm] = None
    gids: Tuple[int] = ()
    population: str = "GC"
    dt: Optional[float] = None
    arena_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    generate_weights: Optional[Tuple[str]] = None
    t_max: Optional[float] = 150.0
    t_min: Optional[float] = None
    nprocs_per_worker: int = 1
    opt_epsilon: float = 1e-2
    opt_seed: Optional[int] = None
    opt_iter: int = 10
    templates: str = "templates"
    config: Optional[str] = None
    param_config_name: Optional[str] = None
    param_type: str = "synaptic"
    spike_events_namespace: str = "Spike Events"
    spike_events_t: str = "t"
    recording_profile: Optional[str] = None
    distances_namespace: str = "Arc Distances"
    phase_mod: bool = False
    input_features_namespaces: Tuple[str] =  ("Place Selectivity", "Grid Selectivity")
    n_trials: int = 1
    trial_regime: str = "mean"
    problem_regime: str = "every"
    target_features_namespace: str = "Input Spikes"
    target_state_variable: Optional[str] = None
    target_state_filter: Optional[str] = None
    use_coreneuron: bool = False
    cooperative_init: bool = False
    target: str = "rate"
    
    def execute(self, inputs, outputs):

        comm = self.comm
        model_configuration = None
        if (comm is None) or ((comm is not None) and (comm.rank == 0)):
            config_path = os.path.join(inputs.get("config_prefix", ""), self.config)
            model_configuration = utils.read_from_yaml(config_path,
                                                       include_loader=utils.IncludeLoader)
        if comm is not None:
            model_configuration = comm.bcast(model_configuration, root=0)

        results_dict, distgfs_params = network_clamp.optimize(
            config=model_configuration,
            config_prefix=None,
            dt=self.dt,
            population=self.population,
            gids=self.gids,
            arena_id=self.arena_id,
            trajectory_id=self.trajectory_id,
            generate_weights=self.generate_weights,
            t_max=self.t_max,
            t_min=self.t_min,
            nprocs_per_worker=self.nprocs_per_worker,
            opt_epsilon=self.opt_epsilon,
            opt_seed=self.opt_seed,
            opt_iter=self.opt_iter,
            template_paths=self.templates,
            dataset_prefix=inputs.get("dataset_prefix", None),
            param_config_name=self.param_config_name,
            param_type=self.param_type,
            recording_profile=self.recording_profile,
            results_file=None,
            results_path=outputs["job_output"],
            spike_events_path=inputs.get("spike_events", None),
            spike_events_namespace=self.spike_events_namespace,
            spike_events_t=self.spike_events_t,
            coords_path=inputs.get("cell_coordinates", None),
            distances_namespace=self.distances_namespace,
            phase_mod=self.phase_mod,
            input_features_path=inputs.get("input_features", None),
            input_features_namespaces=self.input_features_namespaces,
            n_trials=self.n_trials,
            trial_regime=self.trial_regime,
            problem_regime=self.problem_regime,
            target_features_path=inputs.get("target_features", None),
            target_features_namespace=self.target_features_namespace,
            target_state_variable=self.target_state_variable,
            target_state_filter=self.target_state_filter,
            use_coreneuron=self.use_coreneuron,
            target=self.target,
            cooperative_init=self.cooperative_init,
        )

        for population in results_dict:
            mlflow.set_experiment(f"Network Clamp Optimize {population}")
            mlflow.set_experiment_tag("population", population)
            if self.problem_regime == 'every':
                with mlflow.start_run(run_name=self.run_id):
                    for gid in results_dict[population].parameters:
                        with mlflow.start_run(run_name=gid, nested=True):
                            mlflow.log_params(results_dict[population][gid].parameters)
                            mlflow.log_metrics(results_dict[population][gid].objectives)
            else:
                with mlflow.start_run(run_name=self.run_id):
                    mlflow.log_params(results_dict[population].parameters)
                    mlflow.log_metrics(results_dict[population].objectives)

def mpi_get_config(config):
    comm = MPI.COMM_WORLD
    if isinstance(config, str):
        if comm.rank == 0:
            config_dict = utils.read_from_yaml(config, include_loader=utils.IncludeLoader)
        config_dict = comm.bcast(config_dict, root=0)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise RuntimeError(f"mpi_get_config: unable to process configuration object {config}")
    config_dict = utils.yaml_envsubst(config_dict)
    return config_dict

def get_config(config):
    if isinstance(config, str):
        config_dict = utils.read_from_yaml(config, include_loader=utils.IncludeLoader)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise RuntimeError(f"get_config: unable to process configuration object {config}")
    config_dict = utils.yaml_envsubst(config)
    return config_dict
    
@click.group()
def cli():
    pass

@click.command()
@click.option(
    "--operational-config", "-c", required=True, type=str, help="operational configuration file name"
)
@click.option(
    "--inputs", "-i", required=True, type=str, help="input configuration file name"
)
@click.option(
    "--outputs", "-o", required=True, type=str, help="output configuration file name"
)
def run(operational_config, inputs, outputs):

    netclamp_config = mpi_get_config(operational_config)
    inputs = mpi_get_config(inputs)
    outputs= mpi_get_config(outputs)

    if 'run_id' not in netclamp_config:
        run_id = uuid.uuid4()
        netclamp_config['run_id'] = run_id
    
    opt = NetClampOptimize(**netclamp_config)
    
    opt.execute(inputs, outputs)
        
@click.command()
@click.option(
    "--root-config", "-c", required=True, type=str, help="root configuration file name"
)
def job_grid(root_config):

    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise RuntimeError(f"netclamp_optimize.job_grid: unable to run in MPI mode with multiple ranks")

    root_config = get_config(root_config)

    env_config = root_config.get("Environment", {})
    operational_config = root_config["Network Clamp Optimization"]
    input_config = root_config["Inputs"]
    output_config = root_config["Outputs"]
    parameter_grid = root_config["Parameter Grid"]
    
    python_executable = env_config.get("python_executable", "python3")

    for params in parameter_grid:

        this_operational_config = operational_config.copy()
        this_operational_config.update(params)

        run_id = uuid.uuid4()
        operational_config['run_id'] = run_id

        job_input_path = os.path.join(inputs['job_input'], run_id)
        os.makedirs(job_input_path)
        job_output_path = os.path.join(inputs['job_output'], run_id)
        os.makedirs(job_output_path)

        inputs['job_input_path'] = job_input_path
        inputs['job_output_path'] = job_output_path
        
        operational_config_path = os.path.join(job_input_path, "operational_config.yaml")
        inputs_config_path = os.path.join(job_input_path, "inputs.yaml")
        outputs_config_path = os.path.join(job_input_path, "outputs.yaml")
        
        utils.write_to_yaml(operational_config_path, operational_config)
        utils.write_to_yaml(inputs_config_path, inputs)
        utils.write_to_yaml(outputs_config_path, outputs)

        cmd = Command(python_executable,
                      "-m", "dentate.experiment.netclamp_optimize",
                      f"--operational-config {operational_config_path}",
                      f"--inputs {inputs_config_path}",
                      f"--outputs {outputs_config_path}")
        cmd.run()
    
        
@click.command()
@click.option(
    "--root-config", "-c", required=True, type=str, help="root configuration file name"
)
def batch_job_grid(root_config):

    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise RuntimeError(f"netclamp_optimize.batch_job_grid: unable to run in MPI mode with multiple ranks")

    root_config = get_config(root_confi)

    resource_config = root_config["Resources"]
    env_config = root_config.get("Environment", {})
    operational_config = root_config["Network Clamp Optimization"]
    input_config = root_config["Inputs"]
    output_config = root_config["Outputs"]
    parameter_grid = root_config["Parameter Grid"]
    
    python_executable = env_config.get("python_executable", "python3")

    for params in parameter_grid:

        this_operational_config = operational_config.copy()
        this_operational_config.update(params)

        run_id = uuid.uuid4()
        operational_config['run_id'] = run_id

        job_input_path = os.path.join(inputs['job_input'], run_id)
        os.makedirs(job_input_path)
        job_output_path = os.path.join(inputs['job_output'], run_id)
        os.makedirs(job_output_path)

        inputs['job_input_path'] = job_input_path
        inputs['job_output_path'] = job_output_path
        
        operational_config_path = os.path.join(job_input_path, "operational_config.yaml")
        inputs_config_path = os.path.join(job_input_path, "inputs.yaml")
        outputs_config_path = os.path.join(job_input_path, "outputs.yaml")
        
        utils.write_to_yaml(operational_config_path, operational_config)
        utils.write_to_yaml(inputs_config_path, inputs)
        utils.write_to_yaml(outputs_config_path, outputs)

        slurm_job = Slurm(
            job_name=run_id,
            output=os.path.join(outputs['job_output_path'], f"{run_id}.%J.out"),
            **resource_config
        )

        cmd = Command(python_executable,
                      "-m", "dentate.experiment.netclamp_optimize",
                      f"--operational-config {operational_config_path}",
                      f"--inputs {inputs_config_path}",
                      f"--outputs {outputs_config_path}")
            
        slurm.sbatch(str(cmd))
    
cli.add_command(run)
cli.add_command(job_grid)
cli.add_command(batch_job_grid)

if __name__ == "__main__":
    cli(
        args=sys.argv[
            (
                utils.list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv)
                + 1
            ) :
        ],
        standalone_mode=False,
    )
