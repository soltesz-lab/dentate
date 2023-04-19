import os
import stat
import sys
import uuid
import re
from dataclasses import dataclass
from typing import Any, Tuple, Optional, Union, List, Dict, Set
from dentate import network_clamp, utils, optimize_selectivity_snr
from mpi4py import MPI
from simple_slurm import Slurm
from commandlib import Command
import click
import mlflow

def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


@dataclass
class OptimizeSelectivitySNR:
    run_id: Optional[Any] = None
    comm: Optional[MPI.Intracomm] = None
    config: Optional[str] = None
    gid: Optional[int] = None
    population: str = "GC"
    dt: Optional[float] = None
    arena_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    generate_weights: Optional[Tuple[str]] = None
    t_max: Optional[float] = 150.0
    t_min: Optional[float] = None
    nprocs_per_worker: int = 1
    param_config_name: Optional[str] = None
    selectivity_config_name: Optional[str] = None
    param_type: str = "synaptic"
    spike_events_namespace: str = "Spike Events"
    spike_events_t: str = "t"
    phase_mod: bool = False
    input_features_namespaces: Tuple[str] =  ("Place Selectivity", "Grid Selectivity")
    n_iter: int = 10
    n_trials: int = 1
    trial_regime: str = ("mean",)
    problem_regime: str = "every"
    target_features_namespace: str = "Input Spikes"
    use_coreneuron: bool = False


    def execute(self, inputs, outputs):
        comm = self.comm
        model_configuration = None
        if (comm is None) or ((comm is not None) and (comm.rank == 0)):
            config_path = os.path.join(inputs.get("config_prefix", ""), self.config)
            model_configuration = utils.read_from_yaml(config_path,
                                                       include_loader=utils.IncludeLoader)
        if comm is not None:
            model_configuration = comm.bcast(model_configuration, root=0)

        generate_weights = self.generate_weights
        if generate_weights is None:
            generate_weights = set([])

        os.makedirs(outputs["job_output"], exist_ok=True)

        param_results_file=outputs.get("param_results_file",
                                       f"optimize_selectivity_snr.{self.run_id}.yaml")
        results_file=outputs.get("results_file",
                                 f"optimize_selectivity_snr.{self.run_id}.h5")
        param_results_path=os.path.join(outputs["job_output"], param_results_file)
        results_path=os.path.join(outputs["job_output"], results_file)
        
        results_dict, distgfs_params = optimize_selectivity_snr.main(
            config=model_configuration,
            config_prefix=inputs.get("config_prefix", "config"),
            population=self.population,
            dt=self.dt,
            gid=self.gid,
            arena_id=self.arena_id,
            trajectory_id=self.trajectory_id,
            generate_weights=generate_weights,
            t_max=self.t_max,
            t_min=self.t_min,
            nprocs_per_worker=self.nprocs_per_worker,
            template_paths=inputs.get("template_path", "templates"),
            dataset_prefix=inputs.get("dataset_prefix", None),
            param_config_name=self.param_config_name,
            selectivity_config_name=self.selectivity_config_name,
            param_type=self.param_type,
            param_results_file=param_results_file,
            results_file=results_file,
            results_path=outputs["job_output"],
            spike_events_path=inputs.get("spike_events", None),
            spike_events_namespace=self.spike_events_namespace,
            spike_events_t=self.spike_events_t,
            input_features_path=inputs.get("input_features", None),
            input_features_namespaces=self.input_features_namespaces,
            #phase_mod=self.phase_mod,
            n_iter=self.n_iter,
            n_trials=self.n_trials,
            trial_regime=self.trial_regime,
            problem_regime=self.problem_regime,
            target_features_path=inputs.get("target_features", None),
            target_features_namespace=self.target_features_namespace,
            use_coreneuron=self.use_coreneuron,
            n_max_tasks=-1,
            infld_threshold=1e-3,
            cooperative_init=False,
            spawn_workers=False,
            spawn_startup_wait=None,
            spawn_executable=None,
            spawn_args=[],
            verbose=True,
        )

        if results_dict is not None:
            for population in results_dict:
                mlflow.set_experiment(f"Optimize Selectivity SNR {population}")
                mlflow.set_experiment_tag("population", population)
                if self.problem_regime == 'every':
                    with mlflow.start_run(run_name=str(self.run_id)):
                        for gid in results_dict[population]:
                            with mlflow.start_run(run_name=str(gid), nested=True):
                                param_dict = { re.sub(r'\W+|^(?=\d)','_', x): v for x, v in results_dict[population][gid].parameters }
                                mlflow.log_params(param_dict)
                                mlflow.log_metrics(results_dict[population][gid].objectives)
                                mlflow.log_artifact(param_results_path)
                                mlflow.log_artifact(results_path)
                else:
                    with mlflow.start_run(run_name=str(self.run_id)):
                        param_dict = { re.sub(r'\W+|^(?=\d)','_', x): v for x, v in results_dict[population][gid].parameters }
                        mlflow.log_params(param_dict)
                        mlflow.log_metrics(results_dict[population].objectives)
                        mlflow.log_artifact(param_results_path)
                        mlflow.log_artifact(results_path)

def mpi_get_config(config):
    comm = MPI.COMM_WORLD
    if isinstance(config, str):
        config_dict = None
        if comm.rank == 0:
            config_dict = utils.read_from_yaml(config, include_loader=utils.IncludeLoader)
        config_dict = comm.bcast(config_dict, root=0)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise RuntimeError(f"mpi_get_config: unable to process configuration object {config}")
    #config_dict = utils.yaml_envsubst(config_dict)
    return config_dict

def get_config(config):
    if isinstance(config, str):
        config_dict = utils.read_from_yaml(config, include_loader=utils.IncludeLoader)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise RuntimeError(f"get_config: unable to process configuration object {config}")
    #config_dict = utils.yaml_envsubst(config_dict)
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
    
    opt = OptimizeSelectivitySNR(**netclamp_config)
    
    opt.execute(inputs, outputs)
        
@click.command()
@click.option(
    "--root-config", "-c", required=True, type=str, help="root configuration file name"
)
def job_grid(root_config):

    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise RuntimeError(f"optimize_selectivity_snr.job_grid: unable to run in MPI mode with multiple ranks")

    root_config = get_config(root_config)

    env_config = root_config.get("Environment", {})
    operational_config = root_config["Selectivity Optimization SNR"]
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
        os.makedirs(job_input_path, exist_ok=True)
        job_output_path = os.path.join(outputs['job_output'], run_id)
        os.makedirs(job_output_path, exist_ok=True)

        inputs['job_input_path'] = job_input_path
        inputs['job_output_path'] = job_output_path
        
        operational_config_path = os.path.join(job_input_path, "operational_config.yaml")
        inputs_config_path = os.path.join(job_input_path, "inputs.yaml")
        outputs_config_path = os.path.join(job_input_path, "outputs.yaml")

        
        utils.write_to_yaml(operational_config_path, this_operational_config)
        utils.write_to_yaml(inputs_config_path, inputs)
        utils.write_to_yaml(outputs_config_path, outputs)

        cmd = Command(python_executable,
                      "-m", "dentate.experiment.optimize_selectivity_snr",
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
        raise RuntimeError(f"optimize_selectivity_snr.batch_job_grid: unable to run in MPI mode with multiple ranks")

    root_config = get_config(root_config)

    resource_config = root_config["Resources"]
    env_config = root_config.get("Environment", {})
    operational_config = root_config["Selectivity Optimization SNR"]
    input_config = root_config["Inputs"]
    output_config = root_config["Outputs"]
    parameter_grid = root_config["Parameter Grid"]

    env_modules = env_config.get("modules", [])
    env_variables = env_config.get("variables", {})
    env_executables = env_config.get("executables", {})
    python_executable = env_executables.get("python", "python3")
    mpirun_executable = env_executables.get("mpiexec", "mpiexec")

    for params in parameter_grid:

        this_operational_config = operational_config.copy()
        this_operational_config.update(params)

        run_id = uuid.uuid4()
        operational_config['run_id'] = str(run_id)

        job_input_path = os.path.join(input_config['job_input'], str(run_id))
        os.makedirs(job_input_path, exist_ok=True)
        job_output_path = os.path.join(output_config['job_output'], str(run_id))
        os.makedirs(job_output_path, exist_ok=True)

        input_config['job_input_path'] = job_input_path
        output_config['job_output_path'] = job_output_path
        
        operational_config_path = os.path.join(job_input_path, "operational_config.yaml")
        inputs_config_path = os.path.join(job_input_path, "inputs.yaml")
        outputs_config_path = os.path.join(job_input_path, "outputs.yaml")
        output_script_path = os.path.join(job_input_path, "script.sh")
        
        utils.write_to_yaml(operational_config_path, this_operational_config)
        utils.write_to_yaml(inputs_config_path, input_config)
        utils.write_to_yaml(outputs_config_path, output_config)

        job_name = f"optimize_selectivity_snr_{str(run_id)}"
        slurm_job = Slurm(
            job_name=job_name,
            output=os.path.join(output_config['job_output_path'], f"{job_name}.%J.out"),
            **resource_config
        )
        ntasks = resource_config.get("ntasks", 1)

        cmd = Command(mpirun_executable, "-v", 
                      python_executable,
                      "-m", "dentate.experiment.optimize_selectivity_snr",
                      "run",
                      f"--operational-config {operational_config_path}",
                      f"--inputs {inputs_config_path}",
                      f"--outputs {outputs_config_path}")

        pre_cmds = []
        
        for mod in env_modules:
            module_cmd = f"module load {mod}"
            pre_cmds.append(module_cmd)

        for k,v in env_variables.items():
            env_cmd = f"export {k}={v}"
            pre_cmds.append(env_cmd)

        pre_cmds.append("set")
        pre_cmds.append("echo $PWD")
        pre_cmds.append("set -x")

        post_cmds = []

        cmd_block = "\n".join(pre_cmds) + "\n" + str(cmd) + "\n" + "\n".join(post_cmds)


        with open(output_script_path, 'w') as f:
            f.write(cmd_block)
        st = os.stat(output_script_path)
        os.chmod(output_script_path, st.st_mode | stat.S_IEXEC)

        slurm_job.sbatch(output_script_path, shell="/bin/bash", verbose=True)

        
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
