import os, glob, shutil
from urllib.parse import urlparse
import mlflow
import pandas as pd
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import ViewType
from datetime import datetime


time_threshold = datetime(2023, 6, 20, 0, 0, 0, 0)


experiment = mlflow.get_experiment_by_name("Optimize Selectivity SNR GC")
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))


runs = MlflowClient().search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["metrics.objective DESC"],
)
print("Number of runs: {}".format(len(runs)))

run_objectives = {}
run_opt_configs = {}
parameter_keys = None

phenotype_gids = [145775, 999687, 383806, 960280, 841709, 124089]
                 
opt_config_paths = {}
                 
for run in runs:
    run_dict = run.to_dictionary()
    run_id = run_dict["info"]["run_id"]
    run_name = run_dict["info"]["run_name"]
    artifact_uri = run_dict["info"]["artifact_uri"]
    start_time = run_dict["info"]["start_time"]
    start_dt = datetime.fromtimestamp(start_time / 1000.)
    if start_dt < time_threshold:
        continue

    artifact_path = urlparse(artifact_uri).path
    opt_config_path = None
    opt_config_glob = glob.glob(f'{artifact_path}/optimize_selectivity_snr.*.yaml')
    try:
        run_gid = int(run_name)
    except:
        run_gid = None
    if len(opt_config_glob) > 0 and run_gid in phenotype_gids:
        opt_config_path = opt_config_glob[0]
        print(f"Run {run_name}: {opt_config_path}")
        opt_config_dst_path = f"results/optimize_selectivity_snr/20230628/{run_name}"
        os.makedirs(opt_config_dst_path, exist_ok=True)
        shutil.copy(opt_config_path, opt_config_dst_path)
        opt_config_paths[run_gid] = os.path.join(opt_config_dst_path, os.path.basename(opt_config_path))
        
with open('results/optimize_selectivity_snr/20230628/all_opt_configs.yaml', 'a') as out:
    
    for run_gid in sorted(opt_config_paths):
        opt_config_path = opt_config_paths[run_gid]
        print(f"Reading input {opt_config_path}")
        with open(opt_config_path, 'r') as inp:
            shutil.copyfileobj(inp, out)
                    

