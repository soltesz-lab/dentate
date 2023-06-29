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
run_parameters = {}
parameter_keys = None


for run in runs:
    run_dict = run.to_dictionary()
    print(run_dict)
    metrics = run_dict["data"]["metrics"]
    parameters = run_dict["data"]["params"]
    run_name = run_dict["info"]["run_name"]
    start_time = run_dict["info"]["start_time"]
    start_dt = datetime.fromtimestamp(start_time / 1000.)
    if start_dt < time_threshold:
        continue
    if len(parameters) > 0:
        if parameter_keys is None:
            parameter_keys = tuple(parameters.keys())
        if any([x not in parameters for x in parameter_keys]):
            continue
        run_parameters[int(run_name)] = np.asarray([parameters[x] for x in parameter_keys])
    if 'objective' in metrics:
        run_objectives[int(run_name)] = float(metrics['objective'])

        
coords = np.loadtxt("coords.GC.csv",skiprows=1,delimiter=",")
coords_dict = {}
for r in coords:
    gid = int(r[0])
    x = r[4]
    y = r[5]
    coords_dict[gid] = np.asarray((x, y))

coords_objectives_params = {}
for gid in run_objectives:
    run
    coords_objectives_params[gid] = np.concatenate(([run_objectives[gid]], coords_dict[gid], run_parameters[gid]))
    
df = pd.DataFrame.from_dict(coords_objectives_params, orient='index', columns=('SNR score', 'U Distance', 'V Distance')+parameter_keys)
df.to_csv('coords_snr_parameters.csv')
    
