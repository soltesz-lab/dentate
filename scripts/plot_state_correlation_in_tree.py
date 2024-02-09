
import os, sys, click, re
import dentate
from dentate import env, plot, utils, cells, statedata
from dentate.neuron_utils import h, configure_hoc_env
from dentate.env import Env
from mpi4py import MPI

script_name = os.path.basename(__file__)

@click.command()
@click.option("--config-file", required=False, type=str)
@click.option("--config-prefix", required=False, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--state-path", '-p', required=True, type=click.Path())
@click.option("--state-namespace", '-n', type=str)
@click.option("--state-namespace-pattern", type=str)
@click.option("--cell-data-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='cell data file (overrides file provided in configuration)')
@click.option("--dataset-prefix", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='datasets',
              help='path to directory containing required neuroh5 data files')
@click.option("--reference-features-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--reference-features-namespace", type=str, default='Selectivity Features')
@click.option("--reference-features-arena-id", type=str)
@click.option("--reference-features-trajectory-id", type=str)
@click.option("--t-variable", type=str, default='t')
@click.option("--state-variable", type=str, default='v')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--population", required=True, type=str, help='target population')
@click.option("--gid", '-g', required=True, type=int, help='target cell gid')
@click.option("--query", "-q", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config_file, config_prefix, state_path, state_namespace, state_namespace_pattern, cell_data_path, dataset_prefix, reference_features_path, reference_features_namespace, reference_features_arena_id, reference_features_trajectory_id, t_variable, state_variable, t_max, t_min, population, gid, query, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
    
    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    namespace_id_lst, attr_info_dict = statedata.query_state(state_path, [population])

    if query:
        for this_namespace_id in namespace_id_lst:
            if this_namespace_id not in attr_info_dict[population]:
                continue
            print("Population %s; Namespace: %s" % (population, str(this_namespace_id)))
            for attr_name, attr_cell_index in attr_info_dict[population][this_namespace_id]:
                print("\tAttribute: %s" % str(attr_name))
                for i in attr_cell_index:
                    print("\t%d" % i)
        sys.exit()
    

    state_namespaces = []
    if state_namespace is not None:
        state_namespaces.append(state_namespace)
        
    if state_namespace_pattern is not None:
        for namespace_id in namespace_id_lst:
            m = re.match(state_namespace_pattern, namespace_id)
            if m:
                state_namespaces.append(namespace_id)

                
    plot.plot_state_correlation_in_tree (config_file,
                                         config_prefix,
                                         state_path,
                                         state_namespaces,
                                         reference_features_path,
                                         reference_features_namespace,
                                         reference_features_arena_id,
                                         reference_features_trajectory_id,
                                         population=population,
                                         gid=gid,
                                         cell_data_path=cell_data_path,
                                         dataset_prefix=dataset_prefix,
                                         time_range=time_range,
                                         time_variable=t_variable,
                                         state_variable=state_variable, 
                                         saveFig=True,
                                         )
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
