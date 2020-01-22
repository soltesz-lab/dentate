
import os, sys, click, re
import dentate
from dentate import plot, utils, statedata
from mpi4py import MPI

script_name = os.path.basename(__file__)

@click.command()
@click.option("--state-path", '-p', required=True, type=click.Path())
@click.option("--state-namespace", '-n', type=str)
@click.option("--state-namespace-pattern", '-n', type=str)
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--max-units", type=int, default=1)
@click.option("--unit-no", '-u', type=int, default=None, multiple=True)
@click.option("--t-variable", type=str, default='t')
@click.option("--state-variable", type=str, default='v')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--query", "-q", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(state_path, state_namespace, state_namespace_pattern, populations, max_units, unit_no, t_variable, state_variable, t_max, t_min, font_size, query, verbose):

    utils.config_logging(verbose)

    
    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if not populations:
        populations = ['eachPop']
    else:
        populations = list(populations)

    namespace_id_lst = statedata.query_state(state_path, populations, namespace_id=state_namespace)
    if query:
        for this_namespace_id in namespace_id_lst:
            print("Namespace: %s" % str(this_namespace_id))
            #for attr_name, attr_cell_index in attr_info_dict[pop_name][this_namespace_id]:
            #    print("\tAttribute: %s" % str(attr_name))
            #    for i in attr_cell_index:
            #        print("\t%d" % i)
        sys.exit()

    state_namespaces = []
    if state_namespace is not None:
        state_namespaces.append(state_namespace)
        
    if state_namespace_pattern is not None:
        for namespace_id in namespace_id_lst:
            m = re.match(state_namespace_pattern, namespace_id)
            if m:
                state_namespaces.append(namespace_id)
        
    if len(unit_no) == 0:
        unit_no = None

    plot.plot_intracellular_state (state_path, state_namespaces, include=populations, time_range=time_range,
                                   time_variable=t_variable, state_variable=state_variable,
                                   max_units=max_units, unit_no=unit_no,
                                   fontSize=font_size, saveFig=True)
    


    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
