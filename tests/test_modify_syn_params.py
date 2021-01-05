from builtins import str

import click
from dentate.biophysics_utils import *
from dentate.plot import *
from dentate.synapses import modify_syn_param

context = Context()


def standard_modify_syn_param_tests(cell, env, syn_name='AMPA', param_name='g_unit', show=False):
    """

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param syn_name: str
    :param_name: str
    """
    start_time = time.time()
    gid = cell.gid
    pop_name = cell.pop_name
    init_syn_mech_attrs(cell, env)

    syn_attrs = env.synapse_attributes
    sec_type = 'apical'
    syn_mech_name = syn_attrs.syn_mech_names[syn_name]

    param_label = '%s; %s; %s' % (syn_name, syn_mech_name, 'weight')
    plot_synaptic_attribute_distribution(cell, env, syn_name, 'weight', filters=None, from_mech_attrs=True,
                                         from_target_attrs=False, param_label=param_label,
                                         export='syn_weights.hdf5', description='stage0', show=show, overwrite=True,
                                         output_dir=context.output_dir)
    config_biophys_cell_syns(env, gid, pop_name, insert=True, insert_netcons=True, insert_vecstims=True,
                             verbose=context.verbose)
    plot_synaptic_attribute_distribution(cell, env, syn_name, 'weight', filters=None, from_mech_attrs=True,
                                         from_target_attrs=True, param_label=param_label,
                                         export='syn_weights.hdf5', description='stage1', show=show,
                                         output_dir=context.output_dir)

    if param_name in syn_attrs.syn_param_rules[syn_mech_name]['netcon_params']:
        param_label = '%s; %s; %s' % (syn_name, syn_mech_name, param_name)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_attrs.hdf5',
                                             description='stage0', show=show, overwrite=True,
                                             output_dir=context.output_dir)
        modify_syn_param(cell, env, sec_type, syn_name, param_name=param_name, value=0.0005,
                         filters={'syn_types': ['excitatory']}, origin='soma', slope=0.0001, tau=50., xhalf=200.,
                         update_targets=True, verbose=context.verbose)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_attrs.hdf5',
                                             description='stage1', show=show, output_dir=context.output_dir)
        modify_syn_param(cell, env, sec_type, syn_name, param_name=param_name,
                         filters={'syn_types': ['excitatory'], 'layers': ['OML']}, origin='apical',
                         origin_filters={'syn_types': ['excitatory'], 'layers': ['MML']}, update_targets=True,
                         append=True, verbose=context.verbose)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_attrs.hdf5',
                                             description='stage2', show=show, output_dir=context.output_dir)
        if context.verbose:
            print('standard_modify_syn_param tests with cache_queries: %s took %.2f s' % \
                  (str(env.cache_queries), time.time() - start_time))
        plot_syn_attr_from_file(syn_name, param_name, 'syn_attrs.hdf5', param_label=param_label,
                                output_dir=context.output_dir)
    else:
        param_name = 'weight'
        param_label = '%s; %s; %s' % (syn_name, syn_mech_name, param_name)
        modify_syn_param(cell, env, sec_type, syn_name, param_name=param_name, value=0.0005,
                         filters={'syn_types': ['excitatory']}, origin='soma', slope=0.0001, tau=50., xhalf=200.,
                         update_targets=True, verbose=context.verbose)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_weights.hdf5',
                                             description='stage2', show=show, output_dir=context.output_dir)
        modify_syn_param(cell, env, sec_type, syn_name, param_name=param_name,
                         filters={'syn_types': ['excitatory'], 'layers': ['OML']}, origin='apical',
                         origin_filters={'syn_types': ['excitatory'], 'layers': ['MML']}, update_targets=True,
                         append=True, verbose=context.verbose)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_weights.hdf5',
                                             description='stage3', show=show, output_dir=context.output_dir)
        if context.verbose:
            print('standard_modify_syn_param tests with cache_queries: %s took %.2f s' % \
                  (str(env.cache_queries), time.time() - start_time))
    param_name = 'weight'
    param_label = '%s; %s; %s' % (syn_name, syn_mech_name, param_name)
    plot_syn_attr_from_file(syn_name, param_name, 'syn_weights.hdf5', param_label=param_label,
                            output_dir=context.output_dir)


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=str,
              default='Small_Scale_Control_tune_GC_synapses.yaml')
@click.option("--template-paths", type=str, default='../../DGC/Mateos-Aparicio2014:../templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='..')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../datasets')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../config')
@click.option("--mech-file", required=True, type=str, default='20181205_DG_GC_excitability_mech.yaml')
@click.option("--output-dir", type=str, default='data')
@click.option("--load-edges", is_flag=True)
@click.option("--load-synapses", is_flag=True)
@click.option("--load-weights", is_flag=True)
@click.option("--correct-for-spines", is_flag=True)
@click.option("--cache-queries", type=bool, default=False)
@click.option('--show', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, mech_file, output_dir,
         load_edges, load_synapses, load_weights, correct_for_spines, cache_queries, show, verbose):
    """

    :param gid: int
    :param pop_name: str
    :param config_file: str; model configuration file name
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    :param mech_file: str; cell mechanism config file name
    :param output_dir: str; path to directory to export data and figures
    :param load_edges: bool; whether to attempt to load connections from a neuroh5 file
    :param load_weights: bool; whether to attempt to load connections from a neuroh5 file
    :param correct_for_spines: bool
    :param verbose: bool
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm=comm, config_file=config_file, template_paths=template_paths, hoc_lib_path=hoc_lib_path,
              dataset_prefix=dataset_prefix, config_prefix=config_prefix, verbose=verbose,
              cache_queries=cache_queries)
    configure_hoc_env(env)
    mech_file_path = config_prefix + '/' + mech_file

    cell = make_biophys_cell(env, pop_name=pop_name, gid=gid, load_edges=load_edges, load_weights=load_weights,
                            mech_file_path=mech_file_path, load_synapses=load_synapses)
    init_biophysics(cell, reset_cable=True, correct_cm=correct_for_spines,
                    correct_g_pas=correct_for_spines, env=env,
                    verbose=verbose)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    context.update(locals())

    standard_modify_syn_param_tests(cell, env, show=show)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
