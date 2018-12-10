import click
from dentate.biophysics_utils import *
from dentate.plot import *


context = Context()


def standard_modify_syn_mech_param_tests(cell, env, syn_name='AMPA', param_name='g_unit'):
    """

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param syn_name: str
    :param_name: str
    """
    gid = cell.gid
    pop_name = cell.pop_name
    init_syn_mech_attrs(cell, env)
    config_biophys_cell_syns(env, gid, pop_name, insert=True, insert_netcons=True, insert_vecstims=True)
    syn_attrs = env.synapse_attributes
    sec_type = 'apical'
    syn_mech_name = syn_attrs.syn_mech_names[syn_name]

    param_label = '%s; %s; %s' % (syn_name, syn_mech_name, 'weight')
    plot_synaptic_attribute_distribution(cell, env, syn_name, 'weight', filters=None, from_mech_attrs=False,
                                             from_target_attrs=True, param_label=param_label,
                                             export='syn_weights.hdf5', description='stage0', show=False,
                                             overwrite=True)

    if param_name in syn_attrs.syn_param_rules[syn_mech_name]['netcon_params']:
        param_label = '%s; %s; %s' % (syn_name, syn_mech_name, param_name)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_attrs.hdf5',
                                             description='stage0', show=False, overwrite=True)
        modify_syn_mech_param(cell, env, sec_type, syn_name, param_name=param_name, value=0.0005,
                              filters={'syn_types': ['excitatory']}, origin='soma', slope=0.0001, tau=50., xhalf=200.,
                              update_targets=True)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_attrs.hdf5',
                                             description='stage1', show=False)
        modify_syn_mech_param(cell, env, sec_type, syn_name, param_name=param_name,
                              filters={'syn_types': ['excitatory'], 'layers': ['OML']}, origin='apical',
                              origin_filters={'syn_types': ['excitatory'], 'layers': ['MML']}, update_targets=True,
                              append=True)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_attrs.hdf5',
                                             description='stage2', show=False)
        plot_syn_attr_from_file(syn_name, param_name, 'syn_attrs.hdf5', param_label=param_label)
    else:
        param_name = 'weight'
        param_label = '%s; %s; %s' % (syn_name, syn_mech_name, param_name)
        modify_syn_mech_param(cell, env, sec_type, syn_name, param_name=param_name, value=0.0005,
                              filters={'syn_types': ['excitatory']}, origin='soma', slope=0.0001, tau=50., xhalf=200.,
                              update_targets=True)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_weights.hdf5',
                                             description='stage1', show=False)
        modify_syn_mech_param(cell, env, sec_type, syn_name, param_name=param_name,
                              filters={'syn_types': ['excitatory'], 'layers': ['OML']}, origin='apical',
                              origin_filters={'syn_types': ['excitatory'], 'layers': ['MML']}, update_targets=True,
                              append=True)
        plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                             from_target_attrs=True, param_label=param_label, export='syn_weights.hdf5',
                                             description='stage2', show=False)
    param_name = 'weight'
    param_label = '%s; %s; %s' % (syn_name, syn_mech_name, param_name)
    plot_syn_attr_from_file(syn_name, param_name, 'syn_weights.hdf5', param_label=param_label)


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=str,
              default='Small_Scale_Control_LN_weights_Sat.yaml')
@click.option("--template-paths", type=str, default='../../DGC/Mateos-Aparicio2014:../templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='..')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../datasets')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../config')
@click.option("--mech-file", required=True, type=str, default='20181205_DG_GC_excitability_mech.yaml')
@click.option("--load-edges", type=bool, default=True)
@click.option("--load-weights", is_flag=True)
@click.option("--correct-for-spines", type=bool, default=True)
@click.option('--verbose', '-v', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, mech_file,
         load_edges, load_weights, correct_for_spines, verbose):
    """

    :param gid: int
    :param pop_name: str
    :param config_file: str; model configuration file name
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    :param mech_file: str; cell mechanism config file name
    :param load_edges: bool; whether to attempt to load connections from a neuroh5 file
    :param load_weights: bool; whether to attempt to load connections from a neuroh5 file
    :param correct_for_spines: bool
    :param verbose: bool
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, verbose=verbose)
    configure_hoc_env(env)

    context.update(locals())

    cell = get_biophys_cell(env, pop_name=pop_name, gid=gid, load_edges=load_edges, load_weights=load_weights)
    mech_file_path = config_prefix + '/' + mech_file
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path,
                    correct_cm=correct_for_spines, correct_g_pas=correct_for_spines, env=context.env)
    context.update(locals())

    standard_modify_syn_mech_param_tests(cell, env)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
