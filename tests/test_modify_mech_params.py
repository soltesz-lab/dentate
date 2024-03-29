from __future__ import division

from builtins import str

import click
from dentate.biophysics_utils import *
from dentate.plot import *
from dentate.cells import modify_mech_param, update_mechanism_by_sec_type, import_mech_dict_from_file

context = Context()


def compare_single_value(key, x, seg, mech_name, param_name):
    if not hasattr(seg, mech_name):
        print('Segment does not have the mechanism %s' % mech_name)
    else:
        model_val = getattr(getattr(seg, mech_name), param_name)
        exp_val = x[key]
        if model_val == exp_val:
            print('Test %s passed' % key)
        else:
            print('Test %s failed' % key)
            print('Model %s, Expected %s' % (format(model_val, 'e'), format(exp_val, 'e')))


def standard_modify_mech_param_tests(cell):
    """

    :param cell: :class:'BiophysCell'
    :return:
    """
    x = {'dend.g_pas slope': 1.058E-08, 'dend.g_pas tau': 3.886E+01, 'soma.g_pas': 1.050E-10, 'dend.gbar_nas': 0.03,
         'dend.gbar_nas bo': 4, 'dend.gbar_nas min': 0.0, 'dend.gbar_nas slope': -0.0001, 'dend.gkabar': 0.04,
         'soma.gkabar': 0.02108, 'axon.gkabar': 0.05266, 'soma.gbar_nas': 0.0308}

    modify_mech_param(cell, 'soma', 'pas', 'g', x['soma.g_pas'])
    soma_seg = cell.tree.root.sec(0.5)
    compare_single_value('soma.g_pas', x, soma_seg, 'pas', 'g')
    this_sec_types = ['soma', 'apical', 'axon']

    modify_mech_param(cell, 'apical', 'pas', 'g', origin='soma', slope=x['dend.g_pas slope'], tau=x['dend.g_pas tau'])
    for sec_type in ['hillock', 'ais', 'axon']:
        modify_mech_param(cell, sec_type, 'pas', 'g', origin='soma')
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas.hdf5', description='leak_config_1', show=False,
                                 sec_types=this_sec_types, param_label='leak conductance', overwrite=True,
                                 output_dir=context.output_dir)
    modify_mech_param(cell, 'soma', 'pas', 'g', 100000. * x['soma.g_pas'])
    for sec_type in ['hillock', 'ais', 'axon', 'apical']:
        update_mechanism_by_sec_type(cell, sec_type, 'pas')
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas.hdf5', description='leak_config_2', show=False,
                                 sec_types=this_sec_types, param_label='leak conductance',
                                 output_dir=context.output_dir)
    plot_mech_param_from_file('pas', 'g', 'dend_gpas.hdf5', param_label='leak conductance',
                              output_dir=context.output_dir)

    modify_mech_param(cell, 'soma', 'kap', 'gkabar', x['soma.gkabar'])
    slope = (x['dend.gkabar'] - x['soma.gkabar']) / 300.
    for sec_type in ['apical']:
        modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, outside=0.)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope,
                          outside=0., value=(x['soma.gkabar'] + slope * 75.))
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                          value=(x['soma.gkabar'] + slope * 300.), append=True)
    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='dend_ka.hdf5', description='kap_config_1',
                                 param_label='kap conductance', show=False, sec_types=['soma', 'apical'],
                                 overwrite=True, output_dir=context.output_dir)
    plot_mech_param_distribution(cell, 'kad', 'gkabar', export='dend_ka.hdf5', description='kad_config_1',
                                 param_label='kad conductance', show=False, sec_types='dend',
                                 output_dir=context.output_dir)

    update_mechanism_by_sec_type(cell, 'axon_hill', 'kap')  # should do nothing
    modify_mech_param(cell, 'ais', 'kap', 'gkabar', x['axon.gkabar'])
    modify_mech_param(cell, 'axon', 'kap', 'gkabar', origin='ais')
    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='dend_ka.hdf5', description='kap_config_2',
                                 param_label='kap conductance', show=False, sec_types=this_sec_types,
                                 output_dir=context.output_dir)
    plot_mech_param_from_file('kap', 'gkabar', 'dend_ka.hdf5', output_dir=context.output_dir)  # , param_label='kap conductance')
    plot_mech_param_from_file('kad', 'gkabar', 'dend_ka.hdf5', output_dir=context.output_dir)  # , param_label='kad conductance')

    modify_mech_param(cell, 'soma', 'nas', 'gbar', x['soma.gbar_nas'])
    modify_mech_param(cell, 'apical', 'nas', 'gbar', x['dend.gbar_nas'])
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='dend_nas.hdf5', description='nas_config_1', show=False,
                                 sec_types=['apical'], overwrite=True, output_dir=context.output_dir)
    modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x['dend.gbar_nas slope'],
                      min=x['dend.gbar_nas min'],
                      custom={'func': 'custom_filter_by_branch_order',
                              'branch_order': x['dend.gbar_nas bo']}, append=True)
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='dend_nas.hdf5', description='nas_config_2', show=False,
                                 sec_types=['apical'], output_dir=context.output_dir)
    modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x['dend.gbar_nas slope'],
                      min=x['dend.gbar_nas min'], custom={'func': 'custom_filter_modify_slope_if_terminal'},
                      append=True)
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='dend_nas.hdf5', description='nas_config_3', show=False,
                                 sec_types=['apical'], output_dir=context.output_dir)
    plot_mech_param_from_file('nas', 'gbar', 'dend_nas.hdf5', param_label='nas conductance',
                              descriptions=['nas_config_1', 'nas_config_2', 'nas_config_3'],
                              output_dir=context.output_dir)


def count_nseg(cell):
    nseg = defaultdict(list)
    distances = defaultdict(list)
    for sec_type in cell.nodes:
        for node in cell.nodes[sec_type]:
            nseg[sec_type].append(node.sec.nseg)
            distances[sec_type].append(get_distance_to_node(cell, cell.tree.root, node))
    return nseg, distances


def compare_nseg(nseg, distances, labels):
    """

    :param nseg: list of dict: {str: int}
    :param distances: list of dict: {str: float}
    :param labels: list of str
    """
    num_colors = max([len(this_nseg) for this_nseg in nseg])
    markers = mlines.Line2D.filled_markers
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    fig = plt.figure()
    for j, this_nseg in enumerate(nseg):
        for i, sec_type in enumerate(this_nseg):
            this_distances = distances[j]
            plt.scatter(this_distances[sec_type], this_nseg[sec_type], color=colors[j], marker=markers[i],
                        label=sec_type + '_' + labels[j], alpha=0.5)
            print('%s_%s nseg: %s' % (sec_type, labels[j], str(this_nseg[sec_type])))
    plt.legend(loc='best', frameon=False, framealpha=0.5)
    plt.xlabel('Distance from Soma (um)')
    plt.ylabel('Number of segments per section')
    plt.title('Changing Spatial Resolution')
    fig.show()


def cm_correction_test(cell, env, mech_file_path):
    """

    :param cell:
    :param env:
    :param mech_file_path:
    """
    import_mech_dict_from_file(cell, mech_file_path)
    init_biophysics(cell, reset_cable=True, correct_cm=False, correct_g_pas=False, env=env, verbose=context.verbose)
    old_nseg, old_distances = count_nseg(cell)
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas.hdf5', overwrite=True,
                                 param_label='dend.g_pas', show=False, output_dir=context.output_dir)
    plot_cable_param_distribution(cell, 'cm', export='cm.hdf5', param_label='cm', show=False, overwrite=True,
                                  output_dir=context.output_dir)
    init_biophysics(cell, reset_cable=True, reset_mech_dict=True, correct_cm=True,
                    correct_g_pas=True, env=env, verbose=context.verbose)
    new_nseg, new_distances = count_nseg(cell)
    compare_nseg([old_nseg, new_nseg], [old_distances, new_distances], ['before', 'after'])
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas.hdf5', param_label='dend.g_pas', show=False,
                                 output_dir=context.output_dir)
    plot_mech_param_from_file('pas', 'g', 'dend_gpas.hdf5', param_label='dend.gpas')
    plot_cable_param_distribution(cell, 'cm', export='cm.hdf5', param_label='cm', show=False,
                                  output_dir=context.output_dir)
    plot_mech_param_from_file('cm', None, 'cm.hdf5', param_label='cm')


def standard_cable_tests(cell, mech_file_path):
    """

    :param cell: :class:'BiophysCell'
    :param mech_file_path: str
    """
    import_mech_dict_from_file(cell, mech_file_path)
    init_biophysics(cell, reset_cable=True, verbose=context.verbose)
    plot_cable_param_distribution(cell, 'cm', export='cm.hdf5', show=False, overwrite=True,
                                  output_dir=context.output_dir)
    modify_mech_param(cell, 'soma', 'cable', 'cm', value=2.)
    init_biophysics(cell, reset_cable=True, verbose=context.verbose)
    plot_cable_param_distribution(cell, 'cm', export='cm.hdf5', show=False, output_dir=context.output_dir)
    plot_mech_param_from_file('cm', None, 'cm.hdf5', param_label='cm', yunits='uF/cm2', ylabel='Specific capacitance')

    init_biophysics(cell, reset_cable=True, reset_mech_dict=True, verbose=context.verbose)
    plot_cable_param_distribution(cell, 'Ra', export='Ra.hdf5', show=False, overwrite=True,
                                  output_dir=context.output_dir)
    modify_mech_param(cell, 'soma', 'cable', 'Ra', value=200.)
    init_biophysics(cell, reset_cable=True, verbose=context.verbose)
    plot_cable_param_distribution(cell, 'Ra', export='Ra.hdf5', show=False, output_dir=context.output_dir)
    plot_mech_param_from_file('Ra', None, 'Ra.hdf5', param_label='Ra', yunits='Ohm*cm', ylabel='Axial resistivity')

    init_biophysics(cell, reset_cable=True, reset_mech_dict=True, verbose=context.verbose)
    old_nseg, old_distances = count_nseg(cell)
    modify_mech_param(cell, 'soma', 'cable', 'spatial_res', value=2.)
    init_biophysics(cell, reset_cable=True, verbose=context.verbose)
    new_nseg, new_distances = count_nseg(cell)
    compare_nseg([old_nseg, new_nseg], [old_distances, new_distances], ['before', 'after'])

    init_biophysics(cell, reset_cable=True, reset_mech_dict=True, verbose=context.verbose)
    plot_cable_param_distribution(cell, 'cm', export='cm.hdf5', show=False, overwrite=True,
                                  output_dir=context.output_dir)
    modify_mech_param(cell, 'apical', 'cable', 'cm', value=2.)
    init_biophysics(cell, reset_cable=True, verbose=context.verbose)
    plot_cable_param_distribution(cell, 'cm', export='cm.hdf5', show=False, output_dir=context.output_dir)
    nseg2, distances2 = count_nseg(cell)
    modify_mech_param(cell, 'apical', 'cable', 'spatial_res', value=2.)
    plot_cable_param_distribution(cell, 'cm', export='cm.hdf5', show=False, output_dir=context.output_dir)
    nseg3, distances3 = count_nseg(cell)
    plot_mech_param_from_file('cm', None, 'cm.hdf5', param_label='cm', yunits='uF/cm2', ylabel='Specific capacitance')
    compare_nseg([nseg2, nseg3], [distances2, distances3], ['post step 2', 'post step 3'])


def count_spines(cell, env):
    """

    :param cell:
    :param env:
    """
    init_biophysics(cell, env, reset_cable=True, correct_cm=True, verbose=context.verbose)
    gid = cell.gid
    syn_attrs = env.synapse_attributes
    num_spines_list = []
    distances = []
    for node in cell.apical:
        num_spines = len(syn_attrs.filter_synapses(gid, syn_sections=[node.index],
                                                   syn_types=[env.Synapse_Types['excitatory']]))
        stored_num_spines = sum(node.spine_count)
        if num_spines != stored_num_spines:
            raise ValueError('count_spines_test: failed for node: %s; num spines %i != stored %i' %
                             (node.name, num_spines, stored_num_spines))
        num_spines_list.append(num_spines)
        distances.append(get_distance_to_node(cell, cell.tree.root, node, 0.5))
        print('count_spines_test: passed for node: %s; nseg: %i; L: %.2f um; spine_count: %i; density: %.2f /um' %
              (node.name, node.sec.nseg, node.sec.L, num_spines, num_spines / node.sec.L))
    fig, axes = plt.subplots()
    axes.scatter(distances, num_spines_list)
    axes.set_xlabel('Distance from soma (um)')
    axes.set_ylabel('Spine count')
    clean_axes(axes)
    fig.show()


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=str,
              default='Small_Scale_Control_tune_GC_synapses.yaml')
@click.option("--template-paths", type=str, default='../DGC/Mateos-Aparicio2014:templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='.')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='datasets')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--mech-file", required=True, type=str, default='20181205_DG_GC_excitability_mech.yaml')
@click.option("--output-dir", type=str, default='data')
@click.option("--load-edges", is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, mech_file, output_dir,
         load_edges, verbose):
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
    :param load_edges: bool; whether to attempt to load connections for neuroh5
    :param verbose: bool
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm=comm, config=config_file, template_paths=template_paths, hoc_lib_path=hoc_lib_path,
              dataset_prefix=dataset_prefix, config_prefix=config_prefix, verbose=verbose)
    configure_hoc_env(env)
    mech_file_path = config_prefix + '/' + mech_file

    cell = make_biophys_cell(env, pop_name=pop_name, gid=gid, load_edges=load_edges)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    context.update(locals())

    standard_modify_mech_param_tests(cell)
    standard_cable_tests(cell, mech_file_path)
    cm_correction_test(cell, env, mech_file_path)
    count_spines(cell, env)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
