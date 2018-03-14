#Expects there to already be a hoc cell with a python wrapper (as defined in cells.py); the python cell should be called cell.
from cells import *
from optimize_cells.plot_results import *
import matplotlib.pyplot as plt

def param_dict_to_array(x_dict, param_names):
    """

    :param x_dict: dict
    :param param_names: list
    :return:
    """
    return np.array([x_dict[param_name] for param_name in param_names])

def compare_single_value(name, x, param_indexes, seg, mech_name, param_name):
    if not hasattr(seg, mech_name):
        print 'Segment does not have the mechanism %s' %mech_name
    else:
        model_val = getattr(getattr(seg, mech_name), param_name)
        exp_val = x[param_indexes['soma.g_pas']]
        if model_val == exp_val:
            print 'Test %s passed' %name
        else:
            print 'Test %s failed' %name
            print 'Model %s, Expected %s' %(format(model_val, 'e'), format(exp_val, 'e'))


def run_normal_tests(cell):
    x_dict = {'dend.g_pas slope': 1.058E-08, 'dend.g_pas tau': 3.886E+01, 'soma.g_pas': 1.050E-10, 'dend.gbar_nas': 0.03,
              'dend.gbar_nas bo': 4, 'dend.gbar_nas min': 0.0, 'dend.gbar_nas slope': -0.0001, 'dend.gkabar': 0.04,
              'soma.gkabar': 0.02108, 'axon.gkabar': 0.05266}
    param_names = x_dict.keys()
    param_indexes = {param_name: i for i, param_name in enumerate(param_names)}
    x = param_dict_to_array(x_dict, param_names)

    modify_mech_param(cell, 'soma', 'pas', 'g', x[param_indexes['soma.g_pas']])
    soma_seg = list(cell.nodes['soma'][0].sec)[0]
    compare_single_value('soma.g_pas', x, param_indexes, soma_seg, 'pas', 'g')

    plot_mech_param_distribution(cell, 'pas', 'g', export='old_dend_gpas.hdf5', param_label='dend.g_pas', show=False)
    modify_mech_param(cell, 'apical', 'pas', 'g', origin='soma', slope=x[param_indexes['dend.g_pas slope']],
                      tau=x[param_indexes['dend.g_pas tau']])
    for sec_type in ['hillock', 'axon', 'ais', 'apical', 'spine_neck', 'spine_head']:
        reinitialize_subset_mechanisms(cell, sec_type, 'pas')
    plot_mech_param_distribution(cell, 'pas', 'g', export='new_dend_gpas.hdf5', param_label='dend.g_pas', show=False)
    plot_mech_param_from_file('pas', 'g', ['old_dend_gpas.hdf5', 'new_dend_gpas.hdf5'], ['old', 'new'],
                              param_label='dend.gpas')

    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='old_dend_kap.hdf5', param_label='dend.kap', show=False)
    plot_mech_param_distribution(cell, 'kad', 'gkabar', export='old_dend_kad.hdf5', param_label='dend.kad',
                                 show=False)
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='old_dend_nas.hdf5', param_label='dend.nas',
                                 show=False)
    slope = (x[param_indexes['dend.gkabar']] - x[param_indexes['soma.gkabar']]) / 300.
    for sec_type in ['apical']:
        reinitialize_subset_mechanisms(cell, sec_type, 'nas')
        modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma', min_loc=75., value=0.)
        modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, replace=False)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', max_loc=75., value=0.)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope,
                               value=(x[param_indexes['soma.gkabar']] + slope * 75.), replace=False)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                               value=(x[param_indexes['soma.gkabar']] + slope * 300.), replace=False)
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x[param_indexes['dend.gbar_nas slope']],
                               min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_branch_order',
                                       'branch_order': x[param_indexes['dend.gbar_nas bo']]}, replace=False)
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent',
                               slope=x[param_indexes['dend.gbar_nas slope']], min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_terminal'}, replace=False)
    reinitialize_subset_mechanisms(cell, 'axon_hill', 'kap')
    modify_mech_param(cell, 'ais', 'kap', 'gkabar', x[param_indexes['axon.gkabar']])
    modify_mech_param(cell, 'axon', 'kap', 'gkabar', origin='ais')
    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='new_dend_kap.hdf5', param_label='dend.kap', show=False)
    plot_mech_param_distribution(cell, 'kad', 'gkabar', export='new_dend_kad.hdf5', param_label='dend.kad',
                                 show=False)
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='new_dend_nas.hdf5', param_label='dend.nas',
                                 show=False)
    plot_mech_param_from_file('kap', 'gkabar', ['old_dend_kap.hdf5', 'new_dend_kap.hdf5'], ['old', 'new'],
                              param_label='dend.kap')
    plot_mech_param_from_file('kad', 'gkabar', ['old_dend_kad.hdf5', 'new_dend_kad.hdf5'], ['old', 'new'],
                              param_label='dend.kad')
    plot_mech_param_from_file('nas', 'gbar', ['old_dend_nas.hdf5', 'new_dend_nas.hdf5'], ['old', 'new'],
                              param_label='dend.nas')

def count_nseg(cell):
    nseg = {}
    distances = {}
    for sec_type in cell.nodes:
        nseg[sec_type] = []
        distances[sec_type] = []
        for node in cell.nodes[sec_type]:
            nseg[sec_type].append(node.sec.nseg)
            distances[sec_type].append(get_distance_to_node(cell, cell.tree.root, node))
    return nseg, distances


def compare_nseg(old_nseg, old_distances, new_nseg, new_distances, label_old, label_new):
    for sec_type in old_nseg:
        plt.scatter(old_distances[sec_type], old_nseg[sec_type], c='r', label=label_old)
        plt.scatter(new_distances[sec_type], new_nseg[sec_type], c='c', label=label_new)
    plt.legend(loc='best')
    plt.xlabel('Distance from Soma (um)')
    plt.ylabel('Number of segments per section')
    plt.title('Changing Spatial Resolution')
    plt.show()
    plt.close()
    print '%s nseg apical' %(label_old)
    print old_nseg['apical']
    print '%s nseg apical' %(label_new)
    print new_nseg['apical']


def run_cable_test(cell):
    plot_cable_param_distribution(cell, 'cm', export='old_cm.hdf5', param_label='cm', show=False, overwrite=True, scale_factor=1)
    modify_mech_param(cell, 'soma', 'cable', 'cm', value=2.)
    init_mechanisms(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'cm', export='new_cm.hdf5', param_label='cm', show=False, overwrite=True, scale_factor=1)
    plot_mech_param_from_file('cm', None, ['old_cm.hdf5', 'new_cm.hdf5'], ['old', 'new'],
                              param_label='cm')

    plot_cable_param_distribution(cell, 'Ra', export='old_Ra.hdf5', param_label='Ra', show=False, overwrite=True, scale_factor=1)
    init_mechanisms(cell, reset_cable=True, from_file=True)
    plot_cable_param_distribution(cell, 'Ra', export='reinit_Ra.hdf5', param_label='Ra', show=False, overwrite=True, scale_factor=1)
    modify_mech_param(cell, 'soma', 'cable', 'Ra', value=200.)
    init_mechanisms(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'Ra', export='modified_Ra.hdf5', param_label='Ra', show=False, overwrite=True, scale_factor=1)
    plot_mech_param_from_file('Ra', None, ['old_Ra.hdf5', 'reinit_Ra.hdf5', 'modified_Ra.hdf5'],
                              ['old', 'reinit', 'modified'], param_label='Ra')

    init_mechanisms(cell, reset_cable=True, from_file=True)
    old_nseg, old_distances = count_nseg(cell)
    modify_mech_param(cell, 'soma', 'cable', 'spatial_res', value=2.)
    init_mechanisms(cell, reset_cable=True)
    new_nseg, new_distances = count_nseg(cell)
    compare_nseg(old_nseg, old_distances, new_nseg, new_distances, 'old', 'new')

    init_mechanisms(cell, reset_cable=True, from_file=True)
    plot_cable_param_distribution(cell, 'cm', export='cm1.hdf5', param_label='cm1', show=False, overwrite=True, scale_factor=1)
    modify_mech_param(cell, 'apical', 'cable', 'cm', value=2.)
    init_mechanisms(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'cm', export='cm2.hdf5', param_label='cm2', show=False, overwrite=True, scale_factor=1)
    nseg2, distances2 = count_nseg(cell)
    modify_mech_param(cell, 'apical', 'cable', 'spatial_res', value=3.)
    plot_cable_param_distribution(cell, 'cm', export='cm3.hdf5', param_label='cm3', show=False, overwrite=True, scale_factor=1)
    nseg3, distances3 = count_nseg(cell)
    modify_mech_param(cell, 'apical', 'cable', 'cm', value=10.)
    plot_cable_param_distribution(cell, 'cm', export='cm4.hdf5', param_label='cm4', show=False, overwrite=True, scale_factor=1)
    nseg4, distances4 = count_nseg(cell)
    plot_mech_param_from_file('cm', None, ['cm1.hdf5', 'cm2.hdf5', 'cm3.hdf5', 'cm4.hdf5'], ['orig', 'post step 1',
                                                                                             'post step 2', 'post step 3'],
                              param_label='cm')
    compare_nseg(nseg2, distances2, nseg3, distances3, 'post step 2', 'post step 3')
    compare_nseg(nseg3, distances3, nseg4, distances4, 'post step 3', 'post step 4')
    #Try changing cm by a significant amount in apical branches only, and then see if this affects nseg. Then change the spatial
    #res parameter -- this should be a multiplier on the current nseg





