
from dentate.neuron_utils import *
from neuroh5.h5py_io_utils import *
from dentate.env import Env
from dentate.cells import *
from dentate.synapses import *


def load_cell(env, pop_name, gid, mech_file, correct_for_spines):
    """

    :param gid: int
    :param pop_name: str
    :param config_file: str; model configuration file name
    :param mech_file: str; cell mechanism config file name
    :param correct_for_spines: bool

    env = Env(config_file, template_paths, dataset_prefix, config_prefix)
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    
    """
    configure_hoc_env(env)

    print("before get_biophys_cell")
    cell = get_biophys_cell(env, gid, pop_name)
    print("get_biophys_cell: cell = %s" % str(cell))
    mech_file_path = env.configPrefix + '/' + mech_file

    print("mech_file_path = %s" % mech_file_path)
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path,
                    correct_cm=correct_for_spines, correct_g_pas=correct_for_spines, env=env)
    init_syn_mech_attrs(cell, env)
    config_syns_from_mech_attrs(gid, env, pop_name, insert=True)
    return cell

#    report_topology(cell, env)
