from collections import defaultdict

import click
import dentate
from dentate import network_clamp
from dentate.env import Env
from dentate.neuron_utils import *
from dentate.utils import *


@click.command()
@click.option("--config-file", '-c', required=True, type=str)
@click.option("--population", '-p', required=True, type=str, default='GC')
@click.option("--gid", '-g', required=True, type=int, default=0)
@click.option("--template-paths", type=str, required=True)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
def main(config_file, population, gid, template_paths, dataset_prefix, config_prefix):
    """
    Runs network clamp simulation for the specified cell gid.

    :param config_file: str; model configuration file name
    :param population: str
    :param gid: int
    :param tstop: float
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    :param verbose: bool
    """

    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    params = dict(locals())
    env = Env(verbose=True, **params)
    configure_hoc_env(env)

    cell = network_clamp.init_cell(env, population, gid)

    syn_attrs = env.synapse_attributes

    swc_type_apical = env.SWC_Types['apical']
    swc_type_basal = env.SWC_Types['basal']
    swc_type_soma = env.SWC_Types['soma']
    swc_type_axon = env.SWC_Types['axon']
    swc_type_ais = env.SWC_Types['ais']
    swc_type_hill = env.SWC_Types['hillock']

    syn_type_excitatory = env.Synapse_Types['excitatory']
    syn_type_inhibitory = env.Synapse_Types['inhibitory']

    layer_IML = env.layers['IML']
    layer_MML = env.layers['MML']
    layer_OML = env.layers['OML']

    for it in range(10):
        start_time = time.time()
        
        exc_dend_synapses_ML = \
        syn_attrs.filter_synapses(gid,
                                    syn_types=[syn_type_excitatory],
                                    layers=[layer_IML, layer_MML, layer_OML],
                                    swc_types=[swc_type_apical],
                                    cache=True)
        inh_dend_synapses_ML = \
        syn_attrs.filter_synapses(gid,
                                    syn_types=[syn_type_inhibitory],
                                    layers=[layer_IML, layer_MML, layer_OML],
                                    swc_types=[swc_type_apical],
                                    cache=True)
        exc_dend_synapses_PP = \
        syn_attrs.filter_synapses(gid,
                                    syn_types=[syn_type_excitatory],
                                    swc_types=[swc_type_apical],
                                    sources=['MPP', 'LPP'],
                                    cache=True)

        end_time = time.time()
        print("time to filter %d excitatory and %d inhibitory synapses (iteration %i): %f s" % \
              (len(exc_dend_synapses_ML), len(inh_dend_synapses_ML), it, end_time-start_time))
    
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
