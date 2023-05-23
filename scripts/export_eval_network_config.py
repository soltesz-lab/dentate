#!/usr/bin/env python
"""
Dentate Gyrus model script for evaluation of synaptic parameters
"""

import os, sys, logging, datetime, gc, pprint
from functools import partial
import click
import numpy as np
import h5py
from mpi4py import MPI
from neuroh5.io import scatter_read_cell_attribute_selection, read_cell_attribute_info
from collections import defaultdict, namedtuple
import dentate
from dentate import network, utils, optimization
from dentate.env import Env
from dentate.utils import read_from_yaml, write_to_yaml, list_find, viewitems, get_module_logger, config_logging
from dentate.synapses import (SynParam, syn_param_from_dict)
from dentate.optimization import (optimization_params)

ParamSpec = namedtuple("ParamSpec", ['param_names',  'param_tuples', ])

logger = get_module_logger(__name__)

def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--config-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), \
              help='path to evaluation configuration file')
@click.option("--params-id", type=int, help='index of parameter set')
@click.option("--output-file-name", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--verbose", '-v', is_flag=True)
def main(config_path, params_id, output_file_name, verbose):

    config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    eval_config = read_from_yaml(config_path, include_loader=utils.IncludeLoader)
    network_param_spec_src = eval_config['param_spec']
    network_param_values = eval_config['param_values']
    target_populations = eval_config['target_populations']

    network_param_spec = make_param_spec(target_populations, network_param_spec_src)

    def from_param_list(x):
        result = []
        for i, (param_name, param_tuple) in enumerate(zip(network_param_spec.param_names, network_param_spec.param_tuples)):
            param_range = param_tuple.param_range
#            assert((x[i] >= param_range[0]) and (x[i] <= param_range[1]))
            result.append((param_tuple, x[i]))
        return result

    params_id_list = []
    if params_id is None:
        params_id_list = list(network_param_values.keys())
    else:
        params_id_list = [params_id]

    param_output_dict = dict()
    for this_params_id in params_id_list:
        x = network_param_values[this_params_id]
        param_tuple_values = from_param_list(x)
        this_param_list = []
        for param_tuple, param_value in param_tuple_values:
            this_param_list.append((param_tuple.population, 
                                    param_tuple.source,
                                    param_tuple.sec_type,
                                    param_tuple.syn_name,
                                    param_tuple.param_path,
                                    param_value))
        param_output_dict[this_params_id] = this_param_list
                
    pprint.pprint(param_output_dict)
    if output_file_name is not None:
        write_to_yaml(output_file_name, param_output_dict)
    
    

def make_param_spec(pop_names, param_dict):

    """Constructs a flat list representation of synaptic parameters."""
    
    param_names = []
    param_initial_dict = {}
    param_tuples = []

    for pop_name in pop_names:
        param_specs = param_dict[pop_name]
        keyfun = lambda kv: str(kv[0])
        for source, source_dict in sorted(viewitems(param_specs), key=keyfun):
            for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                    for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                        if isinstance(param_rst, dict):
                            for const_name, const_range in sorted(viewitems(param_rst)):
                                param_path = (param_fst, const_name)
                                param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_path, const_range))
                                param_key = '%s.%s.%s.%s.%s.%s' % (pop_name, str(source), sec_type, syn_name, param_fst, const_name)
                                param_names.append(param_key)
                        else:
                            param_name = param_fst
                            param_range = param_rst
                            param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_name, param_range))
                            param_key = '%s.%s.%s.%s.%s' % (pop_name, source, sec_type, syn_name, param_name)
                            param_names.append(param_key)
                                
    return ParamSpec(param_names=param_names, param_tuples=param_tuples)



if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
