from __future__ import absolute_import

import parsl
from dentate import io_utils
from dentate import synapses
from dentate.env import Env
from parsl.app.app import bash_app
from parsl.app.app import python_app


@python_app
def make_h5types(inputs=[],outputs=[],gap_junctions=False):
    config_path=inputs[0]
    output_path=outputs[0]
    env = Env(config_file=config_path)
    io_utils.make_h5types(env, output_path, gap_junctions=gap_junctions)


@bash_app
def neurotrees_import_swc(population, inputs=[], outputs=[]):
    swc_path=swc_inputs[0]
    output_path=outputs[0]
    return 'neurotrees_import -r %s %s %s' % (swc_path, population, output_path)


@bash_app
def neurotrees_copy_fill_swc(population, gid, inputs=[], outputs=[]):
    output_path=outputs[0]
    return 'neurotrees_copy --fill %s %s %d' % (output_path, population, gid)


@python_app
def make_forest_from_singleton(population,config_inputs=[],swc_inputs=[],outputs=[]):
    config_path=config_inputs[0]
    output_path=outputs[0]
    env = Env(config_file=config_path)
    make_h5types(config_inputs, outputs)
    neurotrees_import_swc(population, swc_inputs, outputs)
    neurotrees_copy_fill_swc(population, 0, outputs)


@python_app
def distribute_syns(population,template_path,distribution,dry_run=False,config_inputs=[],forest_inputs=[],outputs=[]):
    config_path=config_inputs[0]
    forest_path=forest_inputs
    output_path=outputs[0]
    env = Env(config_file=config_path,template_paths=template_path)
    synapses.distribute_synapse_locs(env, output_path, forest_path, population, distribution, dry_run=dry_run)
