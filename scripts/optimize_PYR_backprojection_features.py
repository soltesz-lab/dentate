from nested.optimize_utils import *
import sys, os, time, random, click
import numpy as np
from pprint import pprint
import yaml
from dentate.utils import *
from dentate.stimulus import generate_spatial_offsets, generate_spatial_ratemap, generate_mesh, generate_expected_width
from dentate.InputCell import *


config_logging(True)
script_name = os.path.basename(__file__)
logger      = get_script_logger(script_name)

context = Context()

def _build_cells(N, mod_jitter, ctype, start_gid=1):

    assert(ctype == 'place')
    cells = {}
    nfields          = acquire_fields_per_cell(N, context.field_probabilities.values(), context.field_random)

    pseudo_positions = context.field_random.uniform(0., 1., size=(N,))
    expected_field_width = np.interp(pseudo_positions, context.positions, context.mean_expected_width)
    total_fields = np.sum(nfields)

    gid = start_gid
    gid_to_module     = {int(mod): [] for mod in context.modules}
    fields_per_module = {int(mod): 0 for mod in context.modules}
    
    for i in xrange(N):
        pseudo_module = np.where(pseudo_positions[i] < context.offsets)[0][0] - 1
        gid_to_module[pseudo_module].append((gid, nfields[i]))
        fields_per_module[pseudo_module] += nfields[i]
        expected_cell_width = expected_field_width[i]
        kwargs = {'field width':  expected_cell_width, 'jitter': mod_jitter[pseudo_module]}
        cells[gid] = instantiate_place_cell(context, gid, pseudo_module, nfields[i], **kwargs).return_attr_dict()
        gid += 1


    for mod in sorted(gid_to_module.keys()):
        gids_in_mod  = gid_to_module[mod]
        this_scale_factor = context.scale_factors[mod]
        xy_offsets, _, _, _ = generate_spatial_offsets(fields_per_module[mod], arena_dimension=context.arena_dimension, scale_factor=this_scale_factor)
        xy_insertion_order = context.field_random.permutation(np.arange(len(xy_offsets)))
        xy_offsets = xy_offsets[xy_insertion_order]

        curr_pos = 0
        for (gid, gid_nfields) in gid_to_module[mod]:
            cell = cells[gid]
            cell['X Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+gid_nfields,0], dtype='float32')
            cell['Y Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+gid_nfields,1], dtype='float32')
            curr_pos += gid_nfields
    return cells, gid + 1

def init_context():

    local_random = random.Random()
    local_random.seed(context.local_seed)

    feature_type_random = np.random.RandomState(context.local_seed)
    field_random = np.random.RandomState(context.local_seed)

    if 'input_params_file_path' in context():
        input_params = read_from_yaml(context.input_params_file_path, include_loader=IncludeLoader)
    else:
        input_params = read_from_yaml('../config/Input_Features.yaml', include_loader=IncludeLoader)
    nmodules = input_params['number modules']
    field_width_x1 = input_params['field width params']['x1']
    field_width_x2 = input_params['field width params']['x2']
    arena_dimension = input_params['arena dimension']
    resolution = input_params['resolution']
    field_probabilities = input_params['field probs']['PYR']
        
    ctype = 1 # place
    modules            = np.arange(nmodules, dtype='float32')
    field_width_params = [field_width_x1, field_width_x2]
    field_width        = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
    max_field_width    = field_width(1.)
    feature_ctypes     = {'grid': 0, 'place': 1}
    offsets            = np.divide(modules, np.max(modules))
    module_widths      = field_width(offsets)
    scale_factors      = (module_widths / arena_dimension / 2.) + 1.

    mean_expected_width, positions = generate_expected_width(field_width_params, module_widths, offsets, positions=None)


    mesh   = generate_mesh(scale_factor=1., arena_dimension=arena_dimension, resolution=resolution)
    nx, ny = mesh[0].shape[0], mesh[0].shape[1]
    place_cells = {}
    place_gid_start = 1
    context.update(locals())

@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-params-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(exists=True, file_okay=True, dir_okay=True), default=None)
@click.option("--export", is_flag=True, default=False)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--run-tests", is_flag=True, default=False, required=False)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(config_file_path, input_params_file_path, output_dir, export, export_file_path, label, run_tests, verbose):
    context.update(locals())
    disp = verbose > 0
    if disp:
        print('... config interactive underway..')
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp)
    if disp:
        print('... config interactive complete...')
 
    if run_tests:
        tests(plot=True)

def tests(plot=False):
    report_cost(context)

    place_cells = context.place_cells
    kwargs = {'ctype': 'place', 'module': 0, \
              'target': context.fraction_active_target, 'nbins': 40}
    plot_group(place_cells, plot=plot, **kwargs)


def plot_group(cells, plot=False, **kwargs):
    from plot_DG_PP_features import plot_rate_maps_single_module, plot_xy_offsets_single_module, \
                                    plot_fraction_active_single_module, plot_rate_histogram_single_module
    plot_rate_maps_single_module(cells, plot=True, save=False, **kwargs)
    plot_xy_offsets_single_module(cells,plot=True,save=False, **kwargs)
    plot_fraction_active_single_module(cells,(context.nx, context.ny), plot=True,save=False, **kwargs)
    plot_rate_histogram_single_module(cells, plot=plot, save=False, **kwargs)

def report_cost(context):
    x0 = context.x0_array
    features = calculate_features(x0)
    _, objectives = get_objectives(features)

    print(x0)
    print('Place population fraction active: %f' % features['fraction active'])

    for feature in features.keys():
        if feature in context.feature_names:
            print('Feature: %s has value %f' % (feature, features[feature]))
    for objective in objectives.keys():
        print('Objective: %s has cost %f' % (objective, objectives[objective]))

def config_worker():
    if 'num_place' not in context():
        raise Exception('Place and grid cell counts must be defined prior to optimization')
    init_context()
    
def get_cell_types_from_context(context):
    cells = context.place_cells
    if cells is None:
        raise Exception('Could not find proper cells of type %s' % context.cell_type)
    return cells

def calculate_features(parameters, export=False):
    update_source_contexts(parameters, context)
    cells    = context.place_cells
    features = {}

    fraction_active = _fraction_active(cells) 
    c_variation     = _coefficient_of_variation(cells)

    features['coefficient of variation'] = c_variation
    features['fraction active population'] = fraction_active
    features['fraction active'] = np.mean(fraction_active.values())
    return features

def get_objectives(features):
    feature_names = context.feature_names
    for feature in feature_names:
        if feature not in features:
            raise Exception('Feature %s could not be found' % feature)

    objectives = {}

    if features['coefficient of variation'] <= context.target_val['variation error']:
        objectives['variation error'] = 0.0
    else:
        objectives['variation error'] = ((features['coefficient of variation'] - context.target_val['variation error']) / context.target_range['variation error']) ** 2


    fraction_active  = features['fraction active population']
    diff_frac_active = {(i,j): np.abs(fraction_active[(i,j)] - context.fraction_active_target) \
                        for (i,j) in fraction_active.keys()}
    fraction_active_errors = np.asarray([diff_frac_active[(i,j)] for (i,j) in diff_frac_active.keys()])
    fraction_active_mean_error = np.mean(fraction_active_errors)
    fraction_active_var_error  = np.var(fraction_active_errors)
    objectives['fraction active mean error'] = ((fraction_active_mean_error - context.target_val['fraction active mean error']) / context.target_range['fraction active mean error']) ** 2
    objectives['fraction active var error'] = ((fraction_active_var_error - context.target_val['fraction active var error']) / context.target_range['fraction active var error']) ** 2

    return features, objectives

def update(x, context):
    context.local_random = random.Random()
    context.local_random.seed(context.local_seed)
    context.field_random = np.random.RandomState(context.local_seed)
   
    mod_jitter = x[0:context.nmodules]
    context.place_cells, _ = _build_cells(context.num_place, mod_jitter, 'place', start_gid=context.place_gid_start)
    _calculate_rate_maps(context.place_cells, context)

def _merge_cells():
    z = context.grid_cells.copy()
    return z.update(context.place_cells.copy())

def _calculate_field_distribution(pi, pr):
    p1 = (1. - pi) / (1. + (7./4.) * pr)
    p2 = p1 * pr
    p3 = 0.5 * p2
    p4 = 0.5 * p3
    probabilities = np.array([pi, p1, p2, p3, p4], dtype='float32')
    assert( np.abs(np.sum(probabilities) - 1.) < 1.e-5)
    return probabilities 

def _fraction_active(rates):
    from dentate.stimulus import fraction_active
    return fraction_active(rates, context.active_threshold)

def _coefficient_of_variation(cells):
    from dentate.stimulus import coefficient_of_variation

    return coefficient_of_variation(cells)

def _peak_to_trough(cells):
    from dentate.stimulus import peak_to_trough

    return peak_to_trough(cells)
    
def _calculate_rate_maps(cells, context):
    xp, yp       = context.mesh
    ratemap_kwargs = dict()
    ratemap_kwargs['a'] = context.a
    ratemap_kwargs['b'] = context.b
    ratemap_kwargs['c'] = context.c
    
    for gid in cells:
        cell = cells[gid]
        if cell['Num Fields'][0] > 0:
            spacing     = cell['Field Width']
            orientation = [0.0 for _ in range(len(spacing))]
            rate_map = generate_spatial_ratemap(context.ctype, cell, None, xp, yp, 0.0, \
                                                context.place_peak_rate, None, **ratemap_kwargs)
            cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')
        else:
            cell['Rate Map'] = np.zeros( (cell['Nx'][0] * cell['Ny'][0],) ).astype('float32')

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv)+1):])
