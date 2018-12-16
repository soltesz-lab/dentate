from nested.optimize_utils import *
import sys, os, time, random, click
import numpy as np
from pprint import pprint

import dentate.utils as utils
from dentate.utils import list_find, get_script_logger
from dentate.stimulus import generate_spatial_offsets, generate_spatial_ratemap, generate_mesh
from dentate.InputCell import *


utils.config_logging(True)
script_name = 'optimize_DG_PP_features.py'
logger      = utils.get_script_logger(script_name)

context = Context()

def _build_cells(N, ctype, module, start_gid=1):

    cells = {}
    if ctype == 'place':
        nfields      = acquire_fields_per_cell(N, context.field_probabilities, context.field_random)
        total_fields = np.sum(nfields)
    elif ctype == 'grid':
        nfields      = np.ones((N,), dtype='uint8')
        total_fields = N

    gid = start_gid
    for i in xrange(N):
        if ctype == 'grid':
            cells[gid] = instantiate_grid_cell(context, gid, module, nfields[i]).return_attr_dict()
        elif ctype == 'place':
            cells[gid] = instantiate_place_cell(context, gid, module, nfields[i]).return_attr_dict()
        gid += 1

    scale_factor = context.scale_factor
    xy_offsets,_, _, _ = generate_spatial_offsets(total_fields, arena_dimension=context.arena_dimension, scale_factor=scale_factor)
    xy_insertion_order = context.field_random.permutation(np.arange(len(xy_offsets)))
    xy_offsets = xy_offsets[xy_insertion_order]


    curr_pos = 0
    for (i,gid) in enumerate(xrange(start_gid, gid)):
        cell = cells[gid]
        nf   = nfields[i]
        if cell['Cell Type'][0] == 0:
            cell['X Offset'] = np.array([xy_offsets[curr_pos,0]], dtype='float32')
            cell['Y Offset'] = np.array([xy_offsets[curr_pos,1]], dtype='float32')
        elif cell['Cell Type'][0] == 1:
            cell['X Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+nf,0], dtype='float32')
            cell['Y Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+nf,1], dtype='float32')
        curr_pos += nf
    return cells, gid + 1

def init_context():

    local_random = random.Random()
    local_random.seed(context.local_seed)

    feature_type_random = np.random.RandomState(context.local_seed)
    field_random = np.random.RandomState(context.local_seed)
    field_probabilities = None
    
    nmodules           = 10
    modules            = np.arange(nmodules)
    grid_orientation   = [local_random.uniform(0, np.pi/3.) for i in xrange(nmodules)]
    field_width_params = [35.0, 0.32]
    field_width        = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
    max_field_width    = field_width(1.)
    feature_ctypes     = {'grid': 0, 'place': 1}
    arena_dimension    = 100.
    resolution         = 5.
    module_width       = field_width( float(context.module) / np.max(modules))
    scale_factor       = (module_width / 100.) + 1.

    mesh   = generate_mesh(scale_factor=1., arena_dimension=arena_dimension, resolution=resolution)
    nx, ny = mesh[0].shape[0], mesh[0].shape[1]
    grid_cells, place_cells = {}, {}
    place_gid_start = None
    context.update(locals())
    context.grid_cells, context.place_gid_start  = _build_cells(context.num_grid, 'grid', context.module)
    _calculate_rate_maps(context.grid_cells, context)

@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), 
               default="../config/optimize_DG_PP_config_3.yaml")
@click.option("--output-dir", type=click.Path(exists=True, file_okay=True, dir_okay=True), default=None)
@click.option("--export", is_flag=True, default=False)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--run-tests", is_flag=True, default=False, required=False)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(config_file_path, output_dir, export, export_file_path, label, run_tests, verbose):
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
    grid_cells  = context.grid_cells
    kwargs = {'ctype': 'place', 'module': context.module, \
              'target': context.fraction_active_target, 'nbins': 40}

    plot_group(place_cells, plot=plot, **kwargs)
    kwargs['ctype'] = 'grid'
    #plot_group(grid_cells, plot=plot, **kwargs)

    kwargs['ctype'] = 'both'
    both_cells = grid_cells.copy()
    both_cells.update(place_cells)
    plot_group(both_cells, plot=plot, **kwargs)

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

    grid_fa = np.mean(_fraction_active(context.grid_cells).values())

    print('probability inactive: %f' % x0[0])
    print('pr: %0.4f' % x0[1])
    print(_calculate_field_distribution(x0[0], x0[1]))
    print('Module: %d' % context.module)
    print('Place population fraction active: %f' % features['fraction active'])
    print('Grid population fraction active: %f' % grid_fa)
    #print('Total population fraction active: %f' % pop_fa)

    for feature in features.keys():
        if feature in context.feature_names:
            print('Feature: %s has value %f' % (feature, features[feature]))
    for objective in objectives.keys():
        print('Objective: %s has cost %f' % (objective, objectives[objective]))

def config_worker():

    if 'module' not in context():
        raise Exception('Optimization cannot proceed unless it has been given a module')
    if 'num_grid' not in context() or 'num_place' not in context():
        raise Exception('Place and grid cell counts must be defined prior to optimization')
    init_context()
    
def get_cell_types_from_context(context):
    cells = None
    if context.cell_type == 'grid':
        cells = context.grid_cells
    elif context.cell_type == 'place':
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
   
    p_inactive     = x[0] 
    p_r            = x[1]
    context.field_probabilities = _calculate_field_distribution(p_inactive, p_r)
    context.place_cells, _ = _build_cells(context.num_place, 'place', context.module, start_gid=context.place_gid_start)
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
            ctype = cell['Cell Type'][0]
            if ctype == 0: # Grid
                orientation = cell['Grid Orientation']
                spacing     = cell['Grid Spacing']
            elif ctype == 1: # Place
                spacing     = cell['Field Width']
                orientation = [0.0 for _ in range(len(spacing))]
 

            rate_map = generate_spatial_ratemap(ctype, cell, None, xp, yp, context.grid_peak_rate, \
                                                context.place_peak_rate, None, **ratemap_kwargs)
            cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')
        else:
            cell['Rate Map'] = np.zeros( (cell['Nx'][0] * cell['Ny'][0],) ).astype('float32')

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv)+1):])
