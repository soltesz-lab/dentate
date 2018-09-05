import sys, os, time, random, click, logging
import numpy as np
from pprint import pprint

import dentate
import dentate.utils as utils
from dentate.utils import list_find, get_script_logger
from dentate.stimulus import generate_spatial_offsets, generate_spatial_ratemap

from nested.optimize_utils import *
from optimize_cells_utils import *


utils.config_logging(True)
script_name = 'optimize_DG_PP_features_v2.py'
logger      = utils.get_script_logger(script_name)

context = Context()

def _instantiate_place_cell(gid, module, nfields):
    cell = {}
    if nfields == 0:
        nfields_real = 0
        nfields = 1
    else:
        nfields_real = nfields

    cell_field_width = []
    for n in xrange(nfields):
        this_width = context.field_width(float(module)/float(np.max(context.modules)))
        delta_spacing = context.local_random.gauss(0., 50. * (module+1)/float(np.max(context.modules)))
        cell_field_width.append(this_width + delta_spacing)
    
    cell['gid']         = np.array([gid], dtype='int32')
    cell['Num Fields']  = np.array([nfields_real], dtype='uint8')
    cell['Cell Type']   = np.array([context.feature_ctypes['place']], dtype='uint8')
    cell['Module']      = np.array([module], dtype='uint8')
    cell['Field Width'] = np.asarray(cell_field_width, dtype='float32')
    cell['Nx']           = np.array([context.nx], dtype='int32')
    cell['Ny']           = np.array([context.ny], dtype='int32')
    
    return cell 

def _instantiate_grid_cell(gid, module, nfields):
    cell   = {}

    orientation = context.grid_orientation[module]
    spacing     = context.field_width(float(module)/float(np.max(context.modules)))

    delta_spacing     = context.local_random.gauss(0., 50. * (module+1) / float(np.max(context.modules)))
    delta_orientation = context.local_random.gauss(0., np.deg2rad(10. * (module+1.) / float(np.max(context.modules))))

    cell['gid']          = np.array([gid], dtype='int32')
    cell['Num Fields']   = np.array([nfields], dtype='uint8')
    cell['Cell Type']    = np.array([context.feature_ctypes['grid']], dtype='uint8')
    cell['Module']       = np.array([module], dtype='uint8')
    cell['Grid Spacing'] = np.array([spacing + delta_spacing], dtype='float32')
    cell['Grid Orientation']  = np.array([orientation + delta_orientation], dtype='float32')
    cell['Nx']           = np.array([context.nx], dtype='int32')
    cell['Ny']           = np.array([context.ny], dtype='int32')

    return cell 

def acquire_fields_per_cell(ncells, field_probabilities, generator):
    field_probabilities = np.asarray(field_probabilities, dtype='float32')
    field_set = [i for i in range(field_probabilities.shape[0])]
    return generator.choice(field_set, p=field_probabilities, size=(ncells,))
    
def _build_cells(N, ctype, module):
    grid_cells, place_cells = {}, {}
    nfields = acquire_fields_per_cell(N, context.field_probabilities, context.field_random)
    total_fields = np.sum(nfields)
    gid = 1
    for i in xrange(N):
        if ctype == 'grid':
            grid_cells[gid]= _instantiate_grid_cell(gid, module, nfields[i])
        elif ctype == 'place':
            place_cells[gid] = _instantiate_place_cell(gid, module, nfields[i])
        gid += 1
    _, xy_offsets, _, _ = generate_spatial_offsets(total_fields, arena_dimension=context.arena_dimension, scale_factor=1.0)

    if ctype == 'grid':
        cells = grid_cells
    elif ctype == 'place':
        cells = place_cells
    curr_pos = 0
    for (i, gid) in enumerate(cells):
        cell    = cells[gid]
        nf = nfields[i]
        if cell['Cell Type'] == 0 and nf > 0:  
            cell['X Offset'] = np.array([xy_offsets[curr_pos,0]], dtype='float32')
            cell['Y Offset'] = np.array([xy_offsets[curr_pos,1]], dtype='float32')
        elif cell['Cell Type'] == 1 and nf > 0:
            cell['X Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+nf,0], dtype='float32')
            cell['Y Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+nf,1], dtype='float32')
        curr_pos += nf
    return grid_cells, place_cells

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

    mesh   = _generate_mesh()
    nx, ny = mesh[0].shape[0], mesh[0].shape[1]
    grid_cells  = {}
    place_cells = {}
    context.update(locals())


def _generate_mesh(scale_factor=1.0, arena_dimension=100., resolution=5.):
    arena_x_bounds = [-arena_dimension * scale_factor / 2., arena_dimension * scale_factor / 2.]
    arena_y_bounds = [-arena_dimension * scale_factor / 2., arena_dimension * scale_factor / 2.]
    arena_x        = np.arange(arena_x_bounds[0], arena_x_bounds[1], resolution)
    arena_y        = np.arange(arena_y_bounds[0], arena_y_bounds[1], resolution)
    return np.meshgrid(arena_x, arena_y, indexing='ij')

@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), 
               default="../config/optimize_DG_PP_config_2.yaml")
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
        tests(plot=False)

def tests(plot=False):
    from plot_DG_PP_features import plot_rate_maps, plot_xy_offsets, \
                                    plot_fraction_active_map, plot_rate_histogram
    cells = get_cell_types_from_context(context)
    kwargs = {'ctype': context.cell_type, 'module': context.module, \
              'target': context.fraction_active_target, 'nbins': 40}

    report_cost(context)
    plot_rate_maps(cells, plot=False, save=True, **kwargs)
    plot_xy_offsets(cells,plot=False,save=True, **kwargs)
    plot_fraction_active_map(cells,'_fraction_active', plot=False,save=True, **kwargs)
    plot_rate_histogram(cells, plot=plot, save=True, **kwargs)

def report_cost(context):
    x0 = context.x0_array
    features = calculate_features(x0)
    _, objectives = get_objectives(features)

    print('probability inactive: %f' % x0[0])
    print('Module: %d' % context.module)
    print('Population fraction active: %f' % features['fraction active'])
    for objective in objectives.keys():
        print('Objective: %s has cost %f' % (objective, objectives[objective]))

def config_controller(export_file_path, output_dir, **kwargs):
    context.update(locals())
    context.update(kwargs)
    init_context()


def config_worker(update_context_funs, param_names, default_params, feature_names, objective_names, target_val, target_range, temp_output_path, export_file_path, output_dir, disp, **kwargs):

    context.update(locals())
    context.update(kwargs)
    if 'module' not in context():
        raise Exception('Optimization cannot proceed unless it has been given a module')
    if 'cell_type' not in context():
        raise Exception('Optimization cannot proceed unless a cell type has been specified')
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
    cells    = get_cell_types_from_context(context)
    features = {}

    minmax_eval     = _peak_to_trough(cells)
    fraction_active = _fraction_active(cells) 

    features['minmax sum eval'] = minmax_eval[0]
    features['minmax var eval'] = minmax_eval[1]
    features['fraction active population'] = fraction_active
    features['fraction active'] = np.mean(fraction_active.values())
    return features

def get_objectives(features):
    feature_names = context.feature_names
    for feature in feature_names:
        if feature not in features:
            raise Exception('Feature %s could not be found' % feature)

    objectives = {}
    objectives['minmax sum error'] = ((features['minmax sum eval'] - context.target_val['minmax sum error']) / context.target_range['minmax sum error']) ** 2
    objectives['minmax var error'] = ((features['minmax var eval'] - context.target_val['minmax var error']) / context.target_range['minmax var error']) ** 2
    fraction_active  = features['fraction active population']
    diff_frac_active = {(i,j): np.abs(fraction_active[(i,j)] - context.fraction_active_target) \
                        for (i,j) in fraction_active.keys()}
    fraction_active_errors = np.asarray([diff_frac_active[(i,j)] for (i,j) in diff_frac_active.keys()])
    fraction_active_mean_error = np.mean(fraction_active_errors)
    fraction_active_var_error  = np.var(fraction_active_errors)
    objectives['fraction active mean error'] = ((fraction_active_mean_error - context.target_val['fraction active mean error']) / context.target_range['fraction active mean error']) ** 2
    objectives['fraction active var error'] = ((fraction_active_var_error - context.target_val['fraction active var error']) / context.target_range['fraction active var error']) ** 2

    return features, objectives

def _peak_to_trough(cells):
    rate_maps = []
    for gid in cells:
        cell     = cells[gid]
        nx, ny   = cell['Nx'][0], cell['Ny'][0]
        rate_map = cell['Rate Map'].reshape(nx, ny)
        rate_maps.append(rate_map)
    rate_maps  = np.asarray(rate_maps, dtype='float32')
    summed_map = np.sum(rate_maps, axis=0)
    var_map    = np.var(rate_maps, axis=0)
    minmax_eval = np.divide(float(np.max(summed_map)), float(np.min(summed_map)))
    var_eval    = np.divide(float(np.max(var_map)), float(np.min(var_map)))
    return minmax_eval, var_eval 

def _fraction_active(cells):
    rate_maps = []
    for gid in cells:
        cell     = cells[gid]
        nx, ny   = cell['Nx'][0], cell['Ny'][0]
        rate_map = cell['Rate Map'].reshape(nx, ny)
        rate_maps.append(rate_map)
    rate_maps = np.asarray(rate_maps, dtype='float32')
    nxx, nyy  = np.meshgrid(np.arange(nx), np.arange(ny))
    coords    = zip(nxx.reshape(-1,), nyy.reshape(-1,))
    
    factive = lambda px, py: _calculate_fraction_active(rate_maps[:,px,py])
    return {(px,py): factive(px, py) for (px, py) in coords}

def _calculate_fraction_active(rates):
    N = len(rates)
    num_active = len(np.where(rates > context.active_threshold)[0])
    fraction_active = np.divide(float(num_active), float(N))
    return fraction_active    


def update(x, context):
    p_inactive = x[0]
    cells      = get_cell_types_from_context(context)
    context.local_random = random.Random()
    context.local_random.seed(context.local_seed)
    context.field_random = np.random.RandomState(context.local_seed)
    
    if context.cell_type == 'grid':
        context.field_probabilities = np.asarray([p_inactive, 1. - p_inactive], dtype='float32')

    elif context.cell_type == 'place':
        remaining = 1 - p_inactive
        chunk = remaining / 13.
        p1, p2, p3, p4 = 6. * chunk, 4. * chunk, 2. * chunk, 1. * chunk
        context.field_probabilities = np.asarray([p_inactive, p1, p2, p3, p4], dtype='float32')
        assert(np.abs(np.sum(context.field_probabilities) - 1.) < 1.0e-5)

    context.grid_cells, context.place_cells = _build_cells(context.num_cells, context.cell_type, context.module)
    scale_factor = context.field_width( float(context.module) / np.max(context.modules) )
    _calculate_rate_maps(scale_factor, context)
    
    
def _calculate_rate_maps(scale_factor, context):
    cells        = get_cell_types_from_context(context)
    xp, yp       = context.mesh
    ratemap_kwargs = dict()
    ratemap_kwargs['a'] = context.a
    ratemap_kwargs['b'] = context.b
    
    for gid in cells:
        cell  = cells[gid]
        if cell['Num Fields'][0] > 0:
            ctype = cell['Cell Type']

            if ctype == 0: # Grid
                orientation = cell['Grid Orientation']
                spacing     = cell['Grid Spacing']
            elif ctype == 1: # Place
                spacing     = cell['Field Width']
                orientation = [0.0 for _ in range(len(spacing))]
 
            xf, yf    = cell['X Offset'], cell['Y Offset']
            xf_scaled = cell['X Offset'] * (1. + (scale_factor / context.arena_dimension))
            yf_scaled = cell['Y Offset'] * (1. + (scale_factor / context.arena_dimension))
            cell['X Offset Scaled'] = np.asarray(xf_scaled, dtype='float32')
            cell['Y Offset Scaled'] = np.asarray(yf_scaled, dtype='float32')
            rate_map = generate_spatial_ratemap(ctype, cell, None, xp, yp, context.grid_peak_rate, \
                                            context.place_peak_rate, None, **ratemap_kwargs)
            cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')
        else:
            cell['Rate Map'] = np.zeros( (cell['Nx'][0] * cell['Ny'][0],) ).astype('float32')


# Deprecated, now found in stimulus.py
def _rate_map(xp, yp, spacing, orientation, x_offset, y_offset, ctype):    
    if ctype == 1:
        rate_map = context.place_peak_rate *  np.exp(-((xp - x_offset) / (spacing / 3. / np.sqrt(2.))) ** 2.) * \
        np.exp(-((yp - y_offset) / (spacing / 3. / np.sqrt(2.))) ** 2.)
    elif ctype == 0:
        theta_k   = [np.deg2rad(-30.), np.deg2rad(30.), np.deg2rad(90.)]
        inner_sum = np.zeros(xp.shape)
        for theta in theta_k:
            inner_sum += np.cos( ((4 * np.pi) / (np.sqrt(3.) * spacing)) * ( np.cos(theta - orientation) * (xp - x_offset) + np.sin(theta - orientation) * (yp - y_offset) ) )
    
        transfer = lambda z: np.exp(0.3 * (z - (-1.5))) - 1.
        rate_map = context.grid_peak_rate * transfer(inner_sum) / transfer(3.)
    else:
        raise Exception('Could not find proper cell type')
    return rate_map    
   

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv)+1):])
