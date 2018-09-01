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
script_name = 'generate_DG_PP_features_v3.py'
logger      = utils.get_script_logger(script_name)

seed = 64
local_random = random.Random()
local_random.seed(seed)
feature_type_random  = np.random.RandomState(seed)
place_field_random   = np.random.RandomState(seed)
n_place_field_probabilities = np.asarray([0.8, 0.15, 0.05])
grid_field_random    = np.random.RandomState(seed)
n_grid_field_probabilities = np.asarray([0.10, 0.90]) 

nmodules           = 10
modules            = np.arange(nmodules)
grid_orientation   = [local_random.uniform(0, np.pi/3.) for i in xrange(nmodules)]
field_width_params = [35.0, 0.32]
field_width        = lambda x : 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
max_field_width    = field_width(1.)


n_mpp_grid      = 5000
n_mpp_place     = 5
n_lpp_place     = 5
n_lpp_grid      = 0
arena_dimension = 100.
resolution      = 5.
feature_population = {'MPP': 0, 'LPP': 1}
feature_ctypes     = {'grid': 0, 'place': 1}

context = Context()

def _instantiate_place_cell(population, gid, module, nxy):
    cell = {}
    field_set = [1,2,3]
    nfields   = place_field_random.choice(field_set, p=n_place_field_probabilities, size=(1,))
    cell_field_width = []
    for n in xrange(nfields[0]):
        cell_field_width.append(field_width(local_random.random()))
    
    cell['gid']         = np.array([gid], dtype='int32')
    cell['Num Fields']  = np.array([nfields], dtype='uint8')
    cell['Population']  = np.array([feature_population[population]], dtype='uint8')
    cell['Cell Type']   = np.array([feature_ctypes['place']], dtype='uint8')
    cell['Module']      = np.array([module], dtype='uint8')
    cell['Field Width'] = np.asarray(cell_field_width, dtype='float32')
    cell['Nx']           = np.array([nxy[0]], dtype='int32')
    cell['Ny']           = np.array([nxy[1]], dtype='int32')
    
    return cell 

def _instantiate_grid_cell(population, gid, module, nxy):
    cell   = {}
    field_set = [0,1]
    nfields   = grid_field_random.choice(field_set, p=n_grid_field_probabilities, size=(1,))
    
    orientation = grid_orientation[module]
    spacing     = field_width(float(module)/float(np.max(modules)))

    delta_spacing     = local_random.gauss(0., 50. * (module+1) / float(np.max(modules)))
    delta_orientation = local_random.gauss(0., np.deg2rad(10. * (module+1.) / float(np.max(modules))))

    cell['gid']          = np.array([gid], dtype='int32')
    cell['Num Fields']   = np.array([nfields], dtype='uint8')
    cell['Population']   = np.array([feature_population[population]], dtype='uint8')
    cell['Cell Type']    = np.array([feature_ctypes['grid']], dtype='uint8')
    cell['Module']       = np.array([module], dtype='uint8')
    cell['Grid Spacing'] = np.array([spacing + delta_spacing], dtype='float32')
    cell['Grid Orientation']  = np.array([orientation + delta_orientation], dtype='float32')
    cell['Nx']           = np.array([nxy[0]], dtype='int32')
    cell['Ny']           = np.array([nxy[1]], dtype='int32')

    return cell 

def _build_cells(population_context, module, nxy):
    grid_cells, place_cells = {}, {}
    gid = 1
    for population in population_context:
        current_population = population_context[population]
        for ctype in current_population:
            start_gid  = gid
            ncells     = current_population[ctype][0]
            if ncells == 0:
                population_context[population][ctype] += [None, None]
            for i in xrange(ncells):
                if ctype == 'grid':
                    grid_cells[gid] =_instantiate_grid_cell(population, gid, module, nxy)
                elif ctype == 'place':
                    place_cells[gid] = _instantiate_place_cell(population, gid, module, nxy)
                if i == ncells - 1:
                    end_gid = gid
                    population_context[population][ctype].append(start_gid)
                    population_context[population][ctype].append(end_gid)
                gid += 1

            if feature_ctypes[ctype] == 0:
                cells = grid_cells
            elif feature_ctypes[ctype] == 1:
                cells = place_cells
            total_fields = 0
            for gid in cells:
                cell = cells[gid]
                if cell['Cell Type'] == 0:
                    total_fields += 1
                elif cell['Cell Type'] == 1:
                    total_fields += cell['Field Width'].shape[0]
            _, xy_offsets, _, _ = generate_spatial_offsets(total_fields, arena_dimension=arena_dimension, scale_factor=1.0)
            curr_pos   = 0
            for gid in cells:
                cell = cells[gid]
                if cell['Cell Type'] == 0:  
                    cell['X Offset'] = np.array([xy_offsets[curr_pos,0]], dtype='float32')
                    cell['Y Offset'] = np.array([xy_offsets[curr_pos,1]], dtype='float32')
                    curr_pos += 1
                elif cell['Cell Type'] == 1:
                    num_fields = cell['Field Width'].shape[0]
                    cell['X Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+num_fields,0], dtype='float32')
                    cell['Y Offset'] = np.asarray(xy_offsets[curr_pos:curr_pos+num_fields,1], dtype='float32')
                    curr_pos += num_fields
    return grid_cells, place_cells

def init_context():

    mesh   = _generate_mesh()
    nx, ny = mesh[0].shape[0], mesh[0].shape[1]
    population_context = {'MPP': {'grid': [n_mpp_grid], 'place': [n_mpp_place]}, \
                          'LPP': {'grid': [n_lpp_grid], 'place': [n_lpp_place]}}
    grid_cells, place_cells = _build_cells(population_context, context.module, (nx, ny))
    context.update(locals())

def _generate_mesh(scale_factor=1.0, arena_dimension=arena_dimension, resolution=resolution):
    arena_x_bounds = [-arena_dimension * scale_factor / 2., arena_dimension * scale_factor / 2.]
    arena_y_bounds = [-arena_dimension * scale_factor / 2., arena_dimension * scale_factor / 2.]
    arena_x        = np.arange(arena_x_bounds[0], arena_x_bounds[1], resolution)
    arena_y        = np.arange(arena_y_bounds[0], arena_y_bounds[1], resolution)
    return np.meshgrid(arena_x, arena_y, indexing='ij')

@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), 
               default="../config/optimize_DG_PP_config.yaml")
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
        #compare_ratemap(context, plot=False)
        report_cost(context)
        report_xy_offsets(context, plot=False, save=True)
        report_rate_map(context, plot=False, save=True)
        report_fraction_active(context, plot=True, save=True)

#def unit_tests():
    #import plot_DG_PP_features_v2
    #from plot_DG_PP_feau

def compare_ratemap(context, plot=False):
    cells = get_cell_types_from_context(context)
    gids = cells.keys()
    cell = cells[gids[0]]
    xp, yp = context.mesh

    tic = time.time()
    rate_map_original = _rate_map(xp, yp, cell['Grid Spacing'][0], cell['Grid Orientation'][0], cell['X Offset Scaled'][0], cell['Y Offset Scaled'][0], 0)
    elapsed = time.time() - tic
    print('It took %f seconds to calculate rate maps for a cells' % (elapsed)) 
 
    tic = time.time()
    rate_map_new = generate_spatial_ratemap(cell['Cell Type'][0], cell, None, xp, yp, 20.0, 20.0, None)
    elapsed = time.time() - tic
    print('It took %f seconds to calcualte rate map the other way' % elapsed)
    difference_map = np.abs(rate_map_new - rate_map_original)
    print('min diff, max diff = %f, %f' % (np.min(difference_map), np.max(difference_map)))
  
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(rate_map_original, cmap='inferno')
    plt.colorbar()
    plt.title('old')
    plt.subplot(1,3,2)
    plt.imshow(rate_map_new, cmap='inferno')
    plt.colorbar()
    plt.title('new')
    plt.subplot(1,3,3)
    plt.imshow(difference_map, cmap='inferno')
    plt.colorbar()
    plt.title('difference')

    if plot:
        plt.show()

def report_cost(context):
    x0 = context.x0_array
    features = calculate_features(x0)
    _, objectives = get_objectives(features)

    print('Scale factor: %f' % x0[0])
    print('xt1: %f' % x0[1])
    print('xt2: %f' % x0[2])
    print('gain: %f' % x0[3])
    print('Module: %d' % context.module)
    for objective in objectives.keys():
        print('Objective: %s has cost %f' % (objective, objectives[objective]))

def report_rate_map(context, plot=False, save=True):
    cells = get_cell_types_from_context(context)
    gids = cells.keys()
    rate_maps = []

    for gid in gids:
        cell = cells[gid]
        rate_maps.append(cell['Rate Map'].reshape(cell['Nx'][0], cell['Ny'][0]))
    rate_maps = np.asarray(rate_maps, dtype='float32')
    
    summed_map = np.sum(rate_maps, axis=0)
    mean_map   = np.mean(rate_maps, axis=0)
    var_map    = np.var(rate_maps, axis=0)

    ctype  = context.cell_type
    module = context.module

    import matplotlib.pyplot as plt
    plt.figure()

    plt.subplot(1,3,1)
    plt.imshow(summed_map, cmap='inferno')
    plt.colorbar()
    plt.title('Summed map for %s cells module %d' % (ctype, module))

    plt.subplot(1,3,2)
    plt.imshow(mean_map, cmap='inferno')
    plt.colorbar()
    plt.title('Mean map for %s cells module %d' % (ctype, module))

    plt.subplot(1,3,3)
    plt.imshow(var_map, cmap='inferno')
    plt.colorbar()
    plt.title('Var map for %s cells module %d' % (ctype, module))
    
    if save:
        plt.savefig('%s-module-%d-ratemap.png' % (ctype, module))
    if plot:
        plt.show()
    


def report_xy_offsets(context, plot=False, save=True):
    cells = get_cell_types_from_context(context)
    pop_xy_offsets = [] 
    for gid in cells:
        cell   = cells[gid]
        offsets = zip(cell['X Offset Scaled'], cell['Y Offset Scaled'])
        for (x_offset, y_offset) in offsets:
            pop_xy_offsets.append((x_offset, y_offset))
    pop_xy_offsets = np.asarray(pop_xy_offsets, dtype='float32')

    ctype  = context.cell_type
    module = context.module

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(pop_xy_offsets[:,0], pop_xy_offsets[:,1])
    plt.title('xy offsets for %s cells in module %d' % (ctype, module))

    if save:
        plt.savefig('%s-module-%d-xyoffsets.png' % (ctype, module))
    if plot:
        plt.show()

def report_fraction_active(context, plot=False, save=True):
    cells = get_cell_types_from_context(context)
    fraction_active = _fraction_active(cells)
    
    fraction_active_im = np.zeros((20,20))
    for (i,j) in fraction_active:
        fraction_active_im[i,j] = fraction_active[(i,j)]
    
    import matplotlib.pyplot as plt

    plt.subplot(1,2,1)
    plt.imshow(fraction_active_im, cmap='inferno')
    plt.colorbar()
    plt.title('fraction active')

    plt.subplot(1,2,2)
    plt.imshow(fraction_active_im - context.fraction_active_target, cmap='inferno')
    plt.colorbar()
    plt.title('fraction active distance from target')
 
    ctype  = context.cell_type
    module = context.module
    if save:
        plt.savefig('%s-module-%d-fractionactive.png' % (ctype, module))
    if plot:
        plt.show()

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
    
    
def _calculate_rate_maps(x, context):
    cells        = get_cell_types_from_context(context)
    xp, yp       = context.mesh
    scale_factor = x[0]

    ratemap_kwargs         = dict()
    ratemap_kwargs['xt1']  = x[1]
    ratemap_kwargs['xt2']  = x[2]
    ratemap_kwargs['gain'] = x[3]
    
    
    for gid in cells:
        cell  = cells[gid]
        ctype = cell['Cell Type']

        if ctype == 0: # Grid
            orientation = cell['Grid Orientation']
            spacing     = cell['Grid Spacing']
        elif ctype == 1: # Place
            spacing     = cell['Field Width']
            orientation = [0.0 for _ in range(len(spacing))]

        xf, yf    = cell['X Offset'], cell['Y Offset']
        xf_scaled = cell['X Offset'] * scale_factor
        yf_scaled = cell['Y Offset'] * scale_factor
        cell['X Offset Scaled'] = np.asarray(xf_scaled, dtype='float32')
        cell['Y Offset Scaled'] = np.asarray(yf_scaled, dtype='float32')
        rate_map = generate_spatial_ratemap(ctype, cell, None, xp, yp, context.grid_peak_rate, \
                                            context.place_peak_rate, None, **ratemap_kwargs)
            #rate_map += _rate_map(xp, yp, spacing[n], orientation[n], xf_scaled[n], yf_scaled[n], ctype)
        cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')

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
