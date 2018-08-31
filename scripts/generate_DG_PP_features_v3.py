import sys, os, time, random, click, logging
import numpy as np
from pprint import pprint

from scipy.optimize import minimize
from scipy.optimize import basinhopping

import dentate
import dentate.utils as utils
from dentate.utils import list_find, get_script_logger
from dentate.stimulus import generate_spatial_offsets

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
nfield_probabilities = np.asarray([0.8, 0.15, 0.05])

nmodules           = 10
modules            = np.arange(nmodules)
grid_orientation   = [local_random.uniform(0, np.pi/3.) for i in range(nmodules)]
field_width_params = [35.0, 0.32]
field_width        = lambda x : 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
max_field_width    = field_width(1.)


n_mpp_grid      = 5000
n_mpp_place     = 50
n_lpp_place     = 50
n_lpp_grid      = 0
arena_dimension = 100.
resolution      = 5.
feature_population = {'MPP': 0, 'LPP': 1}
feature_ctypes     = {'grid': 0, 'place': 1}

context = Context()

def _instantiate_place_cell(population, gid, module, nxy):
    cell = {}
    #module    = local_random.choice(modules)
    field_set = [1,2,3]
    nfields   = place_field_random.choice(field_set, p=nfield_probabilities, size=(1,))
    cell_field_width = []
    for n in range(nfields[0]):
        cell_field_width.append(field_width(local_random.random()))
    
    cell['gid']         = np.array([gid], dtype='int32')
    cell['Population']  = np.array([feature_population[population]], dtype='uint8')
    cell['Cell Type']   = np.array([feature_ctypes['place']], dtype='uint8')
    cell['Module']      = np.array([module], dtype='uint8')
    cell['Field Width'] = np.asarray(cell_field_width, dtype='float32')
    cell['Nx']           = np.array([nxy[0]], dtype='int32')
    cell['Ny']           = np.array([nxy[1]], dtype='int32')
    
    return cell 

def _instantiate_grid_cell(population, gid, module, nxy):
    cell   = {}
    #module = local_random.choice(modules)
    orientation = grid_orientation[module]
    spacing     = field_width(float(module)/float(np.max(modules)))

    delta_orientation = local_random.uniform(-10, 10) # Look into gaussian
    delta_spacing     = local_random.uniform(-10, 10) # Look into gaussian

    cell['gid']          = np.array([gid], dtype='int32')
    cell['Population']   = np.array([feature_population[population]], dtype='uint8')
    cell['Cell Type']    = np.array([feature_ctypes['grid']], dtype='uint8')
    cell['Module']       = np.array([module], dtype='uint8')
    cell['Grid Spacing'] = np.array([spacing + delta_spacing], dtype='float32')
    cell['Orientation']  = np.array([orientation + delta_orientation], dtype='float32')
    cell['Nx']           = np.array([nxy[0]], dtype='int32')
    cell['Ny']           = np.array([nxy[1]], dtype='int32')

    return cell 

def _build_cells(population_context, module, nxy):
    grid_cells, place_cells = {}, {}
    gid = 1
    for population in population_context.keys():
        current_population = population_context[population]
        for ctype in current_population.keys():
            start_gid  = gid
            ncells     = current_population[ctype][0]
            if ncells == 0:
                population_context[population][ctype] += [None, None]
            for i in range(ncells):
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
            for gid in cells.keys():
                cell = cells[gid]
                if cell['Cell Type'] == 0:
                    total_fields += 1
                elif cell['Cell Type'] == 1:
                    total_fields += cell['Field Width'].shape[0]
            _, xy_offsets, _, _ = generate_spatial_offsets(total_fields, arena_dimension=arena_dimension, scale_factor=1.0)
            curr_pos   = 0
            for gid in cells.keys():
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
        report_cost(context)
        report_xy_offsets(context, plot=False, save=True)
        report_rate_map(context, plot=True, save=True)

def report_cost(context):
    x0 = context.x0
    features = calculate_features(x0)
    _, objectives = get_objectives(features)

    print('Scale factor: %f' % x0['scale_factor'])
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
    gids = cells.keys()
    pop_xy_offsets = [] 
    for gid in gids:
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
    

def config_controller():
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
    cells    = get_cell_types_from_context(context)
    features = {}

    minmax_eval     = _peak_to_trough(cells)
    fraction_active = _fraction_active(cells) 

    features['minmax sum eval'] = minmax_eval[0]
    features['minmax var eval'] = minmax_eval[1]
    features['fraction active'] = fraction_active
    return features

def get_objectives(features):
    feature_names = context.feature_names
    for feature in features.keys():
        if feature not in feature_names:
            raise Exception('Feautre %s could not be found' % feature)

    objectives = {}
    objectives['minmax sum error'] = (features['minmax sum eval'] - context.target_val['minmax sum error']) ** 2
    objectives['minmax var error'] = (features['minmax var eval'] - context.target_val['minmax var error']) ** 2
    fraction_active  = features['fraction active']
    diff_frac_active = {(i,j): np.abs(fraction_active[(i,j)] - context.fraction_active_target) \
                        for (i,j) in fraction_active.keys()}
    fraction_active_errors = np.asarray([diff_frac_active[(i,j)] for (i,j) in diff_frac_active.keys()])
    fraction_active_mean_error = np.mean(fraction_active_errors)
    fraction_active_var_error  = np.var(fraction_active_errors)
    objectives['fraction active mean error'] = (fraction_active_mean_error - context.target_val['fraction active mean error']) ** 2
    objectives['fraction active var error'] = (fraction_active_var_error - context.target_val['fraction active var error']) ** 2

    return features, objectives

def _peak_to_trough(cells):
    rate_maps = []
    for gid in cells.keys():
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

def _fraction_active(cells, target=0.15):
    rate_maps = []
    for gid in cells.keys():
        cell     = cells[gid]
        nx, ny   = cell['Nx'][0], cell['Ny'][0]
        rate_map = cell['Rate Map'].reshape(nx, ny)
        rate_maps.append(rate_map)
    rate_maps = np.asarray(rate_maps, dtype='float32')
    nxx, nyy  = np.meshgrid(np.arange(nx), np.arange(ny))
    coords    = zip(nxx.reshape(-1,), nyy.reshape(-1,))
    
    factive = lambda px, py: _calculate_fraction_active(rate_maps[:,px,py])
    return {(px,py): factive(px, py) for (px, py) in coords}

def _calculate_fraction_active(rates, threshold=1.0):
    #max_rate = np.max(rates)
    #normalized_rates = np.divide(rates, max_rate)
    num_active = len(np.where(rates > threshold)[0])
    fraction_active = np.divide(float(num_active), len(rates))
    return fraction_active    
    
    
def _calculate_rate_maps(x, context):
    cells        = get_cell_types_from_context(context)
    xp, yp       = context.mesh
    scale_factor = x
    for gid in cells.keys():
        cell  = cells[gid]
        ctype = cell['Cell Type']

        if ctype == 0: # Grid
            orientation = cell['Orientation']
            spacing     = cell['Grid Spacing']
        elif ctype == 1: # Place
            spacing     = cell['Field Width']
            orientation = [0.0 for _ in range(len(spacing))]

        xf, yf    = cell['X Offset'], cell['Y Offset']
        xf_scaled = cell['X Offset'] * scale_factor
        yf_scaled = cell['Y Offset'] * scale_factor
        cell['X Offset Scaled'] = np.asarray(xf_scaled, dtype='float32')
        cell['Y Offset Scaled'] = np.asarray(yf_scaled, dtype='float32')
        rate_map = np.zeros((xp.shape[0], xp.shape[1]))
        for n in range(len(spacing)):
            rate_map += _rate_map(xp, yp, spacing[n], orientation[n], xf_scaled[n], yf_scaled[n], ctype)
        cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')

def _rate_map(xp, yp, spacing, orientation, x_offset, y_offset, ctype):    
    nx, ny = xp.shape
    mask   = None
    if ctype == 1:
        radius    = spacing / 2.
        distances = np.sqrt( (xp - x_offset) ** 2 + (yp - y_offset) ** 2)
        mask      = np.zeros(xp.shape)
        mask[distances < radius] = 1   
    theta_k   = [np.deg2rad(-30.), np.deg2rad(30.), np.deg2rad(90.)]
    inner_sum = np.zeros(xp.shape)
    for k in range(len(theta_k)):
        inner_sum += np.cos( ((4 * np.pi) / (np.sqrt(3.) * spacing)) * (np.cos(theta_k[k]) * (xp - x_offset) + (np.sin(theta_k[k]) * (yp - y_offset))))
    
    transfer = lambda z: np.exp(0.3 * (z - (-1.5))) - 1.
    rate_map = transfer(inner_sum)
    if mask is not None:
        return rate_map * mask
    return rate_map    
   




if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv)+1):])
