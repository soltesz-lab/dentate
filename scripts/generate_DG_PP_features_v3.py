
import sys, os, time, random, click, logging
import numpy as np
from pprint import pprint
from copy import deepcopy

from scipy.optimize import minimize
from scipy.optimize import basinhopping

import dentate
import dentate.utils as utils
from dentate.utils import list_find, get_script_logger
from dentate.stimulus import generate_spatial_offsets


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


n_mpp_grid    = 50
n_mpp_place   = 50
n_lpp_place   = 50
n_lpp_grid    = 0
feature_population = {'MPP': 0, 'LPP': 1}
feature_ctypes     = {'grid': 0, 'place': 1}

def _instantiate_place_cell(population, gid, nxy):
    cell = {}
    module    = local_random.choice(modules)
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

def _instantiate_grid_cell(population, gid, nxy):
    cell   = {}
    module = local_random.choice(modules)
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

def init_context(nxy, verbose):
    tic      = time.time()    
    temp_gid = 1

    population_context = {'MPP': {'grid': [n_mpp_grid], 'place': [n_mpp_place]}, \
                          'LPP': {'grid': [n_lpp_grid], 'place': [n_lpp_place]}}
    populations        = population_context.keys()
    cell_corpus        = {population: {ctype: {} for ctype in feature_ctypes.keys()} for population in populations}

    for population in populations:
        current_population = population_context[population]
        for ctype in current_population.keys():
            start_gid = temp_gid
            ncells    = current_population[ctype][0]
            if ncells == 0:
                population_context[population][ctype] += [None, None]
            for i in range(ncells):
                if ctype == 'grid':
                    cell_corpus[population][ctype][temp_gid] =_instantiate_grid_cell(population, temp_gid, nxy)
                elif ctype == 'place':
                    cell_corpus[population][ctype][temp_gid] = _instantiate_place_cell(population, temp_gid, nxy)
                if i == ncells - 1:
                    end_gid = temp_gid
                    population_context[population][ctype].append(start_gid)
                    population_context[population][ctype].append(end_gid)
                temp_gid += 1

    elapsed = time.time() - tic
    if verbose:
        logger.info('It took %f seconds to initialize all cells' % elapsed)
    return cell_corpus, population_context

def _generate_mesh(scale_factor=1.0, arena_dimension=100., resolution=5.):
    arena_x_bounds = [-arena_dimension * scale_factor / 2., arena_dimension * scale_factor / 2.]
    arena_y_bounds = [-arena_dimension * scale_factor / 2., arena_dimension * scale_factor / 2.]
    arena_x        = np.arange(arena_x_bounds[0], arena_x_bounds[1], resolution)
    arena_y        = np.arange(arena_y_bounds[0], arena_y_bounds[1], resolution)
    return np.meshgrid(arena_x, arena_y, indexing='ij')

@click.command()
@click.option("--optimize", '-o', is_flag=True, required=True)
@click.option("--lbound", type=float, required=False, default=1)
@click.option("--ubound", type=float, required=False, default=100)
@click.option("--types-path", default=None, required=True, type=click.Path(file_okay=True, dir_okay=True))
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(optimize, lbound, ubound, types_path, verbose):
    print('here')
    mesh   = _generate_mesh()
    nx, ny = mesh[0][0], mesh[0][1]
    cell_corpus, population_context = init_context((nx, ny), verbose)
    if optimize:
        bounds = (lbound, ubound)
        main_optimization(cell_corpus, mesh, bounds, verbose)

def main_optimization(cell_corpus, mesh, bounds, verbose):
    from mpi4py import MPI
    comm        = MPI.COMM_WORLD
    cell_corpus = comm.bcast(cell_corpus)
    lbound, ubound      = bounds
    module_to_parameter = None
    if comm.rank == 0:
        init_parameters = np.random.randint(lbound, ubound, (nmodules,))
        if verbose:
            print('Initial parameters: ', init_parameters)
        module_to_parameter = distribute_modules_to_ranks(comm, init_parameters)
    module_to_parameter     = comm.scatter(module_to_parameter, root=0)
    module_to_parameter     = {module: scale_factor for (module, scale_factor) in module_to_parameter}


def distribute_modules_to_ranks(comm, init_parameters):
    size = comm.size
    processing_information = [ [] for _ in np.arange(size)]
    for i in range(size):
        for j in range(i, nmodules, size):
            processing_information[i].append((j, init_parameters[j]))
    return processing_information


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv)+1):])
