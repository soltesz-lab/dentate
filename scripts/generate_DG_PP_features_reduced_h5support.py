import sys, os, time, random, click, logging, shutil
import numpy as np
import pickle
from copy import deepcopy

from scipy.optimize import minimize
from scipy.optimize import basinhopping

from mpi4py import MPI
import h5py
from neuroh5.io import append_cell_attributes, read_population_ranges, read_cell_attributes
import dentate
from dentate.utils import list_find
from dentate.stimulus import generate_spatial_offsets

script_name = 'generate_DG_PP_features_reduced_h5support.py'
logging.basicConfig()
logger = logging.getLogger(script_name)

io_size=-1
chunk_size=1000
value_chunk_size=1000

field_width_params = [35.0, 0.32]
field_width = lambda x : 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
max_field_width = field_width(1.)

feature_grid = 0
feature_place_field = 1

N_MPP = 30000
N_MPP_GRID = int(N_MPP * 0.7)
N_MPP_PLACE = N_MPP - N_MPP_GRID

N_LPP = 1300 #int(N_MPP * 1.10)
N_LPP_PLACE = N_LPP 
N_LPP_GRID = 0

arena_dimension = 100.
init_scale_factor = 10.0
init_orientation_jitter = [-10., 10.] #[np.deg2rad(-10.), np.deg2rad(10.)]
init_lambda_jitter = [-10., 10.]
resolution = 5.
nmodules = 10
modules = np.arange(nmodules)

param_list, cost_evals = [], []


def dcheck(x,y,xi,yi,spacing):
    radius = spacing / 2.
    distances = np.sqrt( (x - xi) ** 2 + (y - yi) ** 2)
    mask = np.zeros(x.shape)
    mask[distances < radius] = 1
    return mask

def place_fill_map(xp, yp, spacing, orientation, xf, yf):

    nx, ny = xp.shape
    rate_map = np.zeros((nx, ny))
    rate_map = get_rate(xp,yp, spacing, orientation, xf, yf, ctype='place')
    return rate_map


def grid_fill_map(xp, yp, spacing, orientation, xf, yf):

    nx, ny = xp.shape
    rate_map = np.zeros((nx, ny))
    rate_map = get_rate(xp, yp, spacing, orientation, xf, yf, ctype='grid')
    return rate_map

def get_rate(x, y, grid_spacing, orientation, x_offset, y_offset,ctype='grid'):
    mask = None
    if ctype == 'place':
         mask = dcheck(x,y,x_offset, y_offset, grid_spacing)
    theta_k = [np.deg2rad(-30.), np.deg2rad(30.), np.deg2rad(90.)]
    inner_sum = np.zeros(x.shape)
    for k in range(len(theta_k)):
        inner_sum += np.cos( ((4 * np.pi) / (np.sqrt(3)*grid_spacing)) * (np.cos(theta_k[k]) * (x - x_offset) + (np.sin(theta_k[k]) * (y - y_offset))))
    rate_map = transfer(inner_sum)
    if mask is not None:
        return rate_map * mask
    return rate_map

def transfer(z, a=0.3, b=-1.5):
    return np.exp(a*(z-b)) - 1 


def to_file(rate_map, fn, module=1):
    nx, ny = rate_map.shape
    f = open(fn, 'w')
    for i in range(nx):
        for j in range(ny):
            f.write(str(rate_map[i,j]) + '\t')
        f.write('\n')

def generate_mesh(scale_factor=init_scale_factor):

    mega_arena_x_bounds = [-arena_dimension * scale_factor/2., arena_dimension * scale_factor/2.]
    mega_arena_y_bounds = [-arena_dimension * scale_factor/2., arena_dimension * scale_factor/2.]
    mega_arena_x = np.arange(mega_arena_x_bounds[0], mega_arena_x_bounds[1], resolution)
    mega_arena_y = np.arange(mega_arena_y_bounds[0], mega_arena_y_bounds[1], resolution)
    return np.meshgrid(mega_arena_x, mega_arena_y, indexing='ij')

def init_generate_populations(gen_rate=True, scale_factor=6.*np.ones(nmodules)):
    tic = time.time()
    grid_feature_dict_MPP, place_feature_dict_MPP, xy_offsets_MPP, feature_types_MPP, orientation_MPP = init(population='MPP')
    grid_feature_dict_LPP, place_feature_dict_LPP, xy_offsets_LPP, feature_types_LPP, orientation_LPP = init(population='LPP')
    mega_arena_xp, mega_arena_yp = generate_mesh()
    elapsed = time.time() - tic
    print('Took %f seconds to initialize populations and generate meshgrid' % (elapsed))
 
 
    if gen_rate:
        for module in modules:
            mpp_rm_grid, mpp_rm_place = module_map(mega_arena_xp, mega_arena_yp, grid_feature_dict_MPP, place_feature_dict_MPP, module=module, population='MPP')

            fn = 'MPP/ratemap-module-'+str(module)+'-MPP-grid.txt'
            to_file(mpp_rm_grid,fn,module=module)
            fn = 'MPP/ratemap-module-'+str(module)+'-MPP-place.txt'
            to_file(mpp_rm_place,fn,module=module)

            lpp_rm_grid, lpp_rm_place = module_map(mega_arena_xp, mega_arena_yp, grid_feature_dict_LPP, place_feature_dict_LPP, module=module, population='LPP')

            fn = 'LPP/ratemap-module-'+str(module)+'-LPP-grid.txt'
            to_file(lpp_rm_grid,fn,module=module)
            fn = 'LPP/ratemap-module-'+str(module)+'-LPP-place.txt'
            to_file(lpp_rm_place,fn,module=module)

    MPP_info = (grid_feature_dict_MPP, place_feature_dict_MPP, xy_offsets_MPP, feature_types_MPP, orientation_MPP)
    LPP_info = (grid_feature_dict_LPP, place_feature_dict_LPP, xy_offsets_LPP, feature_types_LPP, orientation_LPP)

    grid_dicts = (grid_feature_dict_MPP, grid_feature_dict_LPP)
    place_dicts = (place_feature_dict_MPP, place_feature_dict_LPP)
    return MPP_info, LPP_info, mega_arena_xp, mega_arena_yp
   

def read_file(fn):
    rate_map = []
    f = open(fn, 'r')
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        curr_rates = []
        for val in line[0:-1]:
            curr_rates.append(float(val))
        rate_map.append(curr_rates)
    return np.asarray(rate_map)

def rate_histogram(features_dict, xp, yp, xoi, yoi, ctype='grid', module=0):
    r = []
    for idx in features_dict.keys():
        cell = features_dict[idx]
        if cell['Module'][0] == module:       
            spacing, orientation = None, 0.0
            if cell.has_key('Grid Spacing'):
                spacing = cell['Grid Spacing']
                orientation = cell['Grid Orientation']
            else:
                spacing = cell['Field Width']
            x_offset, y_offset = cell['X Offset'], cell['Y Offset']
            rate = get_rate_2(np.array([xp[xoi,yoi]]), np.array([yp[xoi,yoi]]), spacing, orientation, x_offset, y_offset,ctype=ctype)
            r.append(rate[0])
    return np.asarray(r).reshape(-1,)


def make_hist(features_dict, xp, yp, population='MPP',ctype='grid',modules=[0],xoi=0, yoi=0):
    for module in modules:
        rate = rate_histogram(features_dict, xp, yp, xoi, yoi, module=module,ctype=ctype)
        fn = population+'/'+ctype+'-module-'+str(module)+'-rates-x-'+str(xoi)+'-y-'+str(yoi)+'.txt'
        list_to_file(fn, rate)

def make_hist_2(rmap, population='MPP', ctype='grid',modules=[0],xoi=0,yoi=0):
    for module in modules:
        rmatrix = rmap[module]
        rates = rmatrix[xoi,yoi,:]
        fn = population+'/'+ctype+'-module-'+str(module)+'-rates-x-'+str(xoi)+'-y-'+str(yoi)+'.txt'
        list_to_file(fn, rates)
    
def list_to_file(fn, r):
    f = open(fn, 'w')
    for v in r:
        f.write(str(v)+'\t')
    f.write('\n')
    f.close()    

def peak_to_trough(module_cells, modules=modules):
    minmax_evaluations = np.asarray([1.0 for _ in np.arange(nmodules)],dtype='float32')
    mean_evaluations = np.asarray([1.0 for _ in np.arange(nmodules)], dtype='float32')
    var_evaluations = np.asarray([0.0 for _ in np.arange(nmodules)], dtype='float32')

    for mod in module_cells.keys():
        cells = module_cells[mod]
        module_rate_map = None
        for (c,cell) in enumerate(cells):
            nx, ny = cell['Nx'][0], cell['Ny'][0]
            rate_map = cell['Rate Map'].reshape(nx, ny)
            if c == 0:
                module_rate_map = np.zeros((nx, ny))
            module_rate_map += rate_map
        minmax_evaluations[mod] = float(np.max(module_rate_map)) / float(np.min(module_rate_map))

        #nxx, nyy = np.meshgrid(np.arange(nx), np.arange(ny))
        #coords = zip(nxx.reshape(-1,), nyy.reshape(-1,))
        #rate_ratio = np.asarray([ [np.divide(float(module_rate_map[i,j]), float(module_rate_map[i2,j2])) for (i2,j2) in coords] for (i,j) in coords], dtype='float32')
        #rate_ratio = rate_ratio - np.identity(len(coords))
        #rate_ratio = rate_ratio[rate_ratio > 0.0].reshape(-1,)
        #mean_evaluations[mod] = 1.0 #np.mean(rate_ratio)
        #var_evaluations[mod]  = np.var(rate_ratio)
        
                

    return minmax_evaluations - 1., mean_evaluations - 1., var_evaluations

def fraction_active(module_cells, modules=modules, target=0.30):
    rates = {mod:[] for mod in modules}
    for mod in module_cells.keys():
        cells = module_cells[mod]
        for cell in cells:
            nx, ny = cell['Nx'][0], cell['Ny'][0]
            rate_map = cell['Rate Map'].reshape(nx, ny)
            rates[mod].append(rate_map)
    nxx, nyy = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = zip(nxx.reshape(-1,), nyy.reshape(-1,))
    frac_active_dict = {(i,j): {k:None for k in modules} for (i,j) in coords}
    diagonal_positions = [ (i,j) for (i,j) in frac_active_dict.keys()]
    for (px, py) in diagonal_positions:
        for key in rates.keys():
            module_maps = np.asarray(rates[key])
            position_rates = module_maps[:,px,py]
            frac_active = calculate_fraction_active(position_rates)
            frac_active_dict[(px,py)][key] = frac_active
    target_fraction_active = {(i,j): {k: target for k in modules} for (i,j) in frac_active_dict.keys()}
    diff_fraction_active = {(i,j): {k: np.abs(target_fraction_active[(i,j)][k]-frac_active_dict[(i,j)][k]) for k in modules} for (i,j) in frac_active_dict.keys()}
    
    module_error = [ [] for _ in range(len(modules))]
    for (i,j) in diff_fraction_active.keys():
        pos_errors = diff_fraction_active[(i,j)]
        for module in pos_errors.keys():
            mod_e = pos_errors[module]
            module_error[module].append(mod_e)
    module_error = np.asarray(module_error, dtype='float32')
    module_mean = np.array([ 0.0 for _ in range(len(modules))])
    module_var = np.array([ 0.0 for _ in range(len(modules))])

    for i in range(len(module_error)):
        module_mean[i] = np.mean(module_error[i])
        module_var[i] = np.var(module_error[i])
    return module_mean, module_var


def calculate_fraction_active(rates, threshold=0.1):
    max_rate = np.max(rates)
    normalized_rates = np.divide(rates, max_rate)
    num_active = len(np.where(normalized_rates > threshold)[0])
    fraction_active = float(num_active) / len(normalized_rates)
    return fraction_active               
            

def cost_func(x, cell_modules, mesh):
    sf = x
    param_list.append(sf)
    xp, yp = mesh
    for mod in cell_modules.keys():
        cells = cell_modules[mod]
        for (c,cell) in enumerate(cells):
            orientation, spacing = cell['Jittered Grid Orientation'], cell['Jittered Grid Spacing']
            xf, yf = cell['X Offset'][0], cell['Y Offset'][0]
            xf_scaled, yf_scaled = xf * sf[mod], yf * sf[mod]
            cell['X Offset Scaled'] = np.array([xf_scaled], dtype='float32')
            cell['Y Offset Scaled'] = np.array([yf_scaled], dtype='float32')
            rate_map = grid_fill_map(xp, yp, spacing, orientation, xf_scaled, yf_scaled)
            nx, ny = rate_map.shape
            cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')
            cell['Nx'] = np.array([nx], dtype='int32')
            cell['Ny'] = np.array([ny], dtype='int32')

    tic = time.time()
    minmax_ratio_evaluations, mean_ratio_evaluations, var_ratio_evaluations = peak_to_trough(cell_modules, modules=modules)
    minmax_ratio_sum = np.sum(minmax_ratio_evaluations)
    mean_ratio_sum = np.sum(mean_ratio_evaluations)
    var_ratio_sum = np.sum(var_ratio_evaluations)
    cost_peak_trough = (minmax_ratio_sum ** 2 + mean_ratio_sum ** 2 + var_ratio_sum ** 2)
 
    fraction_active_mean_evaluation, fraction_active_var_evaluation = fraction_active(cell_modules, modules=modules)
    frac_active_mean_cost = np.sum(fraction_active_mean_evaluation)
    frac_active_var_cost = np.sum(fraction_active_var_evaluation)
    cost_frac_active = frac_active_mean_cost ** 2 + frac_active_var_cost ** 2

    total_cost = 0.5 * (cost_peak_trough + cost_frac_active)
    cost_evals.append(total_cost)
    elapsed = time.time() - tic
    print('Cost: %f calculted in %f seconds' % (total_cost,elapsed))
    return total_cost

class OptimizationRoutine(object):
    def __init__(self, cells, mesh):
        self.cells = cells
        self.mesh = mesh

    def optimize(self, x0, bounds=None, verbose=False):
        param_list, cost_evals = [], []
        if bounds is None:
            bounds = [(1., 50.) for _ in x0]
        fnc = lambda x: cost_func(x, self.cells, self.mesh)
        minimizer_kwargs = dict(method='L-BFGS-B', bounds=bounds, options={'disp':True,'eps':1.0, 'maxiter':3})
        bh_output = basinhopping(fnc, x0, minimizer_kwargs=minimizer_kwargs, stepsize=10.0, T=2.0,disp=True, niter=3)

        if verbose:
            print(x0)
            print(bh_output.x)
            print(fnc(x0))
            print(fnc(bh_output.x))
        return param_list, cost_evals

class Cell_Population(object):
    def __init__(self, comm, coords_path, jitter_orientation=True, jitter_spacing=True):
        self.comm = comm
        self.coords_path = coords_path
        self.jitter_orientation = jitter_orientation
        self.jitter_spacing = jitter_spacing
        self.xp, self.yp = generate_mesh(scale_factor=1.0)

        self.local_random = random.Random()
        self.local_random.seed(0)
        self.feature_type_random = np.random.RandomState(0)

        self.mpp_grid  =  None
        self.mpp_place =  None
        self.lpp_grid  =  None
        self.lpp_place =  None

        self.population_ranges = read_population_ranges(self.coords_path, self.comm)[0]

    def full_init(self, scale_factors=1.0*np.ones(nmodules), full_map=False):
        self.initialize_cells(population='MPP')
        self.initialize_cells(population='LPP')
        self.generate_xy_offsets()
        self.calculate_rate_maps(scale_factors, full_map=full_map)

    def initialize_cells(self, population='MPP'):
        self.population_start, self.population_count = self.population_ranges[population]
        grid_orientation = [self.local_random.uniform(0., np.pi/3.) for i in range(nmodules)]
        feature_type_values = np.asarray([0, 1])
        if population == 'MPP':
            NCELLS = N_MPP
            feature_type_probs = np.asarray([0.3, 0.7])
        elif population == 'LPP':
            NCELLS = N_LPP
            feature_type_probs = np.asarray([0.0, 1.0])
        feature_types = self.feature_type_random.choice(feature_type_values,p=feature_type_probs, size=(NCELLS,))

        if population == 'MPP':
            self.mpp_grid, self.mpp_place = self._build_cells(NCELLS, population, feature_types, grid_orientation)
        elif population == 'LPP':
            self.lpp_grid, self.lpp_place = self._build_cells(NCELLS, population, feature_types, grid_orientation)

    def _build_cells(self, N, pop, feature_types, grid_orientation):
        grid_feature_dict, place_feature_dict = {}, {}
        for i in range(N):
            gid = self.population_start + i
            feature_type = feature_types[i]
            if feature_type == 0: # Grid cell
                this_module = self.local_random.choice(modules)
                orientation = grid_orientation[this_module]
                spacing = field_width(float(this_module)/float(np.max(modules)))
                grid_feature_dict[gid] = self._build_grid_cell(gid, orientation, spacing, this_module)
            elif feature_type == 1:
                this_module = self.local_random.choice(modules)
                cell_field_width = field_width(self.local_random.random())
                place_feature_dict[gid] = self._build_place_cell(gid, cell_field_width, this_module)
        return grid_feature_dict, place_feature_dict

    def _build_place_cell(self, gid, cell_field_width, module):
        cell = {}
        cell['gid'] = np.array([gid], dtype='int32')
        cell['Population'] = np.array([1], dtype='uint8')
        cell['Module'] = np.array([module], dtype='uint8')
        cell['Field Width'] = np.array([cell_field_width], dtype='float32')
        return cell

    def _build_grid_cell(self, gid, orientation, spacing, module):
        cell = {}
        cell['gid'] = np.array([gid], dtype='int32')
        cell['Population'] = np.array([0], dtype='uint8')
        cell['Module'] = np.array([module],dtype='uint8')
        cell['Grid Spacing'] = np.array([spacing],dtype='float32')
        cell['Grid Orientation'] = np.array([orientation],dtype='float32')
        if self.jitter_orientation:
            delta_orientation = self.local_random.uniform(init_orientation_jitter[0], init_orientation_jitter[1])
            cell['Jittered Grid Orientation'] = np.array([cell['Grid Orientation'][0] + delta_orientation], dtype='float32')
        if self.jitter_spacing: 
            delta_spacing = self.local_random.uniform(init_lambda_jitter[0], init_lambda_jitter[1])
            cell['Jittered Grid Spacing'] = np.array([cell['Grid Spacing'][0] + delta_spacing], dtype='float32')
        return cell

    def generate_xy_offsets(self):
        N = 0
        present = [False, False, False, False]
        if self.mpp_grid is not None:
            N += len(self.mpp_grid.keys())
            present[0] = True
        if self.mpp_place is not None:
            N += len(self.mpp_place.keys())
            present[1] = True
        if self.lpp_grid is not None:
            N += len(self.lpp_grid.keys())
            present[2] = True
        if self.lpp_place is not None:
            N += len(self.lpp_place.keys())
            present[3] = True

        _, xy_offsets, _, _ = generate_spatial_offsets(N, arena_dimension=arena_dimension, scale_factor=1.0)
        counter = 0
        if present[0]:
            counter = self._generate_xy_offsets(self.mpp_grid, xy_offsets, counter)
        if present[1]:
            counter = self._generate_xy_offsets(self.mpp_place, xy_offsets, counter)
        if present[2]:
            counter = self._generate_xy_offsets(self.lpp_grid, xy_offsets, counter)
        if present[3]:
            counter = self._generate_xy_offsets(self.lpp_place, xy_offsets, counter)
 
    def _generate_xy_offsets(self, cells, xy_offsets, counter):
        for key in cells.keys():
            cell = cells[key]
            cell['X Offset'] = np.array([xy_offsets[counter,0]], dtype='float32')
            cell['Y Offset'] = np.array([xy_offsets[counter,0]], dtype='float32')
            counter += 1
        return counter

    def calculate_rate_maps(self, scale_factors, full_map=False):
        if self.mpp_grid is not None:
            self._calculate_rate_maps(self.mpp_grid, scale_factors, cell_type='grid', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing, full_map=full_map)
        if self.mpp_place is not None:
            self._calculate_rate_maps(self.mpp_place, scale_factors, cell_type='place', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing, full_map=full_map)
        if self.lpp_grid is not None:
            self._calculate_rate_maps(self.lpp_grid, scale_factors, cell_type='grid', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing, full_map=full_map)
        if self.lpp_place is not None:
            self._calculate_rate_maps(self.lpp_place, scale_factors, cell_type='place', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing, full_map=full_map)

    def _calculate_rate_maps(self,cells, scale_factors, cell_type='grid', jittered_orientation=False, jittered_spacing=False, full_map=False):

        module_map = {k: None for k in np.arange(nmodules)}
        for key in cells.keys():
            cell = cells[key]
            x_offset, y_offset = None, None
            this_module = cell['Module'][0]
            if module_map[this_module] is None:
                module_map[this_module] = generate_mesh(scale_factor=scale_factors[this_module])
            xp,  yp = module_map[this_module]
                
            x_offset_scaled = cell['X Offset'][0] * scale_factors[this_module]
            y_offset_scaled = cell['Y Offset'][0] * scale_factors[this_module]
            cell['X Offset Scaled'] = np.array([x_offset_scaled], dtype='float32')
            cell['Y Offset Scaled'] = np.array([y_offset_scaled], dtype='float32')
            if cell_type == 'grid':
                grid_spacing, grid_orientation = None, None
                if jittered_spacing:
                    grid_spacing = cell['Jittered Grid Spacing'][0]
                else:
                    grid_spacing = cell['Grid Spacing'][0]
                if jittered_orientation:
                    grid_orientation = cell['Jittered Grid Orientation'][0]
                else:
                    grid_orientation = cell['Grid Orientation'][0]
                if full_map:
                    full_rate_map = grid_fill_map(xp, yp, grid_spacing, grid_orientation, x_offset_scaled, y_offset_scaled).reshape(-1,)
                    cell['Full Rate Map'] = full_rate_map.astype('float32')
                    cell['Full Nx'] = np.array([xp.shape[0]], dtype='int32')
                    cell['Full Ny'] = np.array([xp.shape[1]], dtype='int32')

                rate_map = grid_fill_map(self.xp, self.yp, grid_spacing, grid_orientation, x_offset_scaled, y_offset_scaled).reshape(-1,)
                cell['Rate Map'] = rate_map.astype('float32')
                cell['Nx'] = np.asarray([self.xp.shape[0]], dtype='int32')
                cell['Ny'] = np.asarray([self.xp.shape[1]], dtype='int32')
            elif cell_type == 'place':
                place_orientation = 0.0
                place_width = cell['Field Width'][0]
                rate_map = place_fill_map(self.xp, self.yp, place_width, place_orientation, x_offset_scaled, y_offset_scaled).reshape(-1,)
                cell['Rate Map'] = rate_map.astype('float32')
                cell['Nx'] = np.array([self.xp.shape[0]], dtype='int32')
                cell['Ny'] = np.array([self.xp.shape[1]], dtype='int32')

@click.command()
@click.option("--optimize", '-o', is_flag=True, required=True)
@click.option("--input-path", default=None, required=False, type=click.Path(file_okay=True, dir_okay=True))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--output-path", required=False, type=click.Path(file_okay=True, dir_okay=True))
@click.option("--iterations", type=int, required=False, default=10)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option("--lbound", type=float, required=False, default=1.)
@click.option("--ubound", type=float, required=False, default=50.)

def main(optimize, input_path, coords_path, output_path, lbound, ubound, iterations, verbose):
    comm = MPI.COMM_WORLD
    tic = time.time()
    if optimize:
        mpp_grid = None
        if input_path is not None:
            mpp_grid = {}
            neuroh5_cells = read_cell_attributes(input_path, "MPP", namespace="Grid Input Features")
            for (gid, cell_attr) in neuroh5_cells:
                mpp_grid[gid] = cell_attr
            if verbose:
                print('Data read in for %d cells..' % (len(mpp_grid.keys())))
        else:
            cell_corpus = Cell_Population(comm, coords_path)
            cell_corpus.full_init()
            mpp_grid = cell_corpus.mpp_grid
            append_cell_attributes('temp-'+output_path, 'MPP', mpp_grid, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            if verbose:
                print('Cells initialized')
        main_optimization(comm, output_path, mpp_grid, lbound, ubound, iterations, verbose)

    else:
        if input_path is not None:
            mpp_grid = {}
            neuroh5_cells = read_cell_attributes(input_path, "MPP", namespace="Grid Input Features")
            for (gid, cell_attr) in neuroh5_cells:
                mpp_grid[gid] = cell_attr
            if verbose:
                print('Data read in for %d cells..' % (len(mpp_grid.keys())))
        else:
            cell_corpus = Cell_Population(comm, coords_path)
            cell_corpus.full_init()
            mpp_grid = cell_corpus.mpp_grid
            append_cell_attributes('temp_'+output_path, 'MPP', mpp_grid, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            if verbose:
                print('Cells initialized')
            main_hardcoded(comm, output_path, mpp_grid)
            elapsed = time.time() - tic
            if verbose:
                print('Completed in %f seconds...' % elapsed)


def main_optimization(comm, output_path, cells, lbound, ubound, iterations, verbose):
    init_scale_factor = np.random.randint(lbound, ubound+1, (nmodules, iterations))
    for t in range(iterations):
        cell_copy = deepcopy(cells)
        scale_factor0 = init_scale_factor[:,t]
        if verbose:
            print(t, scale_factor0)
        tic = time.time()
        elapsed = time.time() - tic
        module_mpp_grid = gid_to_module_dictionary(cell_copy)
        mesh = generate_mesh(scale_factor=1.0)
        bounds = [(lbound, ubound) for sf in scale_factor0]
        opt = OptimizationRoutine(module_mpp_grid, mesh)
        params, costs = opt.optimize(scale_factor0, bounds=bounds, verbose=verbose)
        list_to_file(params, 'iteration-'+str(t+1)+'-param.txt')        
        list_to_file(costs.reshape(-1,1), 'iteration-'+str(t+1)+'-costs.txt')        

        mpp_grid = module_to_gid_dictionary(module_mpp_grid)
        fn = 'iteration-'+str(t+1)+'-'+output_path
        shutil.copyfile('dentate_h5types.h5', fn)
        append_cell_attributes('iteration-'+str(t+1)+'-'+output_path, 'MPP', mpp_grid, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)

def main_hardcoded(comm, output_path, cells, sf_fn='optimal_sf.txt'):
    scale_factors = []
    f = open(sf_fn, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        scale_factors.append(int(line))
    f.close()
    print(scale_factors)

    mpp_module_grid = gid_to_module_dictionary(cells)
    xp, yp = generate_mesh(scale_factor=1.0) 
    cost = cost_func(scale_factors, mpp_module_grid, (xp, yp))
    mpp_grid = module_to_gid_dictionary(mpp_module_grid)
    append_cell_attributes(output_path, 'MPP', mpp_grid, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)

def list_to_file(data, fn):
    f = open(fn, 'w')
    N, D = data.shape
    for n in range(N):
        for d in range(D):
            f.write(str(data[n,d]) + '\t')
        f.write('\n')
    f.close()

def neuroh5_test(comm, output_file='test.h5'):
    cell_corpus = Cell_Population(comm, jitter_orientation=True, jitter_spacing=True)
    cell_corpus.full_init()
    mpp_grid = cell_corpus.mpp_grid
    keys = mpp_grid.keys()
    tic = time.time()
    print('Appending to h5 file')
    append_cell_attributes(output_file, 'MPP', mpp_grid, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
    elapsed = time.time() - tic
    print('Append complete in %f seconds' % elapsed)

def gid_to_module_dictionary(cells):
    mod = {k:[] for k in np.arange(nmodules)}
    for key in cells.keys():
        cell = cells[key]
        curr_mod = cell['Module'][0]
        mod[curr_mod].append(cell)
    return mod

def module_to_gid_dictionary(module_cells):
    gid_dictionary = {}
    for mod in module_cells.keys():
        cells = module_cells[mod] 
        for cell in cells:
            gid = cell['gid'][0]
            gid_dictionary[gid] = cell
    return gid_dictionary
if __name__ == '__main__':

    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv)+1):])
    sys.exit(1)    

    MPP_info, LPP_info, xp, yp = init_generate_populations(gen_rate=True)
    grid_dict_MPP, place_dict_MPP, xy_offsets_MPP, feature_types_MPP, orientation_MPP = MPP_info
    grid_dict_LPP, place_dict_LPP, xy_offsets_LPP, feature_types_LPP, orientation_LPP = LPP_info
    
    scale_factor_0 = init_scale_factor * np.ones(nmodules)
    grid_modules = init_optimize(MPP_info, xp, yp, scale_factor_0)
    #grid_dict_MPP = generate_cell_dictionary(grid_modules)
    with open('MPP/EC_grid_cells_module_MPP.pkl', 'wb') as f:
        pickle.dump(grid_modules, f)

    f = open('cost_eval.txt','w')
    for cost in cost_value:
        f.write(str(cost) + '\n')
    f.close()
    sys.exit(1)
    
  
    make_hist(grid_dict_MPP, xp, yp, population='MPP',ctype='grid',modules=[0,4,9],xoi=59,yoi=59)
    make_hist(place_dict_MPP, xp, yp, population='MPP',ctype='place',modules=[0,4,9],xoi=59,yoi=59)
    make_hist(grid_dict_LPP, xp, yp, population='LPP',ctype='grid',modules=[0,4,9],xoi=59,yoi=59)
    

    with open('MPP/EC_grid_cells_MPP.pkl', 'wb') as f:
        pickle.dump(grid_dict_MPP, f)
    with open('MPP/EC_grid_cells_LPP.pkl','wb') as f:
        pickle.dump(grid_dict_LPP, f)
    with open('LPP/EC_place_cells_MPP.pkl', 'wb') as f:
        pickle.dump(place_dict_MPP, f)
    with open('LPP/EC_place_cells_LPP.pkl','wb') as f:
        pickle.dump(place_dict_LPP, f)
    print('...done')
    #test

   #append_cell_attributes(output_file, 'MPP', grid_features_dict, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
    
