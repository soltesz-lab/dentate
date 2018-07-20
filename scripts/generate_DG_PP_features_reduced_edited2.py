import sys, os, time, random
import numpy as np
import pickle
from copy import deepcopy

from scipy.optimize import minimize
from scipy.optimize import basinhopping


from mpi4py import MPI
import h5py
from neuroh5.io import append_cell_attributes
import dentate
from dentate.env import Env
from dentate.stimulus import generate_spatial_offsets

import logging
logging.basicConfig()

script_name = 'generate_DG_PP_features_reduced_edited2.py'
logger = logging.getLogger(script_name)

io_size=-1
chunk_size=1000
value_chunk_size=1000


field_width_params = [35.0, 0.32]
field_width = lambda x : 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
max_field_width = field_width(1.)

feature_grid = 0
feature_place_field = 1

N_MPP = 2500
N_MPP_GRID = int(N_MPP * 0.7)
N_MPP_PLACE = N_MPP - N_MPP_GRID

N_LPP = 1300 #int(N_MPP * 1.10)
N_LPP_PLACE = N_LPP 
N_LPP_GRID = 0



cost_value = []
arena_dimension = 100.
init_scale_factor = 10.0
init_orientation_jitter = [-10., 10.] #[np.deg2rad(-10.), np.deg2rad(10.)]
init_lambda_jitter = [-10., 10.]
resolution = 5.
nmodules = 10
modules = np.arange(nmodules)


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
    evaluations = np.asarray([0.0 for _ in np.arange(nmodules)],dtype='float32')
    for mod in module_cells.keys():
        cells = module_cells[mod]
        module_rate_map = None
        for (c, cell) in enumerate(cells):
            rate_map = cell['Rate Map Box']
            if c == 0:
                nx, ny = rate_map.shape
                module_rate_map = np.zeros((nx, ny))
            module_rate_map += rate_map
        evaluations[mod] = float(np.max(module_rate_map)) / float(np.min(module_rate_map))
    return evaluations - 1.

def fraction_active(module_cells, modules=modules, target=0.3):
    rates = {mod:[] for mod in modules}
    for mod in module_cells.keys():
        cells = module_cells[mod]
        for cell in cells:
            rates[mod].append(cell['Rate Map Box'])
   
    nx = 1 
    frac_active_dict = {(i,i): {k:None for k in modules} for i in range(nx)}
    diagonal_positions = [ (i,i) for (i,i) in frac_active_dict.keys()]
    for (px, py) in diagonal_positions:
        for key in rates.keys():
            module_maps = np.asarray(rates[key])
            position_rates = module_maps[:,px,py]
            frac_active = calculate_fraction_active(position_rates)
            frac_active_dict[(px,py)][key] = frac_active
    target_fraction_active = {(i,i): {k: target for k in modules} for (i,i) in frac_active_dict.keys()}

    diff_fraction_active = {(i,i): {k: np.abs(target_fraction_active[(i,i)][k]-frac_active_dict[(i,i)][k]) for k in modules} for (i,i) in frac_active_dict.keys()}
    
    module_error = np.array([ 0. for _ in range(len(modules))])
    for (i,i) in diff_fraction_active.keys():
        pos_errors = diff_fraction_active[(i,i)]
        for module in pos_errors.keys():
            mod_e = pos_errors[module]
            module_error[module] += mod_e
    return module_error


def calculate_fraction_active(rates, threshold=0.1):
    max_rate = np.max(rates)
    normalized_rates = np.divide(rates, max_rate)
    num_active = len(np.where(normalized_rates > threshold)[0])
    fraction_active = float(num_active) / len(normalized_rates)
    return fraction_active               
            


def cost_func(x, cell_modules):
    sf = x
    for mod in cell_modules.keys():
        xp, yp = generate_mesh(scale_factor=sf[mod])
        cells = cell_modules[mod]
        for (c,cell) in enumerate(cells):
            orientation, spacing = cell['Jittered Grid Orientation'], cell['Jittered Grid Spacing']
            xf, yf = cell['X Offset Reduced'], cell['Y Offset Reduced']
            xf_scaled, yf_scaled = xf * sf[mod], yf * sf[mod]
            cell['X Offset'], cell['Y Offset'] = xf_scaled, yf_scaled
            rate_map = grid_fill_map(xp, yp, spacing, orientation, xf_scaled, yf_scaled)
            nx, ny = rate_map.shape
            cell['Rate Map'] = rate_map
            cell['Rate Map Box'] = rate_map[int(nx/2)-10:int(nx/2)+10,int(ny/2)-10:int(ny/2)+10]
            #if mod == 0 and c == 0: 
            #    print(cell['Rate Map Box'])

    peak_trough_evaluation = peak_to_trough(cell_modules, modules=modules)
    fraction_active_evaluation = fraction_active(cell_modules, modules=modules)

    peak_cost = np.sum(peak_trough_evaluation)
    frac_active_cost = np.sum(fraction_active_evaluation)
    cost_value.append(0.5*(peak_cost ** 2 + frac_active_cost ** 2))
    return 0.5 * (peak_cost ** 2 + frac_active_cost ** 2)#[peak_cost, frac_active_cost]

class OptimizationRoutine(object):
    def __init__(self, cells):
        self.cells = cells

    def optimize(self, x0, bounds=None, verbose=False):
        if bounds is None:
            bounds = [(1., 50.) for _ in x0]
        fnc = lambda x: cost_func(x, self.cells)
        minimizer_kwargs = dict(method='L-BFGS-B', bounds=bounds, options={'disp':True,'eps':1.0, 'maxiter':10})
        bh_output = basinhopping(fnc, x0, minimizer_kwargs=minimizer_kwargs, stepsize=10.0, T=2.0,disp=True)

        if verbose:
            print(x0)
            print(bh_output.x)
            print(fnc(x0))
            print(fnc(bh_output.x))

class Initialization(object):
    def __init__(self, jitter_orientation=True, jitter_spacing=True):
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

    def full_init(self):
        self.initialize_cells(population='MPP')
        self.initialize_cells(population='LPP')
        self.generate_xy_offsets()
        #self.calculate_rate_maps(jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing)

    def initialize_cells(self, population='MPP'):
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
            feature_type = feature_types[i]
            if feature_type == 0: # Grid cell
                this_module = self.local_random.choice(modules)
                orientation = grid_orientation[this_module]
                spacing = field_width(float(this_module)/float(np.max(modules)))
                grid_feature_dict[i] = self._build_grid_cell(pop, orientation, spacing, this_module)
            elif feature_type == 1:
                this_module = self.local_random.choice(modules)
                cell_field_width = field_width(self.local_random.random())
                place_feature_dict[i] = self._build_place_cell(pop, cell_field_width, this_module)
        return grid_feature_dict, place_feature_dict

    def _build_place_cell(self, pop, cell_field_width, module):
        cell = {}
        cell['Population'] = pop
        cell['Module'] = np.array([module], dtype='int32')
        cell['Field Width'] = np.array([cell_field_width], dtype='float32')
        return cell

    def _build_grid_cell(self, pop, orientation, spacing, module):
        cell = {}
        cell['Population'] = pop
        cell['Module'] = np.array([module],dtype='int32')
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

    def calculate_rate_maps(self, scale_factors):
        if self.mpp_grid is not None:
            self._calculate_rate_maps(self.mpp_grid, scale_factors, cell_type='grid', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing)
        if self.mpp_place is not None:
            self._calculate_rate_maps(self.mpp_place, scale_factors, cell_type='place', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing)
        if self.lpp_grid is not None:
            self._calculate_rate_maps(self.lpp_grid, scale_factors, cell_type='grid', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing)
        if self.lpp_place is not None:
            self._calculate_rate_maps(self.lpp_place, scale_factors, cell_type='place', jittered_orientation=self.jitter_orientation, jittered_spacing=self.jitter_spacing)

    def _calculate_rate_maps(self,cells, scale_factors, cell_type='grid', jittered_orientation=False, jittered_spacing=False):
        for key in cells.keys():
            cell = cells[key]
            x_offset, y_offset = None, None
            this_module = cell['Module'][0]
            x_offset_scaled = cell['X Offset'][0] * scale_factors[this_module]
            y_offset_scaled = cell['Y Offset'][0] * scale_factors[this_module]
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
                cell['Rate Map'] = grid_fill_map(self.xp, self.yp, grid_spacing, grid_orientation, x_offset_scaled, y_offset_scaled)
            elif cell_type == 'place':
                place_orientation = 0.0
                place_width = cell['Field Width'][0]
                cell['Rate Map'] = place_fill_map(self.xp, self.yp, place_width, place_orientation, x_offset_scaled, y_offset_scaled)

def main(init_scale_factor):
    tic = time.time()
    cell_corpus = Initialization(jitter_orientation=True, jitter_spacing=True)
    cell_corpus.full_init()
    elapsed = time.time() - tic
    print('%d cells generated in %f seconds' % ((N_LPP+N_MPP), elapsed))

    T = init_scale_factor.shape[1]
    generated_cells = {}
    for t in range(T):
        corpus_copy = deepcopy(cell_corpus)
        scale_factor0 = init_scale_factor[:,t]
        tic = time.time()
        corpus_copy.calculate_rate_maps(scale_factor0)
        elapsed = time.time() - tic
        print('Rate maps for %d cells calculated in %f seconds' % (len(corpus_copy.mpp_grid.keys()), elapsed))
        module_mpp_grid = generate_module_dictionary(corpus_copy.mpp_grid)
        opt = OptimizationRoutine(module_mpp_grid)
        opt.optimize(scale_factor0, bounds=None, verbose=True)
        generated_cells[t] = module_mpp_grid
        
       

def generate_module_dictionary(cells):
    mod = {k:[] for k in np.arange(nmodules)}
    for key in cells.keys():
        cell = cells[key]
        curr_mod = cell['Module'][0]
        mod[curr_mod].append(cell)
    return mod

if __name__ == '__main__':

    #comm = MPI.COMM_WORLD
    #rank = comm.rank
    #env = Env(comm=comm,configFile=sys.argv[1])
    #if io_size == -1:
    #    io_size = comm.size
    #output_file = 'EC_grid_cells.h5'
    #output_h5 = h5py.File(output_file, 'w')
    #output_h5.close()
    #comm.barrier

    low, high = 1, 51
    nrandom_start = int(sys.argv[1])
    init_scale_factor = np.random.randint(low,high,(nmodules,nrandom_start))
    main(init_scale_factor)
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


    #append_cell_attributes(output_file, 'MPP', grid_features_dict, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
    
