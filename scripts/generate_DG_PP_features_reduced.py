
import sys, os, time, random
import numpy as np
import pickle

from mpi4py import MPI
import h5py
from neuroh5.io import append_cell_attributes
import dentate
from dentate.env import Env
from dentate.stimulus import generate_spatial_offsets

import logging
logging.basicConfig()

script_name = 'generate_DG_PP_features_reduced.py'
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

N_LPP = int(N_MPP * 1.10)
N_LPP_PLACE = N_LPP 
N_LPP_GRID = 0




arena_dimension = 100.
init_scale_factor = 6.0
resolution = 5.
modules = np.arange(10)

def init(population='MPP'):

    NCELLS, NPLACE, NGRID = None, None, None
    if population == 'MPP':
        NCELLS = N_MPP
    elif population == 'LPP':
        NCELLS = N_LPP

    local_random = random.Random()
    local_random.seed(0)
    feature_type_random = np.random.RandomState(0)



    grid_orientation = [local_random.uniform(0., np.pi/3.) for i in range(len(modules))]
    feature_type_probs = None
    feature_type_values = np.asarray([0, 1])
    if population == 'MPP':
        feature_type_probs = np.asarray([0.3, 0.7])
    elif population =='LPP':
        feature_type_probs = np.asarray([0.0, 1.0])
  
     
    feature_types = feature_type_random.choice(feature_type_values, p=feature_type_probs, size=(NCELLS,))

    xy_offsets,_,_ = generate_spatial_offsets(NCELLS, arena_dimension=arena_dimension, scale_factor=init_scale_factor, maxit=40)
    grid_feature_dict = {}
    place_feature_dict = {}

    for i in range(NCELLS):
        local_random.seed(i)
        feature_type = feature_types[i]
        if feature_type == 0: # grid
            this_module = local_random.choice(modules)
            this_grid_spacing = field_width(float(this_module)/float(np.max(modules)))
            feature_dict = {}
            feature_dict['Population'] = population
            feature_dict['Module'] = np.array([this_module],dtype='int32')
            feature_dict['Grid Spacing'] = np.array([this_grid_spacing], dtype='float32')
            this_grid_orientation = grid_orientation[this_module]
            feature_dict['Grid Orientation'] = np.array([this_grid_orientation], dtype='float32')
            x_offset = xy_offsets[i,0]
            y_offset = xy_offsets[i,1]
            feature_dict['X Offset'] = np.array([x_offset],dtype='float32')
            feature_dict['Y Offset'] = np.array([y_offset],dtype='float32')
            grid_feature_dict[i] = feature_dict
        elif feature_type == 1: #place
            feature_dict = {}
            feature_dict['Population'] = population
            this_module = local_random.choice(modules)
            feature_dict['Module'] = np.array([this_module],dtype='int32')
            this_field_width = field_width(local_random.random())
            feature_dict['Field Width'] = np.array([this_field_width],dtype='float32')
            x_offset = xy_offsets[i,0]
            y_offset = xy_offsets[i,1]
            feature_dict['X Offset'] = np.array([x_offset], dtype='float32')
            feature_dict['Y Offset'] = np.array([y_offset], dtype='float32')
            place_feature_dict[i] = feature_dict

    return grid_feature_dict, place_feature_dict

def dcheck(x,y,xi,yi,spacing):
    radius = spacing / 2.
    distances = np.sqrt( (x - xi) ** 2 + (y - yi) ** 2)
    mask = np.zeros(x.shape)
    mask[distances < radius] = 1
    return mask

def place_fill_map(xp, yp, spacing, orientation, xf, yf):

    nx, ny = xp.shape
    rate_map = np.zeros((nx, ny))
    rate_map = get_rate_2(xp,yp, spacing, orientation, xf, yf, ctype='place')
    #for i in range(nx):
    #    for j in range(ny):
    #        rx, ry = xp[i,j], yp[i,j]
    #        rate_map[i,j] = get_rate(rx, ry, spacing, orientation, xf, yf,ctype='place')
    return rate_map


def grid_fill_map(xp, yp, spacing, orientation, xf, yf):
    nx, ny = xp.shape
    rate_map = np.zeros((nx, ny))
    rate_map = get_rate_2(xp, yp, spacing, orientation, xf, yf, ctype='grid')
    #for i in range(nx):
    #    for j in range(ny):
    #        rx, ry = xp[i,j], yp[i,j]
    #        rate_map[i,j] = get_rate(rx, ry, spacing, orientation, xf, yf, ctype='grid')
    return rate_map

def get_rate_2(x, y, grid_spacing, orientation, x_offset, y_offset,ctype='grid'):
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


def get_rate(x, y, grid_spacing, orientation, x_offset, y_offset,ctype='grid'):
    if ctype == 'place':
        if not dcheck(x,y,x_offset, y_offset, grid_spacing):
            return 0.0
    theta_k = [np.deg2rad(-30.), np.deg2rad(30.), np.deg2rad(90.)]
    inner_sum = 0.0
    for k in range(len(theta_k)):
        inner_sum += np.cos( ((4 * np.pi) / (np.sqrt(3)*grid_spacing)) * (np.cos(theta_k[k]) * (x - x_offset) + (np.sin(theta_k[k]) * (y - y_offset))))
    return transfer(inner_sum)

def transfer(z, a=0.3, b=-1.5):
    return np.exp(a*(z-b)) - 1 

def module_map(xp, yp, grid_feature_dict, place_feature_dict, population='MPP', module=0):
    nx, ny = xp.shape
    rate_map_grid, rate_map_place = None, None
    if population == 'MPP':
        ngrid, nplace = N_MPP_GRID, N_MPP_PLACE
    elif population == 'LPP':
        ngrid, nplace = N_LPP_GRID, N_LPP_PLACE

    rate_map_grid = np.zeros((nx, ny))
    rate_map_place = np.zeros((nx, ny))

    tic = time.time()
    print('...calculating rate map for all grid and place cells...')
    count = 0
    for grid_id in grid_feature_dict.keys():
        grid_cell_info = grid_feature_dict[grid_id]
        if grid_cell_info['Module'][0] == module:
            this_grid_spacing = grid_cell_info['Grid Spacing']
            this_grid_orientation = grid_cell_info['Grid Orientation']
            x_offset, y_offset = grid_cell_info['X Offset'], grid_cell_info['Y Offset']
            cell_map = grid_fill_map(xp, yp, this_grid_spacing, this_grid_orientation, x_offset, y_offset)
            grid_cell_info['Rate Map'] = np.array(cell_map, dtype='float32')
            rate_map_grid += cell_map
            count += 1


    count = 0
    for place_id in place_feature_dict.keys():
        place_cell_info = place_feature_dict[place_id]
        if place_cell_info['Module'][0] == module:
            this_place_width = place_cell_info['Field Width']
            this_place_orientation = 0.0
            x_offset, y_offset = place_cell_info['X Offset'], place_cell_info['Y Offset']
            cell_map = place_fill_map(xp, yp, this_place_width, this_place_orientation, x_offset, y_offset)
            place_cell_info['Rate Map'] = np.array(cell_map, dtype='float32')
            rate_map_place += cell_map
            

    elapsed = time.time() - tic
    print('...that took %f seconds...' % (elapsed))
    return (rate_map_grid, rate_map_place)

def to_file(rate_map, fn, module=1):
    nx, ny = rate_map.shape
    f = open(fn, 'w')
    for i in range(nx):
        for j in range(ny):
            f.write(str(rate_map[i,j]) + '\t')
        f.write('\n')

def generate_mesh():

    mega_arena_x_bounds = [-arena_dimension * init_scale_factor/2., arena_dimension * init_scale_factor/2.]
    mega_arena_y_bounds = [-arena_dimension * init_scale_factor/2., arena_dimension * init_scale_factor/2.]
    mega_arena_x = np.arange(mega_arena_x_bounds[0], mega_arena_x_bounds[1], resolution)
    mega_arena_y = np.arange(mega_arena_y_bounds[0], mega_arena_y_bounds[1], resolution)
    return np.meshgrid(mega_arena_x, mega_arena_y, indexing='ij')

def generate_cells(gen_rate=True):
    tic = time.time()
    grid_feature_dict_MPP, place_feature_dict_MPP = init(population='MPP')
    grid_feature_dict_LPP, place_feature_dict_LPP = init(population='LPP')
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

    grid_dicts = (grid_feature_dict_MPP, grid_feature_dict_LPP)
    place_dicts = (place_feature_dict_MPP, place_feature_dict_LPP)
    return grid_dicts, place_dicts, mega_arena_xp, mega_arena_yp
   

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

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.rank
    env = Env(comm=comm,configFile=sys.argv[1])
    if io_size == -1:
        io_size = comm.size
    output_file = 'EC_grid_cells.h5'
    output_h5 = h5py.File(output_file, 'w')
    output_h5.close()
    comm.barrier()

    grid_dicts, place_dicts, xp, yp = generate_cells(gen_rate=True)
    grid_dict_MPP, grid_dict_LPP = grid_dicts
    place_dict_MPP, place_dict_LPP = place_dicts 
  
    make_hist(grid_dict_MPP, xp, yp, population='MPP',ctype='grid',modules=[0,4,9],xoi=59,yoi=59)
    make_hist(place_dict_MPP, xp, yp, population='MPP',ctype='place',modules=[0,4,9],xoi=59,yoi=59)
    make_hist(grid_dict_LPP, xp, yp, population='LPP',ctype='grid',modules=[0,4,9],xoi=59,yoi=59)
    

    #with open('MPP/EC_grid_cells_MPP.pkl', 'wb') as f:
    #    pickle.dump(grid_dict_MPP, f)
    #with open('MPP/EC_grid_cells_LPP.pkl','wb') as f:
    #    pickle.dump(grid_dict_LPP, f)
    #with open('LPP/EC_place_cells_MPP.pkl', 'wb') as f:
    #    pickle.dump(place_dict_MPP, f)
    #with open('LPP/EC_place_cells_LPP.pkl','wb') as f:
    #    pickle.dump(place_dict_LPP, f)
    print('...done')


    #append_cell_attributes(output_file, 'MPP', grid_features_dict, namespace='Grid Input Features', comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
    
    
 
     
    
