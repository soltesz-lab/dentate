import os, sys
import numpy as np
import h5py
from neuroh5.io import read_cell_attributes
from generate_DG_PP_features_reduced_h5support import generate_mesh
import matplotlib.pyplot as plt

grid_fn = 'grid-MPP.h5'
place_fn = 'place-MPP.h5'
module_fn = 'optimal_sf.txt'
grid_namespace = 'Grid Input Features'
place_namespace = 'Place Input Features'


def read_module_fn(fn):
    f = open(fn, 'r')
    module_size = []
    for line in f.readlines():
        line = line.strip('\n')
        module_size.append(int(line))
    f.close()
    return module_size

def h5_to_dict(fn, namespace, population='MPP'):
    return {gid: cell_attr for (gid, cell_attr) in read_cell_attributes(fn, population, namespace=namespace)}

def visualize_xy_offsets(cells, modules):
    
    gids = list(cells.keys())
    #module_xy_offsets = {k: [] for k in module_meshes.keys()}

    ranks = set()
    for gid in gids:
        cell = cells[gid]
        rank = cell['Rank'][0]
        ranks.add(rank)
    print((len(gids)))
    print(ranks)
    module_xy_offsets = {module: {r: [] for r in ranks} for module in modules}
    for gid in gids:
        cell = cells[gid]
        rank = cell['Rank'][0]
        module = cell['Module'][0]
        xy_offsets = list(zip(cell['X Offset Scaled'], cell['Y Offset Scaled']))
        for (x, y) in xy_offsets:
            module_xy_offsets[module][rank].append((x,y))
    
    tlen = 0
    for rank in ranks:
        #plt.figure()
        count = 1
        for module in list(module_xy_offsets.keys()):
            xy_offsets = np.asarray(module_xy_offsets[module][rank], dtype='float32')
            print((rank, module, len(xy_offsets)))
            tlen += len(xy_offsets)
            #plt.subplot(2,5,count)
            #plt.scatter(xy_offsets[:,0], xy_offsets[:,1])
            #plt.title('Module %d' % (module+1))
            count += 1
    print(tlen)
def visualize_rate_maps(cells, modules):
    module_rate_maps = {k: [] for k in modules}
    gids = list(cells.keys())
    for gid in gids:
        cell = cells[gid]
        module = cell['Module'][0]
        nx, ny = cell['Nx'][0], cell['Ny'][0]
        rate_map = cell['Rate Map'].reshape(nx, ny)
        module_rate_maps[module].append(rate_map)

    summed_module_map = {k: None for k in modules}
    mean_module_map = {k: None for k in modules}
    var_module_map = {k: None for k in modules}
    for module in list(module_rate_maps.keys()):
        maps = np.asarray(module_rate_maps[module], dtype='float32')
        summed_map = np.sum(maps,axis=0)
        mean_map = np.mean(maps,axis=0)
        var_map = np.var(maps,axis=0)
        summed_module_map[module] = summed_map
        mean_module_map[module] = mean_map
        var_module_map[module] = var_map
    im_plot(summed_module_map, 'sum')
    im_plot(mean_module_map, 'mean')
    im_plot(var_module_map, 'var')

def im_plot(module_maps, metric):
    plt.figure()
    count = 1
    for module in list(module_maps.keys()):
        module_map = module_maps[module]
        plt.subplot(2,5,count)
        plt.imshow(module_map, cmap='inferno')
        plt.colorbar()
        plt.clim(0, np.max(module_map))
        plt.title('%s-module %d' % (metric,(module+1)))
        count += 1

if __name__ == '__main__':
    module_size = read_module_fn(module_fn)
    module_meshes = {}
    for i in range(len(module_size)):
        xp, yp = generate_mesh(scale_factor=module_size[i])
        mesh = np.dstack((xp, yp))
        xs, ys, _ = mesh.shape
        module_meshes[i] = (mesh, (xs, ys), module_size[i])
        
    grid_cells = h5_to_dict(grid_fn, grid_namespace)
    place_cells = h5_to_dict(place_fn, place_namespace)
    visualize_xy_offsets(grid_cells, list(module_meshes.keys()))
    #visualize_rate_maps(grid_cells, module_meshes.keys())
    plt.show()

