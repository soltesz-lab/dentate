import os, sys
import numpy as np
import matplotlib.pyplot as plt
from nested.optimize_utils import *

def plot_rate_maps(cells, plot=False, save=True, **kwargs):

    rate_maps = [] 
    for gid in cells:
        cell = cells[gid]
        rate_maps.append(cell['Rate Map'].reshape(cell['Nx'][0], cell['Ny'][0]))
    rate_maps = np.asarray(rate_maps, dtype='float32')
    
    summed_map = np.sum(rate_maps, axis=0)
    mean_map   = np.mean(rate_maps,axis=0)
    var_map    = np.var(rate_maps, axis=0)
    images = [summed_map, mean_map, var_map]

    ctype = kwargs.get('ctype', 'grid')
    module = kwargs.get('module', 0)

    fig, axes = plt.subplots(3,2)
    for j in range(2):
        for i in range(3):
            img = axes[i,j].imshow(images[i], cmap='inferno')
            plt.colorbar(img, ax=axes[i,j])
            if j == 1:
                img.set_clim(0, np.max(images[i]))
            if i == 0:
                axes[i,j].set_title('Cells: %s. Module: %d. Summed RM' % (ctype, module))
            elif i == 1:
                axes[i,j].set_title('Cells: %s. Module: %d. Mean RM' % (ctype, module))
            else:
                axes[i,j].set_title('Cells: %s. Module: %d. Var RM' % (ctype, module))
    if save:
        plt.savefig('%s-module-%d-ratemap.png' % (ctype, module))
    if plot:
        plt.show()

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

