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

def plot_xy_offsets(cells, plot=False, save=True, **kwargs):

    pop_xy_offsets = []
    for gid in cells:
        cell = cells[gid]
        if 'X Offset Scaled' not in cell or 'Y Offset Scaled' not in cell:
            offsets = zip(cell['X Offset'], cell['Y Offset'])
        else:
            offsets = zip(cell['X Offset Scaled'], cell['Y Offset Scaled'])
        for (x_offset, y_offset) in offsets:
            pop_xy_offsets.append((x_offset, y_offset))
    pop_xy_offsets = np.asarray(pop_xy_offsets, dtype='float32')

    ctype  = kwargs.get('ctype', 'grid')
    module = kwargs.get('module', 0)

    plt.figure()
    plt.scatter(pop_xy_offsets[:,0], pop_xy_offsets[:,1])
    plt.title('Cells: %s. Module: %d. xy-offsets' % (ctype, module))
    
    if save:
        plt.savefig('%s-module-%d-xyoffsets.png' % (ctype, module))
    if plot:
        plt.show()


def plot_fraction_active_map(cells, func_name, plot=False,save=True, **kwargs):
    m = sys.modules['__main__']
    if hasattr(m, func_name):
        f = getattr(m, func_name)
    else:
        raise Exception('Function could not be found')

    fraction_active = f(cells)
    fraction_active_img = np.zeros((20,20))
    for (i,j) in fraction_active:
        fraction_active_img[i,j] = fraction_active[(i,j)]

    ctype  = kwargs.get('ctype', 'grid')
    module = kwargs.get('module',0)
    target = kwargs.get('target', 0.15)
   
    fig, axes = plt.subplots(1,2)

    img = axes[0].imshow(fraction_active_img, cmap='inferno')
    plt.colorbar(img, ax=axes[0])
    #img.set_clim(0, 1.0)
    axes[0].set_title('Fraction Active')

    img = axes[1].imshow(fraction_active_img - target, cmap='inferno')
    plt.colorbar(img, ax=axes[1])
    #img.set_clim(0, 1.0)
    axes[1].set_title('Fraction active distance from target')
 
    if save:
        plt.savefig('%s-module-%d-fractionactive.png' % (ctype, module))
    if plot:
        plt.show()

def plot_rate_histogram(cells, plot=False, save=True, **kwargs):
    ctype  = kwargs.get('ctype', 'grid')
    module = kwargs.get('module', 0)
    peak_firing_rate = kwargs.get('peak firing rate', 20.0)
    bins = kwargs.get('nbins', 40)

    rate_maps = []
    for gid in cells:
        cell = cells[gid]
        nx, ny = cell['Nx'][0], cell['Ny'][0]
        rate_maps.append(cell['Rate Map'].reshape(nx, ny))
    rate_maps = np.asarray(rate_maps, dtype='float32')

    N, nx, ny = rate_maps.shape
    weights = np.ones(N) / float(N)
    hists, edges_list = [], []
    for i in xrange(nx):
        for j in xrange(ny):
            hist, edges = np.histogram(rate_maps[:,i,j], bins=bins, range=(0.0, peak_firing_rate), weights=weights)
            hists.append(hist)
            edges_list.append(edges)
    hists = np.asarray(hists)
    hist_mean = np.mean(hists, axis=0)
    hist_std = np.sqrt(np.var(hists, axis=0))

    fig, axes = plt.subplots(2, 1)
    axes[0].bar(edges_list[0][1:], hist_mean, alpha=0.5, log=False, yerr=hist_std)
    axes[0].set_title('Firing rate histogram')
    axes[0].set_ylabel('Probability')

    axes[1].bar(edges_list[0][1:], hist_mean, alpha=0.5, log=True, yerr=hist_std)
    axes[1].set_ylabel('Log Probability')
    axes[1].set_xlabel('Firing rate (Hz')
 
    if save:
        plt.savefig('%s-module-%d-rate-histogram.png' % (ctype, module))
    if plot:
        plt.show()

    

def plot_population_storage(file_path):

    try:
        storage = PopulationStorage(file_path=file_path)
    except:
        raise Exception('Error occured when loading PopulationStorage object')
    storage.plot()


if __name__ == '__main__':
    file_path = str(sys.argv[1])
    plot_population_storage(file_path)


