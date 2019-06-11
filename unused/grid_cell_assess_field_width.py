
"""
    Given transfer function f(x) = e^(a * (x - b)) - 1., this script finds the optimal value 'a'
    such that 2r/lambda ~ 2/3 where lambda is the inter-vertex spacing and r is the radius at which
    firing rate is 10% of peak firing.

    It was found that optimal 'a' for modules are
    {0-1: 0.7, 2-4: 0.65, 5-9: 0.6}
"""
__author__ = 'Darian Hadjiabadi'

import sys, os, click
import numpy as np
import matplotlib.pyplot as plt

import dentate
from dentate.stimulus import generate_spatial_ratemap, linearize_trajectory
from dentate.utils import *
from scipy.spatial.distance import euclidean


def transfer(x,a,b=-1.5):
    return np.exp(a * (x - b)) - 1.

def find_radius_decay(rate_map, xp, yp, xstart, ystart, threshold=0.1):
    response_max = rate_map[xstart, ystart]
    cutoff = None
    for i in range(xstart, xp.shape[0]):
        position_response = rate_map[i, ystart]
        if position_response < threshold * response_max:
            cutoff = i - xstart
            break

    hop_size = euclidean( (xp[xstart, xstart], yp[ystart, ystart]), (xp[xstart+1, xstart], yp[ystart, ystart]) )
    radius   = cutoff * hop_size
    return cutoff, radius

def calculate_lambda(module):
    nmodules    = 10
    modules     = np.arange(nmodules)
    field_width = lambda x: 40. + 35.0 * (np.exp(x / 0.32) - 1.)
    return field_width(float(module) / np.max(modules)) 
    

def get_diagonals(xp, yp):
    return np.diagonal(xp), np.diagonal(yp)


@click.command()
@click.option("--module", type=int, default=0)
@click.option("--scale-factor", type=float, default=1.0)
@click.option("--resolution", type=float, default=1.0)
@click.option("--arena-dimension", type=float, default=100.)
@click.option("--grid-peak-rate", type=float, default=20.)
def main(module, scale_factor, resolution, arena_dimension, grid_peak_rate):

    intervertex_spacing = calculate_lambda(module)
    print('lambda: %0.3f' % intervertex_spacing)
    xp, yp = generate_mesh(scale_factor=scale_factor, arena_dimension=arena_dimension, \
                           resolution=resolution)
    avals  = np.arange(0.1, 1.0, 0.1)

    mock_grid_cell = {}
    x_offset = xp[old_div(xp.shape[0],2), old_div(xp.shape[1],2)]
    y_offset = yp[old_div(yp.shape[0],2), old_div(yp.shape[1],2)]
    mock_grid_cell['X Offset'] = np.array([x_offset], dtype='float32')
    mock_grid_cell['Y Offset'] = np.array([y_offset], dtype='float32')
    mock_grid_cell['Grid Orientation'] = np.array([np.pi/4.], dtype='float32')
    mock_grid_cell['Grid Spacing']     = np.array([intervertex_spacing], dtype='float32') # module 0 cell

    xd, yd = get_diagonals(xp, yp)
    linearized = linearize_trajectory(xd.reshape(-1,1), yd.reshape(-1,1))
    linearized_abridged = linearized[0:old_div(len(linearized),2),0]

    information = []
    for a in avals:
        kwargs   = {'a': a, 'b': -1.5}
        rate_map = generate_spatial_ratemap(0, mock_grid_cell, None, xp, yp, grid_peak_rate, \
                                        0.0, ramp_up_period=None, **kwargs)
        print('a: %0.2f. rate map min: %0.3f' % (a, np.min(rate_map)))
        _, radius_cutoff = find_radius_decay(rate_map, xp, yp, old_div(xp.shape[0],2), old_div(yp.shape[0],2), threshold=0.10)
        information.append((a, radius_cutoff, rate_map))       

    sz = int(np.ceil(np.sqrt(avals.shape[0])))
    fig, axes = plt.subplots(sz, sz)
    for i in range(len(information)):
        a, rstart, response = information[i]
        decay_ratio   = float(rstart) / intervertex_spacing
        print('%0.2f. radius: %f. diameter: %f' %(a,decay_ratio, 2.*decay_ratio))
        img = axes[i%sz,old_div(i,sz)].imshow(response, cmap='viridis')
        plt.colorbar(img, ax=axes[i%sz, old_div(i,sz)])
        axes[i%sz, old_div(i,sz)].set_title('a: %0.2f. r*/lambda: %0.3f' % (a, decay_ratio))

    fig2, axes2 = plt.subplots(sz, sz)
    for i in range(len(information)):
        a, _, response = information[i]
        diagonal_response = np.fliplr(response).diagonal()
        curr_plot = axes2[i%sz, old_div(i,sz)].plot(linearized_abridged, diagonal_response)    
        axes2[i%sz, old_div(i,sz)].set_title('Diagonal trajectory a: %0.2f' % a)

    plt.show()

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):])

    
