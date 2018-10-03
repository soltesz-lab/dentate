
import sys, gc, os, math
import click
import dentate
from dentate import utils, plot
import numpy as np

import matplotlib.pyplot as plt


@click.command()
@click.option("--destination", '-d', type=str)
@click.option("--source", '-s', type=str, multiple=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(destination, source, font_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    for s in source:
            
        dist_u_histoCount = np.loadtxt('%s Distance U Bin Count.dat' % s)                       
        dist_u_bin_edges = np.loadtxt('%s Distance U Bin Edges.dat' % s)                        
        dist_v_histoCount = np.loadtxt('%s Distance V Bin Count.dat' % s)                       
        dist_v_bin_edges = np.loadtxt('%s Distance V Bin Edges.dat' % s)                        
        dist_histoCount = np.loadtxt('%s Distance Bin Count.dat' % s)                           
        dist_bin_edges = np.loadtxt('%s Distance Bin Edges.dat' % s)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
        fig.suptitle('Distribution of connection distances for projection %s -> %s' % (s, destination), fontsize=font_size)
        
        bin_size = 10.
        ax1.bar(dist_bin_edges, dist_histoCount, width=bin_size)
        ax1.set_xlabel('Total distance (um)', fontsize=font_size)
        ax1.set_ylabel('Number of connections', fontsize=font_size)
        
        ax2.bar(dist_u_bin_edges, dist_u_histoCount, width=bin_size)
        ax2.set_xlabel('Septal - temporal (um)', fontsize=font_size)
        
        ax3.bar(dist_v_bin_edges, dist_v_histoCount, width=bin_size)
        ax3.set_xlabel('Supra - infrapyramidal (um)', fontsize=font_size)

        filename = 'Connection distance %s to %s.png' % (s, destination)
        plt.savefig(filename)

    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])

