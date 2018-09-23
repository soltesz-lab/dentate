
# Make sure ghalton is installed -> $ pip install ghalton 

import numpy as np
import ghalton
import matplotlib.pyplot as plt


def generate_points(seed, npoints, dim=2):
    sequencer = ghalton.GeneralizedHalton(dim, seed)
    return np.asarray(sequencer.get(npoints), dtype='float32')

if __name__ == '__main__':
   

    npoints_lst = [100, 250, 500, 1000]
    dim = 2

    fig, axes = plt.subplots(2,2)
    for (i,npoints) in enumerate(npoints_lst):
        points_halton_1 = generate_points(3, npoints, dim=dim)
        points_halton_2 = generate_points(5, npoints, dim=dim)
        points_random   = np.random.uniform(low=0., high=1., size=(npoints, dim))
 
        axes[i/2,i%2].scatter(points_halton_1[:,0], points_halton_1[:,1], c='r')
        axes[i/2,i%2].scatter(points_halton_2[:,0], points_halton_2[:,1], c='b')
        axes[i/2,i%2].scatter(points_random[:,0], points_random[:,1], c='g')
        axes[i/2,i%2].legend(['halton seed 3', 'halton seed 5', 'random uniform'])
        axes[i/2,i%2].set_title('num points: %i' % npoints)


    plt.show()
