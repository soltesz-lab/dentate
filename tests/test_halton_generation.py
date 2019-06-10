from __future__ import division

# Make sure ghalton is installed -> $ pip install ghalton 

from builtins import range
from past.utils import old_div
import numpy as np
import ghalton
import matplotlib.pyplot as plt
import time


def runtime_test(points, n=5):

    halton_times = []
    for i in range(n):
        tic = time.time()
        points_halton = generate_points(3, points)
        elapsed = time.time() - tic
        halton_times.append(elapsed)
    halton_times = np.asarray(halton_times, dtype='float32')
    halton_avg   = np.mean(halton_times)

    uniform_times = []
    for i in range(n):
        tic = time.time()
        points_uniform = np.random.uniform(low=0., high=1., size=(points,2))
        elapsed = time.time() - tic
        uniform_times.append(elapsed)
    uniform_times = np.asarray(uniform_times, dtype='float32')
    uniform_avg   = np.mean(uniform_times)

    #print('Halton took on average %f seconds to run' % halton_avg)
    #print('Uniform average took on average %f seconds to run' % uniform_avg)

    return halton_avg, uniform_avg

def generate_points(seed, npoints, dim=2):
    sequencer = ghalton.GeneralizedHalton(dim, seed)
    return np.asarray(sequencer.get(npoints), dtype='float32')


if __name__ == '__main__':

    halton_times, uniform_times = [], []
    points = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    for point in points:
        halton_time, uniform_time = runtime_test(int(point))
        halton_times.append(halton_time)
        uniform_times.append(uniform_time)

    plt.figure()
    plt.plot(points, halton_times)
    plt.plot(points, uniform_times)
    plt.yscale('log')
    plt.legend(['halton', 'uniform'])


    npoints_lst = [100, 250, 500, 1000]
    dim = 2

    fig, axes = plt.subplots(2,2)
    for (i,npoints) in enumerate(npoints_lst):
        points_halton_1 = generate_points(3, npoints, dim=dim)
        points_halton_2 = generate_points(5, npoints, dim=dim)
        points_random   = np.random.uniform(low=0., high=1., size=(npoints, dim))
 
        axes[old_div(i,2),i%2].scatter(points_halton_1[:,0], points_halton_1[:,1], c='r')
        axes[old_div(i,2),i%2].scatter(points_halton_2[:,0], points_halton_2[:,1], c='b')
        axes[old_div(i,2),i%2].scatter(points_random[:,0], points_random[:,1], c='g')
        axes[old_div(i,2),i%2].legend(['halton seed 3', 'halton seed 5', 'random uniform'])
        axes[old_div(i,2),i%2].set_title('num points: %i' % npoints)


    plt.show()

  
