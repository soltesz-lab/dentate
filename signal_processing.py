import sys, math, numbers, itertools
import numpy as np
import scipy

def gaussian_kde(x, y, bin_size, **kwargs):
    """Kernel Density Estimation with Scipy"""

    data  = np.vstack([x, y])
    kde   = scipy.stats.gaussian_kde(data, **kwargs)

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    dx = int((x_max - x_min) / bin_size)
    dy = int((y_max - x_min) / bin_size)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, dx), \
                         np.linspace(y_min, y_max, dy))

    data_grid = np.vstack([xx.ravel(), yy.ravel()])
    z    = kde.evaluate(data_grid)
    
    return xx, yy, np.reshape(z, xx.shape)


