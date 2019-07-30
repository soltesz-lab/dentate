
import os
import sys

import numpy as np
import scipy.integrate as integrate

import quadpy

if __name__ == '__main__':
    valquad, quad_error = quadpy.line_segment.integrate_adaptive(lambda x: x * np.sin(5 * x), [0.0, np.pi], 1.49e-08)

    valsci, sci_error = integrate.quad(lambda x: x * np.sin(5 * x), 0.0, np.pi)
    

    print('Quadpy value: %0.3f' % valquad)
    print('Scipy value: %0.3f' % valsci)
