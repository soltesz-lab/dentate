import itertools
from collections import defaultdict
import sys, os.path, string, math
from neuron import h
import numpy as np
import utils, cells


## Compartment weights
## linear_coeff        = [-0.0315,9.4210];
## HIPP_long_weights   = [37.5;112.5;187.5]*linear_coeff(1) + linear_coeff(2);
## HIPP_short_weights  = [25;75;125]*linear_coeff(1) + linear_coeff(2);
## Apical_weights      = [37.5;112.5;187.5;262.5]*linear_coeff(1) + linear_coeff(2);
## Basal_weights       = [25;75;125;175]*linear_coeff(1) + linear_coeff(2);




## Set distance-dependent probability (based on Otsuka)
def connection_prob (param, distance):
    param[0] + (param[1] - param[0]) / (1 + 10. ^ ((param[2] - distance) * param[3])

## Select connections based on weighted distance
## selected = datasample(transpose(1:length(distance)),round(gj_prob(pre_type,post_type)*length(distance)),'Weights',prob,'Replace',false);

## Scale gap junction strength based on polynomial fit to Amatai distance-dependence
def coupling_strength(params, distance, cc):
    weights = params[0] * distance ^ 2 + params[1] * distance + params[2]
    weights = weights / np.max(weights)
    return cc / np.mean(weights) * weights

