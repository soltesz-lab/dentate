import itertools
from collections import defaultdict
import sys, os.path, string, math
from neuron import h
import numpy as np


## Compartment weights
## comp_coeff        = [-0.0315,9.4210];
## HIPP_long_weights   = [37.5;112.5;187.5]*linear_coeff(1) + linear_coeff(2);
## HIPP_short_weights  = [25;75;125]*linear_coeff(1) + linear_coeff(2);
## Apical_weights      = [37.5;112.5;187.5;262.5]*linear_coeff(1) + linear_coeff(2);
## Basal_weights       = [25;75;125;175]*linear_coeff(1) + linear_coeff(2);

connection_params = np.asarray([0.4201,-0.0019,150.7465,0.0255])

## Distance-dependent probability (based on Otsuka)
def connection_prob (distance):
    connection_params[0] + (connection_params[1] - connection_params[0]) / (1 + 10. ** ((connection_params[2] - distance) * connection_params[3]))

cc_params = np.asarray([0.0002,-0.0658,7.3211])
    
## Scale gap junction strength based on polynomial fit to Amitai distance-dependence
def coupling_strength(distance, cc):
    weights = cc_params[0] * distance ** 2. + cc_params[1] * distance + cc_params[2]
    weights = weights / np.max(weights)
    return cc / np.mean(weights) * weights

## Connections based on weighted distance
## selected = datasample(transpose(1:length(distance)),round(gj_prob(pre_type,post_type)*length(distance)),'Weights',prob,'Replace',false);


