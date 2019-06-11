
from builtins import str
from builtins import range
import sys
import numpy as np
import matplotlib.pyplot as plt


nmodules = 10
modules = np.arange(nmodules)


def read_ratemap(fn,population='MPP'):
    fn = population + '/' + fn
    rate_map = []
    f = open(fn, 'r')
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        curr_rates = []
        for val in line[0:-1]:
            curr_rates.append(float(val))
        rate_map.append(curr_rates)
    return np.asarray(rate_map)


def read_maps(population='MPP'):

    MPP_place_maps, MPP_grid_maps = [], []
    for module in modules:
        fn_place = 'ratemap-module-'+str(module)+'-'+population+'-place.txt'
        MPP_place_maps.append(read_ratemap(fn_place,population=population))
        fn_grid = 'ratemap-module-'+str(module)+'-'+population+'-grid.txt'
        MPP_grid_maps.append(read_ratemap(fn_grid,population=population))

    MPP_place_sum = np.zeros(MPP_place_maps[0].shape)
    MPP_grid_sum = np.zeros(MPP_place_maps[0].shape)
    MPP_total_sum = np.zeros(MPP_place_maps[0].shape)
    for i in range(len(MPP_place_maps)):
        MPP_place_sum += MPP_place_maps[i]
        MPP_grid_sum += MPP_grid_maps[i]
    MPP_total_sum = MPP_place_sum + MPP_grid_sum
    return MPP_place_maps, MPP_grid_maps, MPP_place_sum, MPP_grid_sum, MPP_total_sum

def show_img(x, title):
    plt.figure()
    plt.imshow(x,cmap='inferno')
    plt.colorbar()
    plt.title(title)


_, _, MPP_place_sum, MPP_grid_sum, MPP_total_sum = read_maps(population='MPP')
_, _, LPP_place_sum, LPP_grid_sum, LPP_total_sum = read_maps(population='LPP')

MPP_grid_module0 = read_ratemap('grid-module-0-rates-x-59-y-59.txt',population='MPP')[0] #(1,N)
MPP_grid_module4 = read_ratemap('grid-module-4-rates-x-59-y-59.txt',population='MPP')[0]
MPP_grid_module9 = read_ratemap('grid-module-9-rates-x-59-y-59.txt',population='MPP')[0]


nbins=50
plt.figure()
plt.hist(MPP_grid_module0,bins=nbins,color='r',alpha=0.8)
plt.hist(MPP_grid_module4,bins=nbins,color='y',alpha=0.8)
plt.hist(MPP_grid_module9,bins=nbins,color='b',alpha=0.8)
plt.legend(['Module 0', 'Module 4', 'Module 9'])
plt.title('MPP grid rate histogram')


MPP_place_module0 = read_ratemap('place-module-0-rates-x-59-y-59.txt',population='MPP')[0] #(1,N)
MPP_place_module4 = read_ratemap('place-module-4-rates-x-59-y-59.txt',population='MPP')[0]
MPP_place_module9 = read_ratemap('place-module-9-rates-x-59-y-59.txt',population='MPP')[0]


plt.figure()
plt.hist(MPP_place_module0,bins=nbins,color='r',alpha=0.8)
plt.hist(MPP_place_module4,bins=nbins,color='y',alpha=0.8)
plt.hist(MPP_place_module9,bins=nbins,color='b',alpha=0.8)
plt.legend(['Module 0', 'Module 4', 'Module 9'])
plt.title('MPP place rate histogram')



show_img(MPP_place_sum,'MPP place')
show_img(MPP_grid_sum, 'MPP grid')
show_img(MPP_total_sum, 'MPP total')

show_img(LPP_place_sum, 'LPP place')
show_img(LPP_grid_sum, 'LPP grid')
show_img(LPP_total_sum, 'LPP total')


plt.show()



