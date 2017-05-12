from function_lib import *
import random
from mpl_toolkits.mplot3d import Axes3D


coords_dir = '../morphologies/'
# coords_file = 'dentate_Sampled_Soma_Locations_test.h5'
coords_file = 'dentate_Sampled_Soma_Locations_test_051017.h5'

f = h5py.File(coords_dir+coords_file, 'r')

namespace = 'Interpolated Coordinates'

populations = [population for population in f['Populations'] if namespace in f['Populations'][population]]
for population in populations:
    re_positioned = []
    # U, V, L
    pop_size = len(f['Populations'][population][namespace]['U Coordinate']['value'])
    if 'Interpolation Error' in f['Populations'][population][namespace]:
        re_positioned = np.where(f['Populations'][population][namespace]['Interpolation Error']['value'][:] > 1.)[0]
    print 'population: %s, re-positioned %i out of %i' % (population, len(re_positioned), pop_size)
    indexes = random.sample(range(pop_size), min(pop_size, 5000))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(f['Populations'][population][namespace]['X Coordinate']['value'][:][indexes],
                f['Populations'][population][namespace]['Y Coordinate']['value'][:][indexes],
                f['Populations'][population][namespace]['Z Coordinate']['value'][:][indexes], c='grey', alpha=0.1)
    ax1.scatter(f['Populations'][population]['Coordinates']['X Coordinate']['value'][:][re_positioned],
                f['Populations'][population]['Coordinates']['Y Coordinate']['value'][:][re_positioned],
                f['Populations'][population]['Coordinates']['Z Coordinate']['value'][:][re_positioned], c='c', alpha=0.1)
    ax1.scatter(f['Populations'][population][namespace]['X Coordinate']['value'][:][re_positioned],
                f['Populations'][population][namespace]['Y Coordinate']['value'][:][re_positioned],
                f['Populations'][population][namespace]['Z Coordinate']['value'][:][re_positioned], c='r', alpha=0.1)
    ax1.set_title('Re-positioned '+population)
    plt.show()
    plt.close()