import sys, logging
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from dentate.utils import MPINoiseGenerator, gauss2d
from mpi4py import MPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_mpi_noisegen')

def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


def plotFFT(pattern, fft_vmax=1):
  fig, axs = plt.subplots(1, 3, figsize=(15, 10))
  im = axs[0].imshow(pattern)
  fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
  axs[1].set_title('Periodogram')
  im = axs[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(pattern - np.mean(pattern))/ pattern.shape[0])), vmax=fft_vmax)
  fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
  axs[2].set_title('Log periodogram')
  eps = 1e-12
  im = axs[2].imshow(np.log10(abs(np.fft.fftshift(np.fft.fft2(pattern - np.mean(pattern))/ pattern.shape[0]))+eps), vmax=fft_vmax)
  fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
  plt.show()

comm = MPI.COMM_WORLD
rank = comm.rank

width = 40
gen = MPINoiseGenerator(comm=comm, bounds=[[-100, 100],[-100, 100]], mask_fraction=0.99, bin_size=0.05, seed=42)
    
def energy_fn(point, grid, width):

    x0, y0 = point.T
    x, y = grid

    fw = 2. * np.sqrt(2. * np.log(100.))
    return gauss2d(x=x, y=y, mx=x0, my=y0, sx=width / fw, sy=width / fw)


for i in range(50):
    p0 = gen.next()
    logger.info(f'Rank {rank}: {p0}')
    gen.add(p0, energy_fn, energy_kwargs={'width': width})

en = gen.energy_map
if comm.rank == 0:
    plotFFT(en)

for i in range(200):
    p1 = gen.next()
    logger.info(f'Rank {rank}: {p1}')
    gen.add(p1, energy_fn, energy_kwargs={'width': width})

        

en = gen.energy_map
if comm.rank == 0:
    plotFFT(en)

