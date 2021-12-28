import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from dentate.utils import NoiseGenerator, gauss2d
    
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

npts = 1000
width = 40
gen = NoiseGenerator(n_tiles_per_dim=1, bounds=[[-100, 100],[-100, 100]], bin_size=0.1, seed=42)

    
def energy_fn(width, point, grid):

    x0, y0 = point.T
    x, y = grid

    fw = 2. * np.sqrt(2. * np.log(100.))
    return gauss2d(x=x, y=y, mx=x0, my=y0, sx=width / fw, sy=width / fw)


for i in range(50):
    p0 = gen.next()
    print(p0)
    gen.add(p0, partial(energy_fn, width))

for i in range(10000):
    p1 = gen.next()
    print(p1)
    gen.add(p1, partial(energy_fn, width))

en = gen.energy_map
plotFFT(en)
