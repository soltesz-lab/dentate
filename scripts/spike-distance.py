import sys
import numpy as np
import pyspike as spk
import matplotlib.pyplot as plt

inputfile = sys.argv[1]
tstop = 500

spike_trains = spk.load_spike_trains_from_txt(inputfile, edges=(0, tstop), is_sorted=False)

isi_profile = spk.isi_profile(spike_trains)
x, y = isi_profile.get_plottable_data()

plt.plot(x, y, '--k')

print("Avg ISI distance: %.8f" % isi_profile.avrg())

plt.show()
