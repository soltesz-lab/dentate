import sys
import random
import numpy as np
import pyspike as spk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


inputfile = sys.argv[1]
tstop = 500

spike_trains = spk.load_spike_trains_from_txt(inputfile, edges=(0, tstop))

spike_trains_sample = random.sample(spike_trains, 5000)

#isi_profile = spk.isi_profile(spike_trains)
isi_profile = spk.isi_profile(spike_trains_sample)
x, y = isi_profile.get_plottable_data()

fig = plt.figure()

plt.plot(x, y, '--k')

fig.savefig('%s_isidist.png' % inputfile)

print("Avg ISI distance: %.8f" % isi_profile.avrg())

fig = plt.figure()

#spike_profile = spk.spike_profile(spike_trains)
spike_profile = spk.spike_profile(spike_trains_sample)
x, y = spike_profile.get_plottable_data()
plt.plot(x, y, '--k')

print("Avg SPIKE distance: %.8f" % spike_profile.avrg())

fig.savefig('%s_spikedist.png' % inputfile)
