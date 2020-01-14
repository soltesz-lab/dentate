## Reads an SWC morphology file and instantiates a hoc cell with it

import sys
from neuron import h, gui

h.load_file("nrngui.hoc")
h.load_file("import3d.hoc")

h('objref nil')

def import_swc_cell(path):
    
    morph = h.Import3d_SWC_read()
    morph.input(path)
    
    i3d = h.Import3d_GUI(morph, 0)
    i3d.instantiate(h.nil)
    
    return i3d


cell = import_swc_cell(sys.argv[1])
h.topology()

