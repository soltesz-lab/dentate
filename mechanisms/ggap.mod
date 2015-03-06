: ggap.mod
: Conductance-based gap junction model

NEURON {
    POINT_PROCESS ggap
    RANGE g, i, vgap
    NONSPECIFIC_CURRENT i
}

PARAMETER { g = 1e-10 (microsiemens) }

ASSIGNED {
    v (millivolt)
    vgap (millivolt)
    i (nanoamp)
}

INITIAL { i = (vgap - v)*g }

BREAKPOINT { i = (vgap - v)*g }
