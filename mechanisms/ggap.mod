: ggap.mod
: Conductance-based gap junction model

NEURON {
    POINT_PROCESS gGap
    RANGE g, i, vgap
    NONSPECIFIC_CURRENT i
}

PARAMETER { g = 1e-10 (1/megohm) }

ASSIGNED {
    v (millivolt)
    vgap (millivolt)
    i (nanoamp)
}

BREAKPOINT { i = (vgap - v)*g }
