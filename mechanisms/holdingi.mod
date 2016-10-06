TITLE Holding current

NEURON {
	POINT_PROCESS holdingi
	ELECTRODE_CURRENT i
	RANGE i, ic
}

PARAMETER {
	ic = 0	(nanoamp)
}

ASSIGNED { i	(nanoamp)}

INITIAL { i = 0}

BREAKPOINT {
	i = ic
}
