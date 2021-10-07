: Ca channel


NEURON {
	SUFFIX Ca_PR
	USEION ca READ eca WRITE ica
	RANGE g, gmax
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
	B = .26 (mM-cm2/mA-ms)
	F = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	gmax = 0	(S/cm2)	: maximum permeability
    }
    
ASSIGNED {
        eca     (mV)
	v		(mV)
	ica		(mA/cm2)
	g		(S/cm2)
        as
        bs
}

STATE { s }

BREAKPOINT {
    SOLVE state METHOD derivimplicit
    g = gmax*(s^2)
    ica = g*(v - eca)
}

DERIVATIVE state {	
    rates(v)
    s' = (as - s*(as + bs))
}

INITIAL {
    rates(v)
    s=0.01086703
}


PROCEDURE rates(v (mV)) { 
    
    as=1.6 / (exp(-0.072*(v-5))+1)
    bs=0.02*(v+8.9)/(exp((v+8.9)/5)-1)
    
}