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
    bs=0.02*vtrap(v+8.9,5)
    
}

FUNCTION vtrap(x,y) {            :Traps for 0 in denominator of rate eqns., based on three terms of infinite series expansion of exp
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
    } else {
        vtrap = x/(exp(x/y) - 1)
    }
}
