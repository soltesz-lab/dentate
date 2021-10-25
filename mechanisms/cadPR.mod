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
    Q10 = 3 (1)
}
    
ASSIGNED {
        eca     (mV)
	v		(mV)
	ica		(mA/cm2)
	g		(S/cm2)
        as
        bs
        celsius (degC)
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
    s = as/(as + bs)
}


PROCEDURE rates(v (mV)) { LOCAL tcorr
    
    tcorr = Q10^((celsius - 36(degC))/10 (degC))
    
    as=tcorr*1.6 / (exp(-0.072*(v-5))+1)
    bs=tcorr*0.02*linoid(v+8.9,5)
    
}

FUNCTION linoid(x,y) {
    if (fabs(x/y) < 1e-6) {
        linoid = y*(1 - x/y/2)
    } else {
        linoid = x/(exp(x/y) - 1)
    }
}
