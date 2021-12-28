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
        ar
        br
        celsius (degC)
}

STATE { s r }

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = gmax*(s^2)*r
    ica = g*(v - eca)
}

DERIVATIVE state {	
    rates(v)
    s' = (as - s*(as + bs))
    r' = (ar - r*(ar + br))
}

INITIAL {
    rates(v)
    s = as/(as + bs)
    r = ar/(ar + br)
}


PROCEDURE rates(v (mV)) { LOCAL tcorr, xs
    
    tcorr = Q10^((celsius - 36(degC))/10 (degC))
    
    as = tcorr*2/(1 + exp(0.09*(5 - v)))
    xs = -0.2*(v + 8.9)
    bs = tcorr*0.1*xs/(1 - exp(-xs))
    
    ar = tcorr*0.1673*exp(-0.03035*(v + 38.5))
    br = tcorr*0.5/(1 + exp(0.3*(8.9 - v)))

}

