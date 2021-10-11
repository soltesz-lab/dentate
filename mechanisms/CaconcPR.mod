

NEURON {
	SUFFIX Ca_conc_PR
	USEION ca READ ica, cao WRITE eca, cai
	RANGE cai0, cai, eca, d
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	F = (faraday) (coulomb)
        R = 8.3134      (joule/degC)
}

PARAMETER {
	cai0 = 7e-5 	(mM)
        d = 13 (mM)
}

ASSIGNED {
	v			(mV)
	ica			(mA/cm2)
	eca			(mV)
	cao			(mM)
        celsius (degC)
}

STATE {
	cai (mM) <1e-5>
    }
    
BREAKPOINT {
        SOLVE state METHOD derivimplicit
}

DERIVATIVE state {
	cai' = -d*10*ica - 0.075*cai
}

INITIAL {
    cai = cai0
    eca = ktf() * log(cao/cai)
    }
    
FUNCTION ktf() (mV) {
        ktf = (1000)*R*(celsius +273.15)/(2*F)
}