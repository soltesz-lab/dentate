: Ca channels (T,N,L-type)


NEURON {
	SUFFIX Caconc_Aradi
	USEION ca READ ica WRITE eca, cai
	RANGE depth, tau, cai0, cao0, cai, eca
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
	tau = 9		(ms) 
	depth = 0.2 	(um)  
	cai0 = 7e-5 	(mM)
 	cao = 2.0 		(mM)
}

ASSIGNED {
	v			(mV)
	ica			(mA/cm2)
	eca			(mV)
 	diam 		(um)
	VSR 		(um)
 	B 			(mM*cm2/mA)
        celsius (degC)
}

STATE {
	cai (mM) <1e-5>
}

BREAKPOINT {
    SOLVE state METHOD derivimplicit
    eca = ktf() * log(cao/cai) 
}

DERIVATIVE state {	: exact when v held constant; integrates over dt step
	cai' = -ica * B - (cai-cai0)/tau
}

INITIAL {
	cai = cai0
        eca = ktf() * log(cao/cai)
	: From Beining (2017)
	if (2. * depth >= diam) {
		VSR = 0.25 * diam : if diameter gets less than double the depth,
						: the surface to volume ratio (here volume to surface ratio VSR)
						: cannot be less than 1/(0.25diam) (instead of diam/(d*(diam-d)) )
	} else{
		VSR = depth*(1-depth/diam)
	}
	B = (1e4)/(2*F*VSR)
    }
    
FUNCTION ktf() (mV) {
        ktf = (1000)*R*(celsius +273.15)/(2*F)
}