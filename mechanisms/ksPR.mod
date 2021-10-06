: K channel


NEURON {
	SUFFIX K_PR
	USEION k READ ek WRITE ik
	RANGE gmax, g
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

PARAMETER {
  gmax = 0 (S/cm2)

}

ASSIGNED {
	v		(mV)
	ek		(mV)
	ik		(mA/cm2)
        g  	        (S/cm2)
        an	        (/ms)	
	bn		(/ms)	
}

STATE { n }

INITIAL { 
	rates(v)
	n = an/(an + bn)
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gmax * n
    ik = g*(v - ek)
} 

DERIVATIVE states {
   rates(v)
   n'=(an - n*(an + bn))
}


PROCEDURE rates(v (mV)) { 
    
    an=0.016*(-24.9-v)/(exp((-24.9-v)/5)-1)
    bn=0.25*exp(-1-0.025*v)
    
}



