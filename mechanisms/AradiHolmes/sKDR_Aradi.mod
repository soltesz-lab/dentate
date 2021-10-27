
NEURON {
	SUFFIX sKDR_Aradi
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
    g = gmax * n^4
    ik = g*(v - ek)
} 

DERIVATIVE states {
   rates(v)
   n' = an*(1 - n) - bn*n
}


PROCEDURE rates(v (mV)) { LOCAL anx
    
    anx = 0.1667*(v + 35) / 1(mV)
    an = 0.168*anx/(1 - exp(-anx))
    bn = 0.1056*exp(-0.025*(v + 60) / 1(mV))
    
}
