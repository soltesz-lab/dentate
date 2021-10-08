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
        ninf
        ntau (ms)
}

STATE { n }

INITIAL { 
	rates(v)
	n = ninf
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gmax * n
    ik = g*(v - ek)
} 

DERIVATIVE states {
   rates(v)
   n' = (ninf - n)/ntau
  
}


PROCEDURE rates(v (mV)) { LOCAL sum
    
    an=0.016*vtrap(-24.9-v, 5)
    bn=0.25*exp(-1-0.025*v)
    
    sum = an + bn
    ninf = an/sum
    ntau = 1/sum
}

 
FUNCTION vtrap(x,y) {            :Traps for 0 in denominator of rate eqns., based on three terms of infinite series expansion of exp
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
    } else {
        vtrap = x/(exp(x/y) - 1)
    }
}
