: K channel


NEURON {
	SUFFIX KAHP_PR
	USEION k READ ek WRITE ik
        USEION ca READ cai
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
        qinf	        (/ms)	
	tauq		(ms)	
        cai (mM)
}

STATE { q }

INITIAL { 
	rates(v)
        q=0.0811213
}
    
BREAKPOINT {
    SOLVE states METHOD derivimplicit
    g = gmax * q
    ik = g*(v - ek)
} 

DERIVATIVE states {
   rates(v)
   q'=(qinf-q)/tauq
}


PROCEDURE rates(v (mV)) { 
    qinf=(0.7894*exp(0.0002726*cai))-(0.7292*exp(-0.01672*cai))    
    tauq=(657.9*exp(-0.02023*cai))+(301.8*exp(-0.002381*cai))
}



