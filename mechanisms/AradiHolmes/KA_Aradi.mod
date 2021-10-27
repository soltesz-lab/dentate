: KA channel


NEURON {
	SUFFIX KA_Aradi
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
        ak	        (/ms)	
	bk		(/ms)	
        al	        (/ms)	
	bl		(/ms)	
}

STATE { k l }

INITIAL { 
	rates(v)
	k = ak/(ak + bk)
	l = al/(al + bl)
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gmax * k * l
    ik = g*(v - ek)
} 

DERIVATIVE states {
   rates(v)
   k' = ak*(1 - k) - bk*k
   l' = al*(1 - l) - bl*l
}


PROCEDURE rates(v (mV)) { LOCAL akx, bkx
    
    akx = 0.06667*(v + 25) / 1(mV)
    ak = 0.75*akx/(1 - exp(-akx)) 
    
    bkx = -0.125*(v + 15) / 1(mV)
    bk = 0.8*bkx/(1 - exp(-bkx)) 

    al = 0.00015*exp(-0.06667*(v + 13) / 1(mV))            
    bl = 0.06/(1 + exp(0.08333*(-68 - v) / 1(mV)))
    
}



