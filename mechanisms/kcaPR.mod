: K channel


NEURON {
	SUFFIX KCa_PR
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
        cinf	        (/ms)	
	tauc		(ms)	
        cai (mM)
}

STATE { c }

INITIAL { 
	rates(v)
        c=0.00809387
}
    
BREAKPOINT {
    SOLVE states METHOD derivimplicit
    g = gmax * c
    ik = g*(1.073*sin(0.003453*cai+0.08095) + 0.08408*sin(0.01634*cai-2.34) + 0.01811*sin(0.0348*cai-0.9918))*(v-ek)
} 

DERIVATIVE states {
   rates(v)
   c'=(cinf-c)/tauc
}


PROCEDURE rates(v (mV)) { 
    cinf=(1.0/(1.0+exp((-10.1-v)/0.1016)))^0.00925
    tauc=3.627*exp(0.03704*v)

}


