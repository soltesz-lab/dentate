: Na channel


NEURON {
	SUFFIX Na_Aradi
	USEION na READ ena WRITE ina
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
	ena		(mV)
	ina		(mA/cm2)
        g  	        (S/cm2)
        am	        (/ms)	
	bm		(/ms)	
        ah	        (/ms)	
	bh		(/ms)	
}

STATE { m h }

INITIAL { 
	rates(v)
	m = am/(am + bm)
	h = ah/(ah + bh)
}
    
BREAKPOINT {
  SOLVE states METHOD cnexp
  g = gmax*(m^3)*h
  ina = g*(v - ena)
} 

DERIVATIVE states {
   rates(v)
   m' = am*(1 - m) - bm*m  
   h' = ah*(1 - h) - bh*h
}


PROCEDURE rates(v (mV)) { LOCAL amx, bmx
    
    amx = 0.2*(v + 45) / 1(mV)
    am = 1.5*amx/(1 - exp(-amx)) 
    bmx = -0.2*(v + 17) / 1(mV)
    bm = 1.5*bmx/(1 - exp(-bmx)) 
    ah = 0.23*exp(-0.05*(v + 67) / 1(mV)) 
    bh = 3.33/(1 + exp(0.1*(-14.5 - v) / 1(mV)))
}