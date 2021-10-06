: Na channel


NEURON {
	SUFFIX Na_PR
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
	m_inf		(mV)
	ena		(mV)
	ina		(mA/cm2)
        g  	        (S/cm2)
        ah	        (/ms)	
	bh		(/ms)	
        am	        (/ms)	
	bm		(/ms)	
}

STATE { h }

INITIAL { 
	rates(v)
	h = 0.99806345
        m_inf=am/(am + bm)
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    rates(v)
    m_inf=am/(am + bm)
    g = gmax*(m_inf^2)*h
    ina = g*(v - ena)
} 

DERIVATIVE states {
   rates(v)
   h' = ah - h*(ah + bh)
}


PROCEDURE rates(v (mV)) { 
    am=0.32*(-46.9-v)/(exp((-46.9-v)/4)-1)
    bm=0.28*(v+19.9)/(exp((v+19.9)/5)-1)
    
    ah=0.128*exp((-43-v)/18)
    bh=4./(1+exp((-20-v)/5))
}