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
    Q10 = 3 (1)

}

ASSIGNED {
	v		(mV)
	m_inf		(mV)
        m_tau (ms)
	ena		(mV)
	ina		(mA/cm2)
        g  	        (S/cm2)
        ah	        (/ms)	
	bh		(/ms)	
        am	        (/ms)	
	bm		(/ms)	
        celsius (degC)
}

STATE { h }

INITIAL { 
	rates(v)
	h = 0.99806345
        m_inf=am/(am + bm)
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gmax*(m_inf^2)*h
    ina = g*(v - ena)
} 

DERIVATIVE states {
   rates(v)
   h' = ah - h*(ah + bh)
}


PROCEDURE rates(v (mV)) { LOCAL tcorr
    
    tcorr = Q10^((celsius - 36(degC))/10 (degC))
    
    am=tcorr*0.32*linoid(-46.9-v, 4)
    bm=tcorr*0.28*linoid(v+19.9, 5)
    m_inf=am/(am + bm)
    
    ah=tcorr*0.128*exp((-43-v)/18)
    bh=tcorr*4./(1+exp((-20-v)/5))
}

 
FUNCTION linoid(x,y) {
    if (fabs(x/y) < 1e-6) {
        linoid = y*(1 - x/y/2)
    } else {
        linoid = x/(exp(x/y) - 1)
    }
}
