TITLE nca.mod  
 
COMMENT
konduktivitas valtozas hatasa- somaban 
ENDCOMMENT
 
UNITS {
        (mA) =(milliamp)
        (mV) =(millivolt)
        (uF) = (microfarad)
	(molar) = (1/liter)
	(nA) = (nanoamp)
	(mM) = (millimolar)
	(um) = (micron)
	FARADAY = 96520 (coul)
	R = 8.3134	(joule/degC)
}
 
? interface 
NEURON { 
SUFFIX nca
USEION nca READ enca WRITE inca VALENCE 2 
RANGE  gnca
RANGE gncabar
RANGE cinf, ctau, dinf, dtau, inca
}
 
:INDEPENDENT {t FROM 0 TO 100 WITH 100 (ms)}
 
PARAMETER {
	gncabar (mho/cm2)
}
 
STATE {
	c d
}
 
ASSIGNED {
    
    v (mV) 
    celsius (degC)
    gnca (mho/cm2)
    inca (mA/cm2)
    enca (mV)
    
    cinf dinf
    ctau (ms) dtau (ms) 
    cexp dexp      
} 

? currents
BREAKPOINT {
	SOLVE states METHOD cnexp
        gnca = gncabar*c*c*d
	inca = gnca*(v-enca)
}
 
UNITSOFF
 
INITIAL {
	trates(v)
	c = cinf
	d = dinf
}

? states
DERIVATIVE states {	:Computes state variables m, h, and n 
        trates(v)	:      at the current v and dt.

	c' = (cinf - c) / ctau
	d' = (dinf - d) / dtau
}
 
LOCAL q10

? rates
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL  alpha, beta, sum
        q10 = 3^((celsius - 6.3)/10)
                :"c" NCa activation system
        alpha = -0.19*vtrap(v-19.88,-10)
	beta = 0.046*exptrap(1,-v/20.73)
	sum = alpha+beta        
	ctau = 1/sum      cinf = alpha/sum
                :"d" NCa inactivation system
	alpha = 0.00016/exptrap(2,-v/48.4)
	beta = 1/(exptrap(3,(-v+39)/10)+1)
	sum = alpha+beta        
	dtau = 1/sum
        dinf = alpha/sum
}
 
PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	rates(v)	: not consistently executed from here if usetable_hh == 1
		: so don't expect the tau values to be tracking along with
		: the inf values in hoc

}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{  
                vtrap = x/(exptrap(0,x/y) - 1)
        }
}


FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    printf("exptrap nca [%d]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}

 
UNITSON

