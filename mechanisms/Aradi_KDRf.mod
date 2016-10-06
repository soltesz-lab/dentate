: Fast delayed rectifier

NEURON {
	SUFFIX fKDR
	USEION k WRITE ik
	RANGE gbar, minf, mtau, i, g, m
	GLOBAL erev, v0, taumult
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
} 

PARAMETER {
	erev = -85	  (mV)		: effective Ek
	v0 = 23       (mV)
	taumult = 1
	gbar = 0   		(S/cm2)
} 


ASSIGNED {
	v 			(mV)
	i 			(mA/cm2)
	ik			(mA/cm2)
	g				(S/cm2)
	malpha	(/ms)		
	mbeta		(/ms)
	minf
	mtau 		(ms)
}

STATE { m }

INITIAL { 
	rates(v)
	m = minf
}

BREAKPOINT {
  SOLVE states METHOD cnexp
	g = gbar*m^4
	ik = g*(v - erev)
	i = ik
} 

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/mtau
}

PROCEDURE rates(v (mV)) {
  malpha = 0.07*(v+v0)/(1-exptrap(1, -0.166*(v+v0)))
  mbeta = 0.264*exptrap(2, -0.025*(v+48))
  mtau = taumult/(malpha + mbeta)
  minf = malpha/(malpha + mbeta)
}



FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap Aradi_KDRf [%d]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}


