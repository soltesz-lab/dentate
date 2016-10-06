: A conductance

NEURON {
	SUFFIX KA
	USEION k WRITE ik
	RANGE gbar, minf, mtau, hinf, htau, i, g, m, h
	GLOBAL erev
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
} 

PARAMETER {
	erev = -85		(mV)		: effective Ek
	gbar = 0   		(S/cm2)
} 


ASSIGNED {
	v 		(mV)
	i 		(mA/cm2)
	ik 		(mA/cm2)
	g			(S/cm2)

	malpha	(/ms)		
	mbeta		(/ms)	
	minf
	mtau 		(ms)

	halpha	(/ms)		
	hbeta		(/ms)
	hinf
	htau 		(ms)
}

STATE { m h }

INITIAL { 
	rates(v)
	m = minf
	h = hinf
}

BREAKPOINT {
  SOLVE states METHOD cnexp
	g = gbar*m*h
	ik = g*(v - erev)
	i = ik
} 

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/mtau
	h' = (hinf - h)/htau
}

PROCEDURE rates(v (mV)) {
  malpha = -0.05*(v+25)/(exptrap(1,-(v+25)/15)-1)
  mbeta = 0.1*(v+15)/(exptrap(2,(v+15)/8)-1)
  mtau = 1/(malpha + mbeta)
  minf = malpha/(malpha + mbeta)

  halpha = (1.5e-4)/exptrap(3,(v+13)/15)
  hbeta = 0.06/(exptrap(4,-(v+68)/12)+1)
  htau = 1/(halpha + hbeta)
  hinf = halpha/(halpha + hbeta)
}


FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap Aradi_KA [%d]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}
