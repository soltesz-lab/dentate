: Transient Na conductance

NEURON {
	SUFFIX Na
	USEION na READ ena WRITE ina
	RANGE gbar, minf, mtau, hinf, htau, i, g, m, h, taumult, htaumult
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
} 

PARAMETER {
	gbar = 0   		(S/cm2)
 	taumult = 1
 	htaumult = 1
} 


ASSIGNED {
	v 		(mV)
	i 		(mA/cm2)
	ina		(mA/cm2)
	g			(S/cm2)
	ena   (mV)

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
	g = gbar*(m^3)*h
	ina = g*(v - ena)
	i = ina
} 

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/mtau
	h' = (hinf - h)/htau
}

PROCEDURE rates(v (mV)) {
  malpha = 0.3*(v+45)/(1-exp(-0.2*(v+45)))
  mbeta = -0.3*(v+17)/(1-exp(0.2*(v+17)))
  mtau = taumult/(malpha + mbeta)
  minf = malpha/(malpha + mbeta)

  halpha = 0.23*exp(-0.05*(v+67))
  hbeta = 3.33/(1+exp(-0.1*(v+14.5)))
  htau = htaumult/(halpha + hbeta)
  hinf = halpha/(halpha + hbeta)
}

