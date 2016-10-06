: ad-hoc unknown conductance

NEURON {
	SUFFIX UK
	NONSPECIFIC_CURRENT i
	USEION ca READ ica
	RANGE i, g, gbar
	GLOBAL erev, taucadiv, kca, ca0, k, tau, p, Vhalf
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
	B = .26 (mM-cm2/mA-ms)
} 

PARAMETER {
	erev = -55    (mV)     
	gbar = 200    (pS/um2)
	p = 4
	Vhalf = -55   (mV)
	k = 1         (mV)
	ca0 = .00007	(mM)
	tauca = 9		  (ms)
	taucadiv = 1
	tau = 10      (ms)
	kca = 0.01    (mM)
} 

ASSIGNED {
	v 				(mV)
	celsius		(degC)
	i 				(mA/cm2)
	g 				(pS/um2)
	ica		    (mA/cm2)
	minf
	qinf
}
 
STATE { 
	ca_i (mM)		<1e-5> 
	q 
}

INITIAL {
	ca_i = ca0
	minf = 1/(1 + exptrap(0,-(v-Vhalf)/k))
	q = minf*(1 - exptrap(0,-ca_i/kca))
}

BREAKPOINT {
	SOLVE state METHOD derivimplicit
	g = gbar*q^p
	i = g*(v - erev)*(1e-4)
}

DERIVATIVE state {
	ca_i' = -B*ica - taucadiv*(ca_i-ca0)/tauca
        minf = 1/(1 + exptrap(1,-(v-Vhalf)/k))
	qinf = 1 - exptrap(2,-ca_i/kca)
	q' = (minf*qinf - q)/tau
} 

FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap DGC_UK [%d]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}

