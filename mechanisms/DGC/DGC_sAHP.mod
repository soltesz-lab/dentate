: sAHP conductance

NEURON {
	SUFFIX sAHP
	USEION ca READ ica
	USEION k READ ek WRITE ik
	RANGE gbar, g, i
	GLOBAL ca0, tau, taucadiv, n, kca, cah
	GLOBAL tau1Ref, tau2, oinf, c1inf, CaRef
	RANGE a2, a1max, b1, b2, tau1
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
	(pS) = (picosiemens)
	(um) = (micron)
}

PARAMETER {
	B = .26       (mM-cm2/mA-ms)
	gbar = .0  		(pS/um2)	
	ca0 = .00007	(mM)
	tau = 9		    (ms)
	taucadiv = 1
	n = 1
	tau1Ref = 200 (ms)
	tau2 = 200    (ms)
	c1inf = 0.25
	oinf = 0.5
	CaRef = .002  (mM)
	kca = 0.001   (mM)
	cah = 0.01    (mM)
}

ASSIGNED {
	v			(mV)
	ek		(mV)
	ik		(mA/cm2)
	i 		(mA/cm2)
	ica		(mA/cm2)
  g     (pS/um2)
  a1ca  (/ms)
  a1maxCaRef  (/ms)
  a1max (/ms/mM)
  a1    (/ms/mM)
  b1    (/ms)
  a2    (/ms)
  b2    (/ms)
  tau1  (ms)
}

STATE { 
	ca_i (mM)		<1e-5> 
	c1 
	o 
}

INITIAL {
	ca_i = ca0
	a2 = -(oinf/((-1 + c1inf)*tau2))
	b1 = -(c1inf/((-1 + oinf)*tau1Ref))
	b2 = 1/tau2 - a2
	a1maxCaRef = 1/tau1Ref - b1 
	a1max = a1maxCaRef/CaRef
	a1ca = (a1max*ca_i)/(1+exptrap(1,-(ca_i-cah)/kca))
	tau1 = 1/(a1ca + b1)
	o = (a2*(-1 + b1*tau1)*tau2)/(-1 + a2*b1*tau1*tau2)
	c1 = (b1*tau1*(-1 + a2*tau2))/(-1 + a2*b1*tau1*tau2)
}

BREAKPOINT {
	SOLVE state METHOD derivimplicit
	g = gbar*o^n
	ik = g*(v - ek)*(1e-4)
	i = ik
}

DERIVATIVE state {
	ca_i' = -B*ica - taucadiv*(ca_i-ca0)/tau
	a1ca = (a1max*ca_i)/(1+exptrap(2,-(ca_i-cah)/kca))
	c1' = b1*(1 - o) - c1*(a1ca + b1)
	o' = a2*(1 - c1) - o*(a2 + b2)
}

FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap DGC_sAHP [%d]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}
