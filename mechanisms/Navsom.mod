TITLE sodium channel (voltage dependent)

COMMENT
sodium channel (voltage dependent)

Ions: na

Style: quasi-ohmic

From: not sure where from

Updates:
2014 December (Marianne Bezaire): documented
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
 
NEURON { 
	SUFFIX Navsom
	USEION na READ ena WRITE ina VALENCE 1
	RANGE g, gmax, minf, mtau, hinf, htau, ina, m, h
	THREADSAFE
}
 
PARAMETER {
	ena  (mV)
	gmax (mho/cm2)   

	mAlphC = -0.3 (1)
	mAlphV = 43 (mV)
	mBetaC = 0.3 (1)
	mBetaV = 15 (mV)

	hAlphC = 0.23 (1)
	hAlphV = 65 (mV)
	hBetaC = 3.33 (1)
	hBetaV = 12.5 (mV)
}
 
STATE {
	m h
}
 
ASSIGNED {
	v (mV) 
	celsius (degC) : temperature - set in hoc; default is 6.3
	dt (ms) 

	g (mho/cm2)
	ina (mA/cm2)
	minf
	hinf
	mtau (ms)
	htau (ms)
	mexp
	hexp 
} 

BREAKPOINT {
	SOLVE states
	g = gmax*m*m*m*h  
	ina = g*(v - ena)
}
 
UNITSOFF
 
INITIAL {
	rates(v)
	m = minf
	h = hinf
}

DERIVATIVE states {	:Computes state variables m, h, and n 
	rates(v)			:      at the current v and dt.
	m' = (minf-m) / mtau
	h' = (hinf-h) / htau
}
 
LOCAL q10	: declare outside a block so available to whole mechanism
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	LOCAL  alpha, beta, sum	: only available to block; must be first line in block

	q10 = 3^((celsius - 34)/10)

	:"m" sodium activation system - act and inact cross at -40
	alpha = mAlphC*vtrap((v+mAlphV),-5)
	beta = mBetaC*vtrap((v+mBetaV),5)
	sum = alpha+beta        
	mtau = 1/sum 
	minf = alpha/sum
	
	:"h" sodium inactivation system
	alpha = hAlphC/exp((v+hAlphV)/20)
	beta = hBetaC/(1+exp((v+hBetaV)/-10))
	sum = alpha+beta
	htau = 1/sum 
	hinf = alpha/sum 		
}
 
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
	if (fabs(x/y) < 1e-6) {
		vtrap = y*(1 - x/y/2)
	}else{  
		vtrap = x/(exp(x/y) - 1)
	}
}
 
UNITSON

