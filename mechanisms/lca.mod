TITLE L-type calcium channel (voltage dependent)
 
COMMENT
L-Type Ca2+ channel (voltage dependent)

Ions: ca

Style: ghk

From: Jaffe et al, 1994


Updates:
2014 December (Marianne Bezaire): documented
ENDCOMMENT


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
	FARADAY = 96520 (coul)
	R = 8.3134 (joule/degC)
	KTOMV = .0853 (mV/degC)
}

PARAMETER {
	v (mV)
      celsius (degC) : temperature - set in hoc; default is 6.3
	gmax		 (mho/cm2)
	ki=.001 (mM)
	cai (mM)
	cao (mM)
        tfa=1
}


NEURON {
	SUFFIX CavLcck
	USEION ca READ cai, cao, eca WRITE ica VALENCE 2 
        RANGE gmax, cai, ica, eca, g, minf, mtau
    THREADSAFE
}

STATE {
	m
}

ASSIGNED {
	ica (mA/cm2)
        g (mho/cm2)
        minf
        mtau   (ms)
	eca (mV)   

}

INITIAL {
	rate(v)
	m = minf
	VERBATIM
    #if NRNBBCORE
    double * _nt_data = _nt->_data;
    #endif
	cai=_ion_cai;
	ENDVERBATIM
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = gmax*m*m*h2(cai)
	ica = g*ghk(v,cai,cao)
}

FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}

FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
        LOCAL nu,f

        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (DegC)) (mV) {
        KTF = ((25./293.15)*(celsius + 273.15))
}


FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}

FUNCTION alp(v(mV)) (1/ms) {
	alp = 15.69*(-1.0*v+81.5)/(exp((-1.0*v+81.5)/10.0)-1.0)
}

FUNCTION bet(v(mV)) (1/ms) {
	bet = 0.29*exp(-v/10.86)
}

DERIVATIVE state {  
	rate(v)
	m' = (minf - m)/mtau
}

PROCEDURE rate(v (mV)) { :callable from hoc
	LOCAL a
	a = alp(v)
	mtau = 1/(tfa*(a + bet(v)))
	minf = tfa*a*mtau
}
