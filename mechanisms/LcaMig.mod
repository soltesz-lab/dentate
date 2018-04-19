TITLE l-calcium channel
: l-type calcium channel


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
	glcabar		 (mho/cm2)
	ki=.001 (mM)
	cao (mM)
        tfa=1
}


NEURON {
	SUFFIX lca
	USEION lca READ elca WRITE ilca VALENCE 2
	USEION ca READ cai, cao
        RANGE glcabar, cai, ilca, elca
        :GLOBAL minf,matu
	RANGE minf,matu
}

STATE {
	m
}

ASSIGNED {
	v (mV)
	celsius 	(degC)
	ilca (mA/cm2)
        glca (mho/cm2)
        minf
        matu   (ms)
	elca (mV)   
	cai (mM)

}

INITIAL {
	rate(v)
	m = minf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	glca = glcabar*m*m*h2(cai)
	ilca = glca*ghk(v,cai,cao)

}

FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}


FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
        LOCAL nu,f

        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exptrap(1,nu))*efun(nu)
}

FUNCTION KTF(celsius (DegC)) (mV) {
        KTF = ((25./293.15)*(celsius + 273.15))
}


FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exptrap(2,z) - 1)
	}
}

FUNCTION alp(v(mV)) (1/ms) {
:	TABLE FROM -150 TO 150 WITH 200
	alp = 15.69*(-1.0*v+81.5)/(exptrap(3,(-1.0*v+81.5)/10.0)-1.0)
}

FUNCTION bet(v(mV)) (1/ms) {
:	TABLE FROM -150 TO 150 WITH 200
	bet = 0.29*exptrap(4,-v/10.86)
}

DERIVATIVE state {  
        rate(v)
        m' = (minf - m)/matu
}

PROCEDURE rate(v (mV)) { :callable from hoc
        LOCAL a
        a = alp(v)
        matu = 1/(tfa*(a + bet(v)))
        minf = tfa*a*matu
}
 



FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap LcaMig [%g]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}
