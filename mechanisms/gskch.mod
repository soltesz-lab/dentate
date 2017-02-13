TITLE gskch.mod  calcium-activated potassium channel (non-voltage-dependent)

COMMENT

gsk granule

ENDCOMMENT

UNITS {
        (molar) = (1/liter)
        (mM)    = (millimolar)
	(mA)	= (milliamp)
	(mV)	= (millivolt)
}

NEURON {
	SUFFIX gskch
	USEION sk READ esk WRITE isk VALENCE 1
	USEION nca READ ncai VALENCE 2
	USEION lca READ lcai VALENCE 2
	USEION tca READ tcai VALENCE 2
	RANGE gsk, gskbar, qinf, qtau, isk
}

:INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {
	gskbar  (mho/cm2)
}

STATE { q }

ASSIGNED {
	isk (mA/cm2) gsk (mho/cm2) qinf qtau (ms)
	esk	(mV)
	celsius (degC)
	v	(mV)
	cai (mM)
	ncai (mM)
	lcai (mM)
	tcai (mM)
}


BREAKPOINT {          :Computes i=g*q^2*(v-esk)
	SOLVE state METHOD cnexp
        gsk = gskbar * q*q
	isk = gsk * (v-esk)
}

UNITSOFF

INITIAL {
    cai = 5.e-5
    rate(cai)
    q=qinf
}


DERIVATIVE state {  :Computes state variable q at current v and dt.
	cai = ncai + lcai + tcai
	rate(cai)
	q' = (qinf - q) / qtau
}

PROCEDURE rate(cai) {  :Computes rate and other constants at current v.
	LOCAL alpha, beta, q10
	q10 = 3^((celsius - 6.3)/10)
		:"q" activation system
        alpha = 1.25e1 * cai * cai
        beta = 0.00025 

:	alpha = 0.00246/exp((12*log10(cai)+28.48)/-4.5)
:	beta = 0.006/exp((12*log10(cai)+60.4)/35)
: alpha = 0.00246/fctrap(cai)
: beta = 0.006/fctrap(cai)
	qtau = 1 / (alpha + beta)
	qinf = alpha * qtau
}

UNITSON
