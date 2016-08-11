TITLE intracellular calcium accumulation

COMMENT
intracellular Ca2+ accumulation
From: 
Notes:
	calcium accumulation into a volume of area*depth next to the
	membrane with a decay (time constant tau) to resting level
	given by the global calcium variable cai0_ca_ion
	
Ions: ca

From: Modified from Aradi & Holmes 1999

Updates:
2014 December (Marianne Bezaire): documented
ENDCOMMENT



NEURON {
SUFFIX iconc_Ca
USEION ca READ cai, ica, eca WRITE eca, cai VALENCE 2
RANGE caiinf, catau, cai, eca
}

UNITS {
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (milli/liter)
	(mA) = (milliamp)
	FARADAY = 96520 (coul)
	R = 8.3134	(joule/degC)
}

:INDEPENDENT {t FROM 0 TO 100 WITH 100 (ms)}

PARAMETER {
    depth = 200 (nm)	: assume volume = area*depth
    catau = 9 (ms)
    caiinf = 50.e-6 (mM)
    cao = 2 (mM)
}

ASSIGNED {
    celsius (degC) 
    eca (mV)
    ica (mA/cm2)
}

STATE {
	cai
}

: verbatim blocks are not thread safe (perhaps related, this mechanism cannot be used with cvode)
INITIAL {
	cai = caiinf	
	eca = ktf() * log(cao/caiinf)	
}


BREAKPOINT {
	SOLVE integrate METHOD derivimplicit
	eca = ktf() * log(cao/cai)	
}

DERIVATIVE integrate {
cai' = -(ica)/depth/FARADAY * (1e7) + (caiinf - cai)/catau
}

FUNCTION ktf() (mV) {
	ktf = (1000)*R*(celsius +273.15)/(2*FARADAY)
} 
