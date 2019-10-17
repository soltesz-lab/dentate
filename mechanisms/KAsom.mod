TITLE K-A current for hippocampal interneurons from Lien et al (2002)

NEURON {
	SUFFIX KAsom
	USEION k READ ek WRITE ik
	RANGE  gmax, minf, hinf, htau, mtau, ik
}

PARAMETER {
	gmax = 0.0002   	(mho/cm2)	
	celsius (degC)
	a0h=0.17
	vhalfh=-105
	q10=3
	hmin=5
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
    v
    ik 		(mA/cm2)
    ek
    minf
    mtau (ms)
    hinf
    htau (ms)
}
 

STATE { m h}

BREAKPOINT {
        SOLVE states METHOD cnexp
	ik = gmax*m*h*(v - ek)
} 

INITIAL {
	rates(v)
	m=minf  
	h=hinf  
}

DERIVATIVE states {   
        rates(v)      
        m' = (minf-m)/mtau
        h' = (hinf-h)/htau
}

PROCEDURE rates(v) {  
	LOCAL qt
        qt=q10^((celsius-23)/10)
        minf = (1/(1 + exp(-(v+41.4)/26.6)))^4
	mtau=0.5/qt
        hinf = 1/(1 + exp((v+78.5)/6))
	htau = a0h*(v-vhalfh)/qt
	if (htau<hmin/qt) {htau=hmin/qt}
}

