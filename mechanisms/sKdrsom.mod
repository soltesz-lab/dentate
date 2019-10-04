TITLE Slow K-DR current for hippocampal interneurons from Lien et al (2002)

NEURON {
	SUFFIX sKdrsom
	USEION k READ ek WRITE ik
	RANGE gmax, minf, mtau, hinf, ik
}

PARAMETER {
	gmax = 0.0002   	(mho/cm2)	
								
	celsius (degC)
	a0m=0.015
	vhalfm=-25
	zetam=0.15
	gmm=0.5
	htau=1000
	mmin=7
	q10=3
	f=0.93
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
        minf = (1/(1 + exp(-(v+41.9)/23.1)))^4
	mtau = betm(v)/(qt*a0m*(1+alpm(v)))
	if (mtau<mmin) {mtau=mmin}
        hinf = f*(1/(1 + exp((v+52.2)/15.2)))+(1-f)
}

FUNCTION alpm(v(mV)) {
  alpm = exp(zetam*(v-vhalfm)) 
}

FUNCTION betm(v(mV)) {
  betm = exp(zetam*gmm*(v-vhalfm)) 
}
