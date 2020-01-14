TITLE Fast delayed rectifier potassium channel from Lien et al 2002

COMMENT
Fast delayed rectifier potassium channel (voltage dependent)

From: Lien ... Jonas, J Physiol 2002: Gating, modulation and subunit
composition of voltage-gated K+channels in dendritic inhibitory
interneurones of rat hippocampus.

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
	SUFFIX fKdrcck
	USEION k READ ek WRITE ik VALENCE 1
	RANGE gmax, minf, mtau, hinf, ik
}
 
PARAMETER {
	celsius (degC) : temperature - set in hoc; default is 6.3
        
	gmax = 0.0002   	(mho/cm2)	
								
	a0m=0.036
	vhalfm=-33
	zetam=0.1
	gmm=0.7
	htau=1000
	q10=3
	f=0.92
}

ASSIGNED {
    v
    ik 		(mA/cm2)
    ek
    minf
    mtau (ms)	 	
    hinf	 	
}

STATE { m h }

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
        minf = (1/(1 + exp(-(v+36.2)/16.1)))^4
	mtau = betm(v)/(qt*a0m*(1+alpm(v)))

        hinf = f*(1/(1 + exp((v+40.6)/7.8)))+(1-f)
}

FUNCTION alpm(v(mV)) {
  alpm = exp(zetam*(v-vhalfm)) 
}

FUNCTION betm(v(mV)) {
  betm = exp(zetam*gmm*(v-vhalfm)) 
}

