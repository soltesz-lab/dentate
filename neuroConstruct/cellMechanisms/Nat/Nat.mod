TITLE ichan2.mod  
 
COMMENT
Sodium current extracted from ichan2.mod
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
 
? interface 
NEURON { 
SUFFIX %Name% 
USEION nat READ enat WRITE inat VALENCE 1
RANGE  gnat
RANGE gmax
RANGE minf, mtau, hinf, htau,  inat
}
 
INDEPENDENT {t FROM 0 TO 100 WITH 100 (ms)}
 
PARAMETER {
        v (mV) 
        celsius = 6.3 (degC)
        dt (ms) 
        enat = 45 (mV)
	gmax = %Max Conductance Density% (mho/cm2)   
}
 
STATE {
	m h 
}
 
ASSIGNED {
         
        gnat (mho/cm2) 

        inat (mA/cm2)


	minf hinf 
 	mtau (ms) htau (ms) 
	mexp hexp 
} 

? currents
BREAKPOINT {
	SOLVE states
        gnat = gmax*m*m*m*h  
        inat = gnat*(v - enat)
}
 
UNITSOFF
 
INITIAL {
	trates(v)
	
	m = minf
	h = hinf
	
	VERBATIM
	return 0;
	ENDVERBATIM
}

? states
PROCEDURE states() {	:Computes state variables m, h, and n 
        trates(v)	:      at the current v and dt.
        m = m + mexp*(minf-m)
        h = h + hexp*(hinf-h)
        VERBATIM
        return 0;
        ENDVERBATIM
}
 
LOCAL q10

? rates
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL  alpha, beta, sum
       q10 = 3^((celsius - 6.3)/10)
                :"m" sodium activation system - act and inact cross at -40
	alpha = -0.3*vtrap((v+60-17),-5)
	beta = 0.3*vtrap((v+60-45),5)
	sum = alpha+beta        
	mtau = 1/sum      minf = alpha/sum
                :"h" sodium inactivation system
	alpha = 0.23/exp((v+60+5)/20)
	beta = 3.33/(1+exp((v+60-47.5)/-10))
	sum = alpha+beta
	htau = 1/sum 
        hinf = alpha/sum 
	
}
 
PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	LOCAL tinc
        TABLE minf, mexp, hinf, hexp,  mtau, htau
	DEPEND dt, celsius FROM -100 TO 100 WITH 200
                           
	rates(v)	: not consistently executed from here if usetable_hh == 1
		: so don't expect the tau values to be tracking along with
		: the inf values in hoc

	       tinc = -dt * q10
        mexp = 1 - exp(tinc/mtau)
        hexp = 1 - exp(tinc/htau)
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{  
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON

