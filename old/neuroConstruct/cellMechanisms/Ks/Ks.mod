TITLE ichan2.mod  
 
COMMENT
Slow potassium current extracted from ichan2.mod
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
    R = 8.3134  (joule/degC)
}
 

NEURON { 
SUFFIX %Name% 

USEION ks READ eks WRITE iks  VALENCE 1

RANGE gks
RANGE  gmax
RANGE nsinf, nstau, iks
}
 
INDEPENDENT {t FROM 0 TO 100 WITH 100 (ms)}
 
PARAMETER {
        v (mV) 
        celsius = 6.3 (degC)
        dt (ms) 
        
        eks = -90 (mV)
    gmax = %Max Conductance Density% (mho/cm2)
    
}
 
STATE {
    ns
}
 
ASSIGNED {
         

        gks (mho/cm2)

        
        iks (mA/cm2)

        
        
        

    nsinf
     nstau (ms)
    nsexp
} 

? currents
BREAKPOINT {
    SOLVE states
    
        gks = gmax*ns*ns*ns*ns
        iks = gks*(v-eks)

        
}
 
UNITSOFF
 
INITIAL {
    trates(v)
    
    
      ns = nsinf
    
    VERBATIM
    return 0;
    ENDVERBATIM
}

? states
PROCEDURE states() {    :Computes state variables m, h, and n 
        trates(v)   :      at the current v and dt.
  
        
        ns = ns + nsexp*(nsinf-ns)
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
       
  
             :"ns" sKDR activation system
             
    alpha = -0.028*vtrap((v+65-35),-6)
    beta = 0.1056/exp((v+65-10)/40)
    sum = alpha+beta        
    nstau = 1/sum      nsinf = alpha/sum
    
}
 
PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
    LOCAL tinc
        TABLE nsinf, nsexp, nstau
    DEPEND dt, celsius FROM -100 TO 100 WITH 200
                           
    rates(v)    : not consistently executed from here if usetable_hh == 1
        : so don't expect the tau values to be tracking along with
        : the inf values in hoc

           tinc = -dt * q10
           
    nsexp = 1 - exp(tinc/nstau)
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{  
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON

