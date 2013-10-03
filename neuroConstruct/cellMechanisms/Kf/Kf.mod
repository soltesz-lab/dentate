TITLE Kf.mod  
 
COMMENT
Fast potassium current extracted from ichan2.mod
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

USEION kf READ ekf WRITE ikf  VALENCE 1

RANGE   gkf
RANGE  gmax
RANGE nfinf, nftau,  ikf


}
 
INDEPENDENT {t FROM 0 TO 100 WITH 100 (ms)}
 
PARAMETER {
        v (mV) 
        celsius = 6.3 (degC)
        dt (ms)  
        ekf  = -90 (mV)
        gmax = %Max Conductance Density% (mho/cm2)
    
}
 
STATE {
     nf 
}
 
ASSIGNED {
         
       ? gnat (mho/cm2) 
        gkf (mho/cm2)
      ?  gks (mho/cm2)

     ?   inat (mA/cm2)
        ikf (mA/cm2)
    ?    iks (mA/cm2)


    ?il (mA/cm2)

    ?minf hinf nfinf nsinf
    ?mtau (ms) htau (ms) nftau (ms) nstau (ms)
    ?mexp hexp nfexp nsexp
    nfinf 

    nftau (ms)
     nfexp 
} 



BREAKPOINT {

    SOLVE states
     
        gkf = gmax*nf*nf*nf*nf
        ikf = gkf*(v-ekf)
    
}
 
UNITSOFF
 
INITIAL {
    trates(v)
    

      nf = nfinf
    
    VERBATIM
    return 0;
    ENDVERBATIM
}

? states
PROCEDURE states() {    :Computes state variables n 
        trates(v)   :      at the current v and dt.
?        m = m + mexp*(minf-m)
    ?    h = h + hexp*(hinf-h)
        nf = nf + nfexp*(nfinf-nf)
    ?    ns = ns + nsexp*(nsinf-ns)
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
                
                

    
            :"nf" fKDR activation system
            
    alpha = -0.07*vtrap((v+65-47),-6)
    beta = 0.264/exp((v+65-22)/40)
    
    
    sum = alpha+beta        
    nftau = 1/sum      nfinf = alpha/sum
    
}
 
PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
    LOCAL tinc
        TABLE  nfinf, nfexp,  nftau
    DEPEND dt, celsius FROM -100 TO 100 WITH 200
                           
    rates(v)    : not consistently executed from here if usetable_hh == 1
        : so don't expect the tau values to be tracking along with
        : the inf values in hoc

           tinc = -dt * q10
     ?   mexp = 1 - exp(tinc/mtau)
     ?   hexp = 1 - exp(tinc/htau)
    nfexp = 1 - exp(tinc/nftau)
    ?nsexp = 1 - exp(tinc/nstau)
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{  
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON

