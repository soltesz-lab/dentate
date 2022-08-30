TITLE dual-exponential model of NMDA receptors with HH-type gating

COMMENT
This is a simple double-exponential model of an NMDAR 
that has a slow voltage-dependent gating component in its conductance (3rd differential equations)

Mg++ voltage dependency from Spruston95 -> Woodhull, 1973
Keivan Moradi 2011 

--- (and now back to the original exp2syn comments) ---

Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

In the initial block we initialize the factor and total and A and B to starting values. The factor is 
defined in terms of tp, a local variable which defined the time of the peak of the function as 
determined by the tau1 and tau2.  tp is the maximum of the function exp(-t/tau2)  exp(-t/tau1).  To 
verify this for yourself, take the derivative, set it to 0 and solve for t.  The result is tp as defined 
here. Factor is the value of this function at time tp, and 1/factor is the normalization applied so 
that the peak is 1.  Then the synaptic weight determines the maximum synaptic conductance.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method. 

ENDCOMMENT

NEURON {
	POINT_PROCESS Exp3NMDA
	NONSPECIFIC_CURRENT i
	RANGE tau1, tau2, v0_tau2, st_tau2, tau3, tauV, e, i, gVI, st_gVD, v0_gVD, Mg, K0, delta, wf
	GLOBAL inf
	THREADSAFE
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
	(S)  = (siemens)
	(pS) = (picosiemens)
	(um) = (micron)
	(J)  = (joules)
}

PARAMETER {
: Parameters Control Neurotransmitter and Voltage-dependent gating of NMDAR
	tau1 = 8.8		(ms)	<1e-9,1e9>	: Spruston95 CA1 dend [at Mg = 0 v=-80]	be careful: Mg can change these values
	tau2 = 500		(ms)
	v0_tau2 = 161.11	(mV)	: Calculated based on Kampa04 data, this is an imaginary membrane voltage in which tau2 reaches zero
	st_tau2 =0.30342 (ms/mV)	: Calculated based on Kampa04 data, degree of change in tau2 with respect to the membrane potential
: Parameters Control voltage-dependent gating of NMDAR
	tauV = 7		(ms)	<1e-9,1e9>	: Kim11 
							: at 26 degC & [Mg]o = 1 mM, 
							: [Mg]o = 0 reduces value of this parameter
							: Because TauV at room temperature (20) & [Mg]o = 1 mM is 9.12 Clarke08 & Kim11 
							: and because Q10 at 26 degC is 1.52
							: then tauV at 26 degC should be 7 
	st_gVD = 0.007	(1/mV)	: steepness of the gVD-V graph from Clarke08 -> 2 units / 285 mv
	v0_gVD = -100	(mV)	: Membrane potential at which there is no voltage dependent current, from Clarke08 -> -90 or -100
	gVI = 1			(uS)	: Maximum Conductance of Voltage Independent component, This value is used to calculate gVD
	Q10 = 1.52				: Kim11
	T0 = 26			(degC)	: reference temperature 
: Parameters Control Mg block of NMDAR
	Mg = 1			(mM)	: external magnesium concentration from Spruston95
	K0 = 4.1		(mM)	: IC50 at 0 mV from Spruston95
	delta = 0.8 	(1)		: the electrical distance of the Mg2+ binding site from the outside of the membrane from Spruston95
: Parameter Controls Ohm's law in NMDAR
	e = -0.7		(mV)	: in CA1-CA3 region = -0.7 from Spruston95
}

CONSTANT {
	T = 273.16	(degC)
	F = 9.648e4	(coul)	: Faraday's constant (coulombs/mol)
	R = 8.315	(J/degC): universal gas constant (joules/mol/K)
	z = 2		(1)		: valency of Mg2+
}

ASSIGNED {
	v		(mV)
	i		(nA)
	g		(uS)
	factor
	wf
	inf		(uS)
	tau		(ms)
	celsius 		(degC)	: actual temperature for simulation, defined in Neuron, usually about 35
}

STATE {
	A
	B
	C
	gVD (uS)
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
	
	: temperature-sensitivity of the slow unblock of NMDARs
	tau = tauV * Q10^((T0 - celsius)/10(degC))

	gVD = 0
	wf = 1
	Mgblock(v)
	rates(v)
}

BREAKPOINT {
	SOLVE state METHOD derivimplicit

	i = (B - A)*(gVI + gVD)*Mgblock(v)*(v - e)
}

DERIVATIVE state { LOCAL x
	rates(v)
	A' = -A/tau1
	B' = -B/tau2
	: Voltage Dependent Gating of NMDA needs prior binding to Glutamate Kim11
        x = 0
        if (wf > 0) {
            x = B/wf
        }
	gVD' = x*(inf-gVD)/tau
}

NET_RECEIVE(weight) {
	wf = weight*factor
	A = A + wf
	B = B + wf
}

FUNCTION Mgblock(v(mV)) {
	: from Spruston95
	Mgblock = 1 / (1 + (Mg/K0)*exp((0.001)*(-z)*delta*F*v/R/(T+celsius)))
}

PROCEDURE rates(v (mV)) { 
	inf = (v - v0_gVD) * st_gVD * gVI
}