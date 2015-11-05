

%function DGnetwork_8_s_ppsyn(directory,slice)

% Distribution of PP synapses
% Create a vector of 1000 random spine sizes in the range 0.01 to 0.2

a = 0.01
b = 0.2

PP_GC_syn_sizes = (b-a).*rand(1000,1) + a;

