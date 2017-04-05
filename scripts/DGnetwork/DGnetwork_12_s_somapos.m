
function [Soma_Points] =  DGnetwork_12_s_somapos()

HL_layer_min       = -3.95;
HL_layer_max       = -1.95;
HL_layer_mid       = -2.95;

Soma_Points = cell(12,1);

N_HIPP = 9000;
Soma_Grid_HIPP = DGnetwork_11_s_somagrid(50,50,@layer_eq_GCL,HL_layer_min,HL_layer_mid,HL_layer_max);
sz = size(Soma_Grid_HIPP);
Soma_Points_HIPP = Soma_Grid_HIPP(randsample(sz(1),N_HIPP),:);

Soma_Points{1} = Soma_Points_HIPP;

% Soma_Grid_IS = DGnetwork_11_s_somagrid(100,50,@layer_eq_GCL,HL_layer_min,HL_layer_mid,HL_layer_max);


% Save somata to file
% save(sprintf('Outputs/Soma_Grid_%s.mat',label),'Soma_Points','-v7.3');
