
function [soma_points] =  DGnetwork_12_s_somapos()


% layer boundaries
HL_layer_min       = -3.95;
HL_layer_mid       = -2.95;
HL_layer_max       = -1.95;

GCL_layer_min       = -1.95;
GCL_layer_mid       = -0.95;
GCL_layer_max       = 0;

IML_layer_min       = 0;
IML_layer_mid       = 0.5;
IML_layer_max       = 1;

MML_layer_min       = 1;
MML_layer_mid       = 1.5;
MML_layer_max       = 2;

OML_layer_min       = 2;
OML_layer_mid       = 2.5;
OML_layer_max       = 3.1;

% indices for each cell type
N_MC       = 2;
N_HIPP     = 3;
N_PVBC     = 4;
N_AA       = 5;
N_HICAP    = 6;
N_NGFC     = 7;
N_IS       = 8;
N_MOPP     = 9;
N_MPP      = 10;
N_LPP      = 11;

num_types       = 11;
num_cells       = cell(num_types,1);
soma_disth      = zeros(num_types,1);
soma_distv      = zeros(num_types,1);
soma_border     = zeros(num_types,1);
distribution    = cell(num_types,1);
soma_points     = cell(num_types,1);
soma_locations  = cell(num_types,1);

%2 Mossy
num_cells{N_MC}    = [30000;0;0;0;0;30000]; %Hilus;GCL;IML;MML;OML;Total
soma_disth(N_MC)   = 20;
soma_distv(N_MC)   = 20;
septotemporal      = [20.8;38.2;67.6;75.4;51.2;48.6;84.1;68.5;145.7;385.0];
distribution{N_MC} = septotemporal/sum(septotemporal(:,1));

%3 HIPP
num_cells{N_HIPP}    = [9000;0;0;0;0;9000];
soma_disth(N_HIPP)  = 50;
soma_distv(N_HIPP)  = 50;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_HIPP} = septotemporal/sum(septotemporal(:,1));

%4 PVBC
num_cells{N_PVBC}    = [1700;1700;400;0;0;3800];
soma_disth(N_PVBC)   = 80;
soma_distv(N_PVBC)   = 80;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{N_PVBC} = septotemporal/sum(septotemporal(:,1));

%5 AA 
num_cells{N_AA}    = [200;200;50;0;0;450];
soma_disth(N_AA)   = 100;
soma_distv(N_AA)   = 100;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{N_AA} = septotemporal/sum(septotemporal(:,1));

%6 HICAP/CCK+ Basket Cells
num_cells{N_HICAP}    = [1150;250;0;0;0;1400];
soma_disth(N_HICAP)   = 100;
soma_distv(N_HICAP)   = 100;
septotemporal   = [9.1;16.3;13.1;11.3;9.3;8.3;8.3;9.9;9.8;6.1];
distribution{N_HICAP} = septotemporal/sum(septotemporal(:,1));

%7 NGFC
num_cells{N_NGFC}    = [0;0;0;2500;2500;5000];
soma_disth(N_NGFC)   = 50;
soma_distv(N_NGFC)   = 50;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{N_NGFC} = septotemporal/sum(septotemporal(:,1));

%8 IS
num_cells{N_IS}    = [3000;0;0;0;0;3000];
soma_disth(N_IS)   = 50;
soma_distv(N_IS)   = 50;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_IS} = septotemporal/sum(septotemporal(:,1));

%9 MOPP
num_cells{N_MOPP}    = [0;0;2000;500;500;3000];
soma_disth(N_MOPP)   = 80;
soma_distv(N_MOPP)   = 80;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_MOPP} = septotemporal/sum(septotemporal(:,1));

%10 MPP
num_cells{N_MPP}    = [0;0;0;38000;0;38000];
soma_disth(N_MPP)   = 20;
soma_distv(N_MPP)   = 20;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_MPP} = septotemporal/sum(septotemporal(:,1));

%11 LPP
num_cells{N_LPP}    = [0;0;0;0;34000;34000];
soma_disth(N_LPP)   = 20;
soma_distv(N_LPP)   = 20;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_LPP} = septotemporal/sum(septotemporal(:,1));

% Load somata and define GCL points
[x_o,y_o,z_o]   = layer_eq_GCL_2(GCL_layer_max);
[x_i,y_i,z_i]   = layer_eq_GCL_2(GCL_layer_min);
M_o             = [x_o,y_o,z_o];
M_i             = [x_i,y_i,z_i];

% Split GCL points into 100 micron bins for speedup
bin_width = 100;
xmax    = ceil(max(x_o)/bin_width)*bin_width;
xmin    = floor(min(x_o)/bin_width)*bin_width;
n_bins  = (xmax-xmin)/bin_width;
GCL_o   = cell(1,n_bins);
GCL_i   = cell(1,n_bins);
for bin = 1:n_bins
    GCL_o{bin} = M_o(M_o(:,1)>=(xmin+(bin-1)*bin_width)& M_o(:,1)<=(xmin+bin*bin_width),:);
    GCL_i{bin} = M_i(M_i(:,1)>=(xmin+(bin-1)*bin_width)& M_i(:,1)<=(xmin+bin*bin_width),:);
end

% Define granule cell layer parameters from layer_eq_GCL
u_params   = [pi*1/100,pi*98/100,2000];
v_params   = [pi*-23/100,pi*142.5/100,1000];


for i = 2:num_types
    i
    soma_xyz_points = [];
    for section = 1:5
        section
        switch section
        case 1
          k = num_cells{i}(section);
          if k > 0
             soma_grid = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_GCL,HL_layer_min,HL_layer_mid,HL_layer_max);
             sz = size(soma_grid);
             soma_xyz_points = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
          end
        case 2
          k = num_cells{i}(section);
          if k > 0
             soma_grid = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_GCL,GCL_layer_min,GCL_layer_mid,GCL_layer_max);
             sz = size(soma_grid);
             soma_xyz_points = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
          end
        case 3
          k = num_cells{i}(section);
          if k > 0
             soma_grid = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,IML_layer_min,IML_layer_mid,IML_layer_max);
             sz = size(soma_grid);
             soma_xyz_points = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
          end
        case 4
          k = num_cells{i}(section);
          if k > 0
             soma_grid = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,MML_layer_min,MML_layer_mid,MML_layer_max);
             sz = size(soma_grid);
             soma_xyz_points = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
          end
        case 5
          k = num_cells{i}(section);
          if k > 0
             soma_grid = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,OML_layer_min,OML_layer_mid,OML_layer_max);
             sz = size(soma_grid);
             soma_xyz_points = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
          end
        end
    end

    soma_uv_points = zeros(size(soma_xyz_points,1),2);
    for p = 1:size(soma_xyz_points,1)
        p
        % Limit GCL points tested to those near soma
        bin_number_soma     = ceil((soma_xyz_points(p,1)-xmin)/100);
        bin_start_surface   = bin_number_soma - 3;
        bin_end_surface     = bin_number_soma + 3;
        if bin_start_surface < 1
            bin_start_surface = 1;
        end
        if bin_end_surface > n_bins
            bin_end_surface = n_bins;
        end
        GCL_o_current = vertcat(GCL_o{bin_start_surface:bin_end_surface});
        GCL_i_current = vertcat(GCL_i{bin_start_surface:bin_end_surface});
    
        % Find closest point on the inner and outer GCL
        [k_o,d_o]   = dsearchn(GCL_o_current,Somata(i,:));
        [k_i,d_i]   = dsearchn(GCL_i_current,Somata(i,:));
        index_o     = find(M_o(:,1) == GCL_o_current(k_o,1) & M_o(:,2) == GCL_o_current(k_o,2) & M_o(:,3) == GCL_o_current(k_o,3));
        index_i     = find(M_i(:,1) == GCL_i_current(k_i,1) & M_i(:,2) == GCL_i_current(k_i,2) & M_i(:,3) == GCL_i_current(k_i,3));
        
        % Find u and v coordinates from closest points
        u_bin_o     = ceil(index_o/v_params(1,3));
        u_bin_i     = ceil(index_i/v_params(1,3));
        u_o         = u_params(1,1) + (u_bin_o - 1) * ((u_params(1,2)-u_params(1,1))/(u_params(1,3)-1));
        u_i         = u_params(1,1) + (u_bin_i - 1) * ((u_params(1,2)-u_params(1,1))/(u_params(1,3)-1));
        v_bin_o     = index_o - ((u_bin_o - 1) * v_params(1,3));
        v_bin_i     = index_i - ((u_bin_i - 1) * v_params(1,3));
        v_o         = v_params(1,1) + (v_bin_o - 1) * ((v_params(1,2)-v_params(1,1))/(v_params(1,3)-1));
        v_i         = v_params(1,1) + (v_bin_i - 1) * ((v_params(1, 2)-v_params(1,1))/(v_params(1,3)-1));

        if d_o > d_i
            soma_uv_points(p,1) = u_o;
            soma_uv_points(p,2) = v_o;
        else
            soma_uv_points(p,1) = u_i;
            soma_uv_points(p,2) = v_i;
        end
    end
    
    soma_locations{i} = horzcat(soma_xyz_points, soma_uv_points);

end

% Save somata to file
save('Outputs/Soma_Locations.mat','soma_locations','-v7.3');
