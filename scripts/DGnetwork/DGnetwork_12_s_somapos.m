
function [soma_points] =  DGnetwork_12_s_somapos(output_path)


% layer boundaries
HL_layer_min       = -3.95;
HL_layer_mid       = -2.95;
HL_layer_max       = -1.95;

GCL_layer_min       = -1.95;
GCL_layer_mid       = -1;
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

% Define points used for nearest-neighbor queries
[x_mid,y_mid,z_mid]   = layer_eq_GCL_2(GCL_layer_mid);
[x_min,y_min,z_min]   = layer_eq_GCL_2(GCL_layer_min);
[x_max,y_max,z_max]   = layer_eq_GCL_2(GCL_layer_max);
GCL_x                 = [x_min;x_mid;x_max];
GCL_y                 = [y_min;y_mid;y_max];
GCL_z                 = [z_min;z_mid;z_max];
GCL_pts               = [GCL_x(:),GCL_y(:),GCL_z(:)];
clear GCL_x GCL_y GCL_z 

[x_mid,y_mid,z_mid]   = layer_eq_GCL_2(HL_layer_mid);
[x_min,y_min,z_min]   = layer_eq_GCL_2(HL_layer_min);
[x_max,y_max,z_max]   = layer_eq_GCL_2(HL_layer_max);
HL_x                 = [x_min;x_mid;x_max];
HL_y                 = [y_min;y_mid;y_max];
HL_z                 = [z_min;z_mid;z_max];
HL_pts               = [HL_x(:),HL_y(:),HL_z(:)];
clear HL_x HL_y HL_z 

[x_mid,y_mid,z_mid]   = layer_eq_ML(IML_layer_mid);
[x_min,y_min,z_min]   = layer_eq_ML(IML_layer_min);
[x_max,y_max,z_max]   = layer_eq_ML(IML_layer_max);
IML_x                 = [x_min;x_mid;x_max];
IML_y                 = [y_min;y_mid;y_max];
IML_z                 = [z_min;z_mid;z_max];
IML_pts               = [IML_x(:),IML_y(:),IML_z(:)];
clear IML_x IML_y IML_z 

[x_mid,y_mid,z_mid]   = layer_eq_ML(MML_layer_mid);
[x_min,y_min,z_min]   = layer_eq_ML(MML_layer_min);
[x_max,y_max,z_max]   = layer_eq_ML(MML_layer_max);
MML_x                 = [x_min;x_mid;x_max];
MML_y                 = [y_min;y_mid;y_max];
MML_z                 = [z_min;z_mid;z_max];
MML_pts               = [MML_x(:),MML_y(:),MML_z(:)];
clear MML_x MML_y MML_z 

[x_mid,y_mid,z_mid]   = layer_eq_ML(OML_layer_mid);
[x_min,y_min,z_min]   = layer_eq_ML(OML_layer_min);
[x_max,y_max,z_max]   = layer_eq_ML(OML_layer_max);
OML_x                 = [x_min;x_mid;x_max];
OML_y                 = [y_min;y_mid;y_max];
OML_z                 = [z_min;z_mid;z_max];
OML_pts               = [OML_x(:),OML_y(:),OML_z(:)];
clear OML_x OML_y OML_z 

clear x_mid x_min x_max y_mid y_min y_max z_mid z_min z_max 

% Define granule cell layer parameters from layer_eq_GCL
GCL_u_params   = [pi*1/100,pi*98/100,4000];
GCL_v_params   = [pi*-23/100,pi*142.5/100,4000];

ML_u_params = [pi*-1.6/100,pi*101/100,4000];
ML_v_params = [pi*-23/100,pi*142.5/100,4000];


HL_ns = createns(HL_pts,'nsmethod','kdtree','BucketSize',500);
clear HL_pts
GCL_ns = createns(GCL_pts,'nsmethod','kdtree','BucketSize',500);
clear GCL_pts
IML_ns = createns(IML_pts,'nsmethod','kdtree','BucketSize',500);
clear IML_pts
MML_ns = createns(MML_pts,'nsmethod','kdtree','BucketSize',500);
clear MML_pts
OML_ns = createns(OML_pts,'nsmethod','kdtree','BucketSize',500);
clear OML_pts

knn = 15;
for i = 2:num_types
    i
    soma_xyz_points = [];
    soma_uv_points  = [];
    for section = 1:5
        section
        switch section
        case 1
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_GCL,HL_layer_min,HL_layer_mid,HL_layer_max);
             sz                = size(soma_grid);
             soma_xyz_points   = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
             [index_nn,d_nn]   = knnsearch(GCL_ns,soma_xyz_points,'K',knn);
             uv_points         = nearest_uv(knn,index_nn,GCL_u_params,GCL_v_params);
             soma_uv_points    = vertcat(soma_uv_points, uv_points);
          end
        case 2
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_GCL,GCL_layer_min,GCL_layer_mid,GCL_layer_max);
             sz                = size(soma_grid);
             soma_xyz_points   = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
             uv_points         = nearest_uv(knn,index_nn,GCL_u_params,GCL_v_params);
             soma_uv_points    = vertcat(soma_uv_points, uv_points);
          end
        case 3
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,IML_layer_min,IML_layer_mid,IML_layer_max);
             sz                = size(soma_grid);
             soma_xyz_points   = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
             uv_points         = nearest_uv(knn,index_nn,ML_u_params,ML_v_params);
             soma_uv_points    = vertcat(soma_uv_points, uv_points);
          end
        case 4
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,MML_layer_min,MML_layer_mid,MML_layer_max);
             sz                = size(soma_grid);
             soma_xyz_points   = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
             uv_points         = nearest_uv(knn,index_nn,ML_u_params,ML_v_params);
             soma_uv_points    = vertcat(soma_uv_points, uv_points);
          end
        case 5
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DGnetwork_11_s_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,OML_layer_min,OML_layer_mid,OML_layer_max);
             sz                = size(soma_grid);
             soma_xyz_points   = vertcat(soma_xyz_points,soma_grid(randsample(sz(1),k),:));
             uv_points         = nearest_uv(knn,index_nn,ML_u_params,ML_v_params);
             soma_uv_points    = vertcat(soma_uv_points, uv_points);
          end
        end
    end

    locs = horzcat(soma_xyz_points, soma_uv_points);
    % sort locations according to u coordinate
    [y,sortidx]=sort(locs(:,4));
    soma_locations{i} = locs(sortidx,:);

    clear locs sortidx soma_xyz_points soma_uv_points
end

% Save somata to file
save(output_path,'soma_locations','-v7.3');
