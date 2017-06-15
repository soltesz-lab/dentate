
function [soma_points] =  DG_somapos_sample(output_path)

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
N_GC       = 1;
N_MC       = 2;
N_HIPP     = 3;
N_PVBC     = 4;
N_AA       = 5;
N_HICAP    = 6;
N_NGFC     = 7;
N_IS       = 8;
N_MOPP     = 9;
N_MEC      = 10;
N_LEC      = 11;

num_types       = 11;
num_cells       = cell(num_types,1);
cell_names      = cell(num_types,1);
soma_disth      = zeros(num_types,1);
soma_distv      = zeros(num_types,1);
soma_border     = zeros(num_types,1);
distribution    = cell(num_types,1);
soma_points     = cell(num_types,1);
soma_locations  = cell(num_types,1);

%2 GC
cell_names{N_GC}   = 'GC';
num_cells{N_GC}    = [0;1000000;0;0;0;1000000]; %Hilus;GCL;IML;MML;OML;Total
soma_disth(N_GC)   = 20;
soma_distv(N_GC)   = 10;
septotemporal      = [20.8;38.2;67.6;75.4;51.2;48.6;84.1;68.5;145.7;385.0];
distribution{N_GC} = septotemporal/sum(septotemporal(:,1));

%2 Mossy
cell_names{N_MC}   = 'MC';
num_cells{N_MC}    = [30000;0;0;0;0;30000]; %Hilus;GCL;IML;MML;OML;Total
soma_disth(N_MC)   = 20;
soma_distv(N_MC)   = 20;
septotemporal      = [20.8;38.2;67.6;75.4;51.2;48.6;84.1;68.5;145.7;385.0];
distribution{N_MC} = septotemporal/sum(septotemporal(:,1));

%3 HIPP
cell_names{N_HIPP}   = 'HC';
num_cells{N_HIPP}    = [9000;0;0;0;0;9000];
soma_disth(N_HIPP)  = 50;
soma_distv(N_HIPP)  = 50;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_HIPP} = septotemporal/sum(septotemporal(:,1));

%4 PVBC
cell_names{N_PVBC}   = 'BC';
num_cells{N_PVBC}    = [1700;1700;400;0;0;3800];
soma_disth(N_PVBC)   = 80;
soma_distv(N_PVBC)   = 80;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{N_PVBC} = septotemporal/sum(septotemporal(:,1));

%5 AA 
cell_names{N_AA}   = 'AAC';
num_cells{N_AA}    = [200;200;50;0;0;450];
soma_disth(N_AA)   = 100;
soma_distv(N_AA)   = 100;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{N_AA} = septotemporal/sum(septotemporal(:,1));

%6 HICAP/CCK+ Basket Cells
cell_names{N_HICAP}   = 'HCC';
num_cells{N_HICAP}    = [1150;250;0;0;0;1400];
soma_disth(N_HICAP)   = 100;
soma_distv(N_HICAP)   = 100;
septotemporal   = [9.1;16.3;13.1;11.3;9.3;8.3;8.3;9.9;9.8;6.1];
distribution{N_HICAP} = septotemporal/sum(septotemporal(:,1));

%7 NGFC
cell_names{N_NGFC}   = 'NGFC';
num_cells{N_NGFC}    = [0;0;0;2500;2500;5000];
soma_disth(N_NGFC)   = 50;
soma_distv(N_NGFC)   = 50;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{N_NGFC} = septotemporal/sum(septotemporal(:,1));

%8 IS
cell_names{N_IS}   = 'IS';
num_cells{N_IS}    = [3000;0;0;0;0;3000];
soma_disth(N_IS)   = 50;
soma_distv(N_IS)   = 50;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_IS} = septotemporal/sum(septotemporal(:,1));

%9 MOPP
cell_names{N_MOPP}   = 'MOPP';
num_cells{N_MOPP}    = [0;0;2000;500;500;3000];
soma_disth(N_MOPP)   = 80;
soma_distv(N_MOPP)   = 80;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_MOPP} = septotemporal/sum(septotemporal(:,1));

%10 MEC
cell_names{N_MEC}   = 'MEC';
num_cells{N_MEC}    = [0;0;0;38000;0;38000];
soma_disth(N_MEC)   = 20;
soma_distv(N_MEC)   = 20;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_MEC} = septotemporal/sum(septotemporal(:,1));

%11 LEC
cell_names{N_LEC}   = 'LEC';
num_cells{N_LEC}    = [0;0;0;0;34000;34000];
soma_disth(N_LEC)   = 20;
soma_distv(N_LEC)   = 20;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{N_LEC} = septotemporal/sum(septotemporal(:,1));

for i = 1:11
    i
    rng(i);
    sampled_xyz_points = [];
    for section = 1:5
        section
        switch section
        case 1
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DG_somagrid(soma_disth(i),soma_distv(i),@layer_eq_GCL_2,HL_layer_min,HL_layer_mid,HL_layer_max);
             sz                = size(soma_grid);
             sel_xyz_points    = soma_grid(randsample(sz(1),k),:);
	     sampled_xyz_points = vertcat(sampled_xyz_points, sel_xyz_points);
          end
        case 2
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DG_somagrid(soma_disth(i),soma_distv(i),@layer_eq_GCL_2,GCL_layer_min,GCL_layer_mid,GCL_layer_max);
             sz                = size(soma_grid)
             sel_xyz_points    = soma_grid(randsample(sz(1),k),:);
	     sampled_xyz_points = vertcat(sampled_xyz_points, sel_xyz_points);
          end
        case 3
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DG_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,IML_layer_min,IML_layer_mid,IML_layer_max);
             sz                = size(soma_grid);
             sel_xyz_points    = soma_grid(randsample(sz(1),k),:);
	     sampled_xyz_points = vertcat(sampled_xyz_points, sel_xyz_points);
          end
        case 4
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DG_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,MML_layer_min,MML_layer_mid,MML_layer_max);
             sz                = size(soma_grid);
             sel_xyz_points    = soma_grid(randsample(sz(1),k),:);
	     sampled_xyz_points = vertcat(sampled_xyz_points, sel_xyz_points);
          end
        case 5
          k = num_cells{i}(section);
          if k > 0
             soma_grid         = DG_somagrid(soma_disth(i),soma_distv(i),@layer_eq_ML,OML_layer_min,OML_layer_mid,OML_layer_max);
             sz                = size(soma_grid);
             sel_xyz_points    = soma_grid(randsample(sz(1),k),:);
	     sampled_xyz_points = vertcat(sampled_xyz_points, sel_xyz_points);
          end
        end
    end

    h5_output_path = sprintf('%s.h5',output_path);
    sz = size(sampled_xyz_points,1);
    h5create(h5_output_path, sprintf('/%s/X Coordinate',cell_names{i}), [sz 1], 'Datatype', 'double', 'ChunkSize', [min(sz,1000) 1], 'Deflate', 9);
    h5create(h5_output_path, sprintf('/%s/Y Coordinate',cell_names{i}), [sz 1], 'Datatype', 'double', 'ChunkSize', [min(sz,1000) 1], 'Deflate', 9);
    h5create(h5_output_path,sprintf('/%s/Z Coordinate',cell_names{i}), [sz 1], 'Datatype', 'double', 'ChunkSize', [min(sz,1000) 1], 'Deflate', 9);
    
    h5write(h5_output_path, sprintf('/%s/X Coordinate',cell_names{i}), sampled_xyz_points(:,1));
    h5write(h5_output_path, sprintf('/%s/Y Coordinate',cell_names{i}), sampled_xyz_points(:,2));
    h5write(h5_output_path, sprintf('/%s/Z Coordinate',cell_names{i}), sampled_xyz_points(:,3));
    
    clear sampled_xyz_points soma_grid
end

