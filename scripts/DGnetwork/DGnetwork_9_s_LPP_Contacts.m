function LPPCellModules = DGnetwork_9_s_LPP_Contacts(directory)

%% Distribution of putative LEC cell contacts in the inner and middle molecular layers of the dentate gyrus.
%% Assumptions:
%% 
%% - Approximately 52,000 LEC neurons in the rat, 67% fan/stellate cells.
%%   (Based on summary in Connectivity of the Hippocampus, Menno Witter 2007, but see supporting evidence.)
%% - Assuming similar modular organization as in MEC, tentative 10 modules with 3400 stellate cells each.
%%
%%


N_LPPCellsPerModule = 3400;
N_LPPCellModules = 10;
N_LPPCells = N_LPPCellModules * N_LPPCellsPerModule;
N_LPPCellLongExtent = 1000;

LPPCell_percent_start = 0.1;
LPPCell_percent_end = 0.9;
LPPCell_percent_step = (LPPCell_percent_end - LPPCell_percent_start) / N_LPPCells;

% Draw line to estimate septotemporal extent
[x,y,z]  = layer_eq_line(0,1000000);
u        = linspace(pi*1/100,pi*98/100,1000000);
sums      = zeros(length(x)-1,1);
for i = 1:length(x)-1
    d = pdist2([x(i),y(i),z(i)],[x(i+1),y(i+1),z(i+1)]);
    switch i
      case 1
        sums(i,1) = d;
      otherwise
        sums(i,1) = d + sums(i-1,1);
    end
end
stsums = [0;sums];

clearvars sums x y z

LPPCellModules = cell(N_LPPCellModules,1);
lppCellIndex = 0;

%% Determine LPP cell module sections
for lppModule = 1:N_LPPCellModules
    
    LPPCellModules{lppModule} = cell(N_LPPCellsPerModule,1);

    
    for lppCell = 1:N_LPPCellsPerModule

        lppCell
        
        %% Generate new set of LPP points every 40 cells
        if (mod(lppCell-1,40) == 0)

            X = cell(4,1);
            Y = cell(4,1);
            Z = cell(4,1);
            for sublayer = 1:4
                if sublayer > 1
                    [x_i,y_i,z_i]   = layer_eq_ML_poisson(sublayer-2);
                    [x_m,y_m,z_m]   = layer_eq_ML_poisson(sublayer-1.5);
                    switch sublayer
                      case 4
                        [x_o,y_o,z_o]   = layer_eq_ML_poisson(4);
                      otherwise
                        [x_o,y_o,z_o]   = layer_eq_ML_poisson(sublayer-1);
                    end
                    X{sublayer}     = [x_o;x_m;x_i];
                    Y{sublayer}     = [y_o;y_m;y_i];
                    Z{sublayer}     = [z_o;z_m;z_i];
                elseif sublayer == 1
                    [x_i,y_i,z_i]  	= layer_eq_GCL_poisson(-1.95);
                    [x_m,y_m,z_m]   = layer_eq_GCL_poisson(-1.0);
                    [x_o,y_o,z_o]   = layer_eq_GCL_poisson(0);
                    X{sublayer}     = [x_o;x_m;x_i];
                    Y{sublayer}     = [y_o;y_m;y_i];
                    Z{sublayer}     = [z_o;z_m;z_i];
                end
            end
            
            X_g = [X{3};X{4}];
            Y_g = [Y{3};Y{4}];
            Z_g = [Z{3};Z{4}];
            
            M = [X_g(:),Y_g(:),Z_g(:)];
            
        end
            
        percent = LPPCell_percent_start + LPPCell_percent_step*lppCellIndex
        width = N_LPPCellLongExtent;
        
        % Get septotemporal center
        [~,center_index]                = min(abs(stsums - (percent*max(stsums))));
        u_center                        = u(center_index);
        
        % Get points to make up plane
        plane_pts    = zeros(3,3);
        
        % Get longitudinal points
        [x_pt1,y_pt1,z_pt1]             = layer_eq_point(0,u_center-0.001,59.75*pi/100);
        [x_center,y_center,z_center]    = layer_eq_point(0,u_center,59.75*pi/100);
        [x_pt2,y_pt2,z_pt2]             = layer_eq_point(0,u_center+0.001,59.75*pi/100);
        plane_pts(1,:)                  = [x_center,y_center,z_center];
        direction                       = [x_pt2-x_pt1,y_pt2-y_pt1,z_pt2-z_pt1];
        unit_direction                  = direction/norm(direction);
        full_direction                  = unit_direction*width/2;
        plane_center1                   = [x_center - full_direction(1), y_center - full_direction(2), z_center + full_direction(3)];
        plane_center2                   = [x_center + full_direction(1), y_center + full_direction(2), z_center + full_direction(3)];
        
        % Get transverse points
        [x_pt1,y_pt1,z_pt1]             = layer_eq_point(0,u_center,59.75*pi/100-1);
        plane_pts(2,:)                  = [x_pt1,y_pt1,z_pt1];
        
        % Get layer points
        [x_pt1,y_pt1,z_pt1]             = layer_eq_point(-1,u_center,59.75*pi/100);
        plane_pts(3,:)                  = [x_pt1,y_pt1,z_pt1];
        
        % Get plane information
        vec1  = plane_pts(3,:) - plane_pts(1,:);
        vec2  = plane_pts(2,:) - plane_pts(1,:);
        plane = cross(vec1,vec2);
        start_d = sum(plane.*plane_center1);
        stop_d  = sum(plane.*plane_center2);
        
        LPPCellModules{lppModule}{lppCell} = M(find(M(:,1:3)*transpose(plane) + start_d < 0 & M(:,1:3)*transpose(plane) + stop_d > 0),:);
        lppCellIndex = lppCellIndex + 1;
    end
end

save(sprintf('%s/LPPCellModules.mat',directory),'LPPCellModules');
