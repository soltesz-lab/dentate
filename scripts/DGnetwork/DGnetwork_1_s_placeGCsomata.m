% Place GC Somata

%function DGnetwork_1_s_findsomata()

% Set random number generator for reproducibility
rng(1)
time_out = 60*60*4;

%% Define number and distribution of granule cells
num_cells       = 1000000;
septotemporal   = [0.235362931;0.259838929;0.273841016;0.282544828;0.301903448;0.35925;0.366409755;0.30202677;0.243310322;0.234742015;0.193152087;0.104103539];
distribution    = septotemporal/sum(septotemporal(:,1));


%% Determine splits

% Determine total length
[x1,y1,z1]  = layer_eq_GCL_line(0);
sums        = zeros(length(x1)-1,1);
for type = 1:length(x1)-1
    d = pdist2([x1(type),y1(type),z1(type)],[x1(type+1),y1(type+1),z1(type+1)]);
    switch type
        case 1
            sums(type,1) = d;
        otherwise
            sums(type,1) = d + sums(type-1,1);
    end
end

% Determine where GC splits should be and corresponding u coordinate
num_GC_sections = 12;
num_GC_splits   = num_GC_sections - 1;
splits_GC       = zeros(num_GC_splits,1);
end_septal      = pi*1/100;
end_temporal    = pi*98/100;
v_min           = pi*-23/100;
v_max           = pi*142.5/100;
u_var           = transpose(linspace(end_septal,end_temporal,10000));
for type = 1:num_GC_splits
    total_length        = sums(end,1);
    current_split       = total_length * type/num_GC_sections;
    split_point         = find(sums(:,1)>current_split,1,'first');
    splits_GC(type,1)   = u_var(split_point,1);
end

%% Distribute Granule Cells

% Set parameters
layer_min       = -1.95;
layer_max       = 0;    
type            = 1;
section         = 2;
x_min           = -3200;
x_max           = 3200;
bin_size        = 15;

% Draw sphere
n = 20;
theta = (-n:2:n)/n*pi;
phi = (-n:2:n)'/n*pi/2;
cosphi = cos(phi); cosphi(1) = 0; cosphi(n+1) = 0;
sintheta = sin(theta); sintheta(1) = 0; sintheta(n+1) = 0;
x_sphere = cosphi*cos(theta);
y_sphere = cosphi*sintheta;
z_sphere = sin(phi)*ones(1,n+1);
% Keep track of duplicate points
soma_xyz = [reshape(x_sphere,[],1),reshape(y_sphere,[],1),reshape(z_sphere,[],1)];
[~,keep,~] = unique(soma_xyz,'rows');

GC_locations = cell(num_GC_sections,1);
edge_somata  = cell(num_GC_sections,1);
edge_ellipsoids = cell(num_GC_sections,1);

for j = 1:num_GC_sections
    prev_somata     = cell(ceil((x_max-x_min)/bin_size),1);
    % Define splits and total number of cells to place
    if j == 1
        split1              = end_septal;
        split2              = splits_GC(1,1);
        total_cells         = round(num_cells * distribution(j,1));
        num_edge_cells      = 0;
        prev_ellipsoids     = cell(total_cells,1);   
        edge_ellipsoids{j}  = cell(total_cells,1);
        prev_section_ellipsoids = zeros(1,3);
    elseif j == 12
        split1          = splits_GC(end,1);
        split2          = end_temporal;  
        % Add remainder to last section
        total_cells     = num_cells - size(vertcat(GC_locations{1:(num_GC_sections-1)}),1);
        num_edge_cells                      = size(edge_somata{j-1},1);
        prev_ellipsoids                     = cell(total_cells,1);
        prev_ellipsoids                     = vertcat(edge_ellipsoids{j-1},prev_ellipsoids);
        edge_ellipsoids{j}                  = cell(total_cells,1);
        prev_section_ellipsoids = vertcat(edge_ellipsoids{j-1});
        prev_section_ellipsoids = vertcat(prev_section_ellipsoids{:});         
    else
        split1                              = splits_GC(j-1,1);
        split2                              = splits_GC(j,1);  
        total_cells                         = round(num_cells * distribution(j,1));
        num_edge_cells                      = size(edge_somata{j-1},1);
        prev_ellipsoids                     = cell(total_cells,1);
        prev_ellipsoids                     = vertcat(edge_ellipsoids{j-1},prev_ellipsoids);
        edge_ellipsoids{j}                  = cell(total_cells,1);
        prev_section_ellipsoids = vertcat(edge_ellipsoids{j-1});
        prev_section_ellipsoids = vertcat(prev_section_ellipsoids{:});        
    end
    
    tic
    display(total_cells)
    GC_locations{j} = zeros(total_cells,8);
    for cell_num = 1:total_cells
        if mod(cell_num,5000) == 0
            display(cell_num)
            toc
        end
        if toc < time_out
            prev_ellipsoids{cell_num+num_edge_cells} = zeros(382,3);
            % Use Claiborne average length and diameter     
            soma_diam1 = 10.3;
            soma_length1 = 18.6;
            max_soma_radius  = max(soma_diam1,soma_length1)/2;
            min_soma_radius  = min(soma_diam1,soma_length1)/2;
            
            while true
                % Choose a random point
                random_uvl = zeros(1,3);
                random_uvl(:,1) = split1 + (split2 - split1)*rand(1,1);
                random_uvl(:,2) = v_min + (v_max-v_min)*rand(1,1);
                random_uvl(:,3) = layer_min + (layer_max-layer_min)*rand(1,1);
                [x,y,z] = layer_eq(random_uvl(:,1),random_uvl(:,2),random_uvl(:,3));
                pt = [x,y,z];         
                bin_num = ceil((pt(1)-x_min)/bin_size);

                % Find already placed somata that are close
                if ~isempty(prev_somata)
                    close_somata    = vertcat(prev_somata{(bin_num-1):(bin_num+1)});
                else
                    close_somata = [];
                end
                if ~isempty(close_somata)
                    max_radius = max(close_somata(:,7:8),[],2);
                    close_somata = close_somata(close_somata(:,1) > (pt(1) - (max_radius+max_soma_radius)) & close_somata(:,1) < (pt(1) + (max_radius+max_soma_radius)),:);
                end
                % If there are close somata
                if ~isempty(close_somata)
                    max_radius = max(close_somata(:,7:8),[],2);
                    close_somata = close_somata(close_somata(:,2) > (pt(2) - (max_radius+max_soma_radius)) & close_somata(:,2) < (pt(2) + (max_radius+max_soma_radius)),:);
                    if ~isempty(close_somata)
                        max_radius = max(close_somata(:,7:8),[],2);
                        close_somata = close_somata(close_somata(:,3) > (pt(3) - (max_radius+max_soma_radius)) & close_somata(:,3) < (pt(3) + (max_radius+max_soma_radius)),:);
                    end
                    % If there are even closer somata
                    if ~isempty(close_somata)

                        % Load in ellipsoid points
                        all_pts = zeros(size(close_somata,1)*382,4);
                        for soma = 1:size(close_somata,1)
                            all_pts(((382*(soma-1)+1):(382*soma)),1:3)  = prev_ellipsoids{close_somata(soma,9)};
                        end

                        % Get distance from ellipsoid points to soma
                        d_closesomata   = sqrt((all_pts(:,1)-pt(1)).^2 + (all_pts(:,2)-pt(2)).^2 + (all_pts(:,3)-pt(3)).^2);
                        closest_pts     = all_pts(d_closesomata < min_soma_radius,:);

                        % If there arent points within the smaller radius
                        if isempty(closest_pts)
                            close_pts       = all_pts(d_closesomata < max_soma_radius,:);
                            % If there are points within the bigger radius
                            if ~isempty(close_pts)
                                % Unrotate points and test if close points are inside ellipse
                                moved_close_pts = [close_pts(:,1)-x,close_pts(:,2)-y,close_pts(:,3)-z];
                                [x_gcl,y_gcl,z_gcl] = layer_eq(random_uvl(:,1),random_uvl(:,2),0);                            
                                direction         = [x_gcl-x,y_gcl-y,z_gcl-z];
                                unit_direction    = direction/norm(direction);
                                r       = vrrotvec([0 0 1],[unit_direction(:,1) unit_direction(:,2) unit_direction(:,3)]);
                                R1      = vrrotvec2mat(r);
                                rotated_close_pts = moved_close_pts * R1;
                                IN_soma = (rotated_close_pts(:,1).^2)/((soma_diam1/2)^2) + (rotated_close_pts(:,2).^2)/((soma_diam1/2)^2)  + (rotated_close_pts(:,1).^2)/((soma_length1/2)^2)  < 1;
                            else
                                IN_soma     = 0;
                            end
                        else
                            IN_soma     = 1;
                        end
                    else
                        IN_soma     = 0;                 
                    end                                
                else
                    IN_soma = 0;
                end

                % If there arent any close somata that overlap
                if any(IN_soma) == 0;
                    %  Make sure doesn't overlap with previous section ellipsoids
                    if random_uvl(1) < (split1 + (split2-split1)/20)
                        % Unrotate points and test if close points are inside ellipse
                        d_closesomata   = sqrt((prev_section_ellipsoids(:,1)-pt(1)).^2 + (prev_section_ellipsoids(:,2)-pt(2)).^2 + (prev_section_ellipsoids(:,3)-pt(3)).^2);
                        trimmed_pts     = prev_section_ellipsoids(d_closesomata < max_soma_radius*2,:);
                        moved_close_pts = [trimmed_pts(:,1)-x,trimmed_pts(:,2)-y,trimmed_pts(:,3)-z];
                        rotated_close_pts = moved_close_pts * R1;
                        IN_prev_section = (rotated_close_pts(:,1).^2)/((soma_diam1/2)^2) + (rotated_close_pts(:,2).^2)/((soma_diam1/2)^2)  + (rotated_close_pts(:,1).^2)/((soma_length1/2)^2)  < 1;
                    else
                        IN_prev_section = 0;
                    end
                    
                    if any(IN_prev_section) == 0;
                        % Write out location information
                        GC_locations{j}(cell_num,:)  = [pt random_uvl soma_diam1 soma_length1];
                        prev_somata{bin_num} = [prev_somata{bin_num};pt random_uvl soma_diam1 soma_length1 cell_num];

                        % Keep track of ellipsoid points
                        X_soma                  = x_sphere*soma_diam1/2;
                        Y_soma                  = y_sphere*soma_diam1/2;
                        Z_soma                  = z_sphere*soma_length1/2;
                        soma_xyz                = [reshape(X_soma,[],1),reshape(Y_soma,[],1),reshape(Z_soma,[],1)];
                        soma_xyz                = soma_xyz(keep,:);
                        [x_gcl,y_gcl,z_gcl]     = layer_eq(random_uvl(:,1),random_uvl(:,2),0);
                        direction               = [x_gcl-x,y_gcl-y,z_gcl-z];
                        unit_direction          = direction/norm(direction);
                        r                       = vrrotvec([unit_direction(:,1) unit_direction(:,2) unit_direction(:,3)],[0 0 1]);
                        R1                      = vrrotvec2mat(r);
                        soma_xyz                = soma_xyz * R1;
                        soma_xyz                = [soma_xyz(:,1)+x,soma_xyz(:,2)+y,soma_xyz(:,3)+z];
                        prev_ellipsoids{cell_num} = soma_xyz;

                        % If on the right edge
                        if j < 12
                            if random_uvl(1) > (split1 + (split2-split1)*19/20)
                                edge_somata{j}(cell_num,:) = [pt random_uvl soma_diam1 soma_length1];
                                edge_ellipsoids{j}{cell_num} = soma_xyz;
                            end                    
                        end
                    
                        break
                    end

                end
            end
        end
    end
    if j < 12
        edge_somata{j}(edge_somata{j}(:,1)==0,:) = [];
        edge_ellipsoids{j} = edge_ellipsoids{j}(~cellfun('isempty',edge_ellipsoids{j}));
    end
    GC_locations{j}(GC_locations{j}(:,1)==0,:) = [];
end

GC_locations_all = vertcat(GC_locations{:});
save('./Outputs/GC_Locations.mat','GC_locations_all','-v7.3');