% Determines which cells to make connections to

function DGnetwork_3_p_getconnections(input,directory)

if exist(sprintf('%s/Connections/%s.mat',directory,input),'file') == 2
else
    % Select cells to get distances for
    load('Outputs/Locations.mat')
    all_cells   = length(vertcat(locations{:}));
    subset_size = 50;
    start_cell  = subset_size*(str2double(input)-1)+1;
    if subset_size*str2double(input) < all_cells
        end_cell = subset_size*str2double(input);
    else
        end_cell = all_cells;
    end
    range = start_cell:end_cell;
    end_septal      = pi*-1.6/100;
    end_temporal    = pi*101/100;

    % Write out types so can keep track
    types = cell(length(locations),1);
    index_ends = zeros(length(locations),1);    
    for i = 1:length(locations)
        types{i} = ones(length(locations{i}),1,'int8')*i;
        switch i
            case length(locations)
                index_ends(i) = length(vertcat(locations{:}));
            otherwise
                index_ends(i) = length(vertcat(locations{1:i}));
        end
    end
    types = vertcat(types{:});

    
    % Import connectivity matrix
    num_synapses            = csvread('Syn_Connectivity.csv',1,1);
    axon_distributions      = csvread('Axon_Distributions.csv',2,1);
    synapse_locations       = textscan(fopen('Syn_Locations.csv'),'%s','Delimiter',',');
    synapse_locations       = transpose(reshape(synapse_locations{1},8,8));
    synapse_locations(1,:)  = [];
    synapse_locations(:,1)  = [];
    
    % Predetermine longitudinal approximations
    [x1,y1,z1]      = layer_eq_ML_line(0);
    u_distance_sums = zeros(length(x1)-1,1);
    for i = 1:length(x1)-1
        d = pdist2([x1(i),y1(i),z1(i)],[x1(i+1),y1(i+1),z1(i+1)]);
        switch i
            case 1
                u_distance_sums(i,1) = d;
            otherwise
                u_distance_sums(i,1) = d + u_distance_sums(i-1,1);
        end
    end
    u_distance_sums = [0;u_distance_sums];
    
    syn_pairs   = cell(length(range),length(locations));
    distances   = cell(length(range),length(locations));
    connections = cell(length(range),length(locations));
    counter = 1;
    
    % Get distances to all possible target cells
    for i = range
        rng(i)
        % Define presynaptic type and location
        pre_type        = types(i,1);
        if pre_type > 1
            cell_num = i - size(vertcat(locations{1:(pre_type-1)}),1);
        else
            cell_num = i;
        end
        
        % Keep track of presynaptic cell variables 
        u_start     = locations{pre_type}(cell_num,4);
        v_start     = locations{pre_type}(cell_num,5);
        
        for post_type = 1:length(locations)
            % If there are synaptic connections between these types
            if num_synapses(pre_type,post_type) > 0 && ~strcmp(synapse_locations{pre_type,post_type},'None')

                % Define layer where make synapse
                switch synapse_locations{pre_type,post_type}
                    case 'Hilus'
                        layer       = locations{post_type}(:,6);                      
                    case 'GCL'
                        layer       = -0.975;
                    case 'GCL/IML'
                        layer       = -0.475;
                    case 'IML'
                        layer       = 0.5;         
                    case 'MOML' 
                        layer       = 2;                       
                end

                % Filter based on longitudinal distance approximation
                cutoff_stdev    = 5;
                cutoff_distance = axon_distributions(pre_type,1) + cutoff_stdev*axon_distributions(pre_type,2);
                u_start_approx  = round((u_start - end_septal)/((end_temporal - end_septal)/10000));
                closest         = round((locations{post_type}(:,4) - end_septal)/((end_temporal - end_septal)/10000));
                long_distance   = abs(u_distance_sums(closest) - u_distance_sums(u_start_approx));
                M = long_distance < cutoff_distance;

                % Write out pre- and post-synaptic cells
                syn_pairs{counter,post_type}        = zeros(length(locations{post_type}),2,'single');
                distances{counter,post_type}        = zeros(length(locations{post_type}),2,'single');
                syn_pairs{counter,post_type}(:,1)   = i;
                
                if post_type == 1
                    syn_pairs{counter,post_type}(:,2)   = transpose(1:index_ends(post_type));
                else
                    syn_pairs{counter,post_type}(:,2)   = transpose((index_ends(post_type-1)+1):index_ends(post_type));
                end
                
                syn_pairs{counter,post_type}(~M,:)      = [];
                distances{counter,post_type}(~M,:)      = [];
                
                % Only calculate distance to those within limit
                u_end       = locations{post_type}(M,4);
                v_end       = locations{post_type}(M,5);
                layer_end   = locations{post_type}(M,6);

                % Get longitudinal distance by placing 100 points between cells
                num_points = 99;                
                u = zeros(size(locations{post_type}(M,:),1),num_points+1);
                v = ones(size(u))*v_start;    
                if length(layer) > 1
                    layer = repmat(locations{post_type}(M,6),1,(num_points+1));
                end            
                u(:,1) = u_start;            
                for point = 2:size(v,2)
                    u(:,point) = u_start + point*(u_end-u_start)/num_points;
                end            

                [x,y,z] = layer_eq(u,v,layer);
                distances{counter,post_type}(:,1) = sum(sqrt(diff(x,[],2).^2 + diff(y,[],2).^2 + diff(z,[],2).^2),2);   
                
                % Get trans distance
                switch pre_type
                    case 1
                        % Granule cells choose randomly within 250 micron transverse distance
                        if post_type == 1
                            % GC to GC connections are in the ML and use assumed normal distritbution
                            % Get transverse distance
                            v = zeros(size(u_end,1),num_points+1);
                            u = repmat(locations{post_type}(M,4),1,(num_points+1));
                            if length(layer) > 1
                                layer = repmat(locations{post_type}(M,6),1,(num_points+1));
                            end
                            v(:,1) = v_start;
                            for point = 2:size(v,2)
                                v(:,point) = v_start + point*(v_end-v_start)/num_points;
                            end

                            [x,y,z] = layer_eq(u,v,layer);
                            distances{counter,post_type}(:,2) = sum(sqrt(diff(x,[],2).^2 + diff(y,[],2).^2 + diff(z,[],2).^2),2);    
                            
                            % Remove cells that are outside of range
                            syn_pairs{counter,post_type}(distances{counter,post_type}(:,2) ==0,:) = [];
                            distances{counter,post_type}(distances{counter,post_type}(:,2) ==0,:) = [];
                            weights_M = normpdf(distances{counter,post_type}(:,1),axon_distributions(pre_type,1),axon_distributions(pre_type,2))...
                                      .*normpdf(distances{counter,post_type}(:,2),axon_distributions(pre_type,3),axon_distributions(pre_type,4));                            
                        else
                            % Granule cells choose randomly within 250 micron transverse distance
                            % Get presynaptic xyz
                            [x,y,z] = layer_eq(u_start,v_start,-1.95);
                            pre_xyz = [x,y,z];

                            % Get postsynaptic xyz based on presynaptic longitudinal position
                            [x,y,z] = layer_eq(u_start*ones(length(v_end),1),v_end,layer_end);

                            % Calculate transverse distance
                            d = transpose(pdist2(pre_xyz,[x,y,z]));
                            M = d < 250;
                            distances{counter,post_type}(M,2) = d(M);
                            
                            % Remove cells that are outside of range
                            syn_pairs{counter,post_type}(distances{counter,post_type}(:,2) ==0,:) = [];
                            distances{counter,post_type}(distances{counter,post_type}(:,2) ==0,:) = [];   
                            weights_M = normpdf(distances{counter,post_type}(:,1),axon_distributions(pre_type,1),axon_distributions(pre_type,2)); 
                        end                                               
                    case 2
                        % Mossy cells choose randomly for transverse distance, so dont calculate                        
                        distances{counter,post_type}(:,2) = 1;
                        weights_M = normpdf(distances{counter,post_type}(:,1),axon_distributions(pre_type,1),axon_distributions(pre_type,2));
                    otherwise
                        % For interneurons, get transverse distance
                        v = zeros(size(u_end,1),num_points+1);
                        u = repmat(locations{post_type}(M,4),1,(num_points+1));
                        if length(layer) > 1
                            layer = repmat(locations{post_type}(M,6),1,(num_points+1));
                        end
                        v(:,1) = v_start;
                        for point = 2:size(v,2)
                            v(:,point) = v_start + point*(v_end-v_start)/num_points;
                        end

                        [x,y,z] = layer_eq(u,v,layer);
                        distances{counter,post_type}(:,2) = sum(sqrt(diff(x,[],2).^2 + diff(y,[],2).^2 + diff(z,[],2).^2),2);     
                        
                        
                        % Remove cells that are outside of range
                        syn_pairs{counter,post_type}(distances{counter,post_type}(:,2) ==0,:) = [];
                        distances{counter,post_type}(distances{counter,post_type}(:,2) ==0,:) = [];
                        weights_M = normpdf(distances{counter,post_type}(:,1),axon_distributions(pre_type,1),axon_distributions(pre_type,2))...
                                  .*normpdf(distances{counter,post_type}(:,2),axon_distributions(pre_type,3),axon_distributions(pre_type,4));
                end
                                
                if length(weights_M) > num_synapses(pre_type,post_type)
                    if ~isinteger(num_synapses(pre_type,post_type))
                        rand_num = randi([1 2]);
                        if rand_num == 1
                            num_connections = floor(num_synapses(pre_type,post_type));
                        else
                            num_connections = floor(num_synapses(pre_type,post_type));
                        end
                    else
                        num_connections = num_synapses(pre_type,post_type);
                    end
                    selected                        = transpose(datasample(1:length(weights_M),num_connections,'Weights',double(weights_M),'Replace',false));
                    connections{counter,post_type}  = [syn_pairs{counter,post_type}(selected,:) distances{counter,post_type}(selected,:)];
                else
                    connections{counter,post_type}  = [syn_pairs{counter,post_type} distances{counter,post_type}];
                end
            end
        end
        counter = counter + 1;
    end
    
    % Save possible connections to file
    %save(sprintf('%s/Syn_Pairs/%s.mat',directory,input),'syn_pairs','-v6');
    %save(sprintf('%s/Distances/%s.mat',directory,input),'distances','-v6');
    %save(sprintf('%s/Weights/%s.mat',directory,input),'weights','-v6');
    save(sprintf('%s/Connections/%s.mat',directory,input),'connections','-v6');
end