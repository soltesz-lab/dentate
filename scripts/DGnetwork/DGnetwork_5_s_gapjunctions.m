%function DGnetwork_5_s_gapjunctions(directory)

load('Outputs/Locations.mat')
gj_prob             = csvread('GJ_Connectivity.csv',1,1);
gj_weight           = csvread('GJ_Weights.csv',1,1);
gap_junctions       = cell(length(locations),length(locations));
counter             = 1;

% Choose compartment based on distance dependence of Fukuda
linear_coeff        = [-0.0315,9.4210];
HIPP_long_weights   = [37.5;112.5;187.5]*linear_coeff(1) + linear_coeff(2);
HIPP_short_weights  = [25;75;125]*linear_coeff(1) + linear_coeff(2);
Apical_weights      = [37.5;112.5;187.5;262.5]*linear_coeff(1) + linear_coeff(2);
Basal_weights       = [25;75;125;175]*linear_coeff(1) + linear_coeff(2);

for pre_type = 1:length(locations)
    rng(counter)
    counter = counter + 1;
    for post_type = 1:length(locations)
    % If there are gap junctions between these two types
        if gj_prob(pre_type,post_type) > 0
            % Get distances between each cell
            all_distances = pdist2(locations{pre_type}(:,1:3),locations{post_type}(:,1:3));

            % Remove duplicate gap junctions
            trimmed_distances = tril(all_distances,-1);

            % Eliminate gap junctions that are too far apart
            trimmed_distances(trimmed_distances>200) = 0;    

            % Get indices for those remaining and make new matrix
            [row,col,distance] = find(trimmed_distances);

            % Set distance-dependent probability (based on Otsuka)
            param   = [0.4201,-0.0019,150.7465,0.0255];
            prob    = param(1)+(param(2)-param(1))./(1+10.^((param(3)-distance)*param(4)));

            % Select connections based on weighted distance
            selected = datasample(transpose(1:length(distance)),round(gj_prob(pre_type,post_type)*length(distance)),'Weights',prob,'Replace',false);
                        
            if pre_type == 3
                % For HIPP cells, choose dendrites randomly
                dend1       = randi([0 3],length(selected),1);
                dend2       = randi([0 3],length(selected),1);
                
                % Keep track of which are long and short dendrites
                long1       = sum(dend1 <= 1);
                long2       = sum(dend2 <= 1);
                short1      = sum(dend1 >= 2);
                short2      = sum(dend2 >= 2);
                
                % Write blank compartment selections
                comp1       = zeros(length(selected),1);
                comp2       = zeros(length(selected),1);
                
                % Choose compartments for each type of dendrite
                long_comps  = randsample(transpose(0:2),long1+long2,true,HIPP_long_weights);
                short_comps = randsample(transpose(0:2),short1+short2,true,HIPP_short_weights);
                
                % Write compartments to appropriate variable
                comp1(dend1 <= 1) = long_comps(1:long1);
                comp1(dend1 >= 2) = short_comps(1:short1);
                comp2(dend2 <= 1) = long_comps(long1+1:end);
                comp2(dend2 >= 2) = short_comps(short1+1:end);
                
            elseif pre_type == 7
                % If NGFC involved, only choose distal apical
                dend1 = randi([0 3],length(selected),1);
                comp1 = 0*ones(length(selected),1);
                if post_type < 7
                    dend2 = randi([0 1],length(selected),1);
                    comp2 = 3*ones(length(selected),1);
                else
                    dend2 = randi([0 3],length(selected),1);
                    comp2 = 0*ones(length(selected),1);
                end
            else
                % For basket-type morphologies, choose either apical or basal
                dend1       = randi([0 3],length(selected),1);
                dend2       = zeros(length(selected),1);
                
                % Keep track of which are long and short dendrites
                apical1     = sum(dend1 <= 1);
                basal1      = sum(dend1 >= 2);
                
                % Choose corresponding apical or basal
                apical2     = randi([0 1],apical1,1);
                basal2      = randi([2 3],basal1,1);
                dend2(dend1<=1) = apical2;
                dend2(dend1>=2) = basal2;
                
                % Write blank compartment selections
                comp1       = zeros(length(selected),1);
                comp2       = zeros(length(selected),1);
                
                % Choose compartments for each type of dendrite
                apical_comps    = randsample(transpose(0:3),apical1,true,Apical_weights);
                basal_comps     = randsample(transpose(0:3),basal1,true,Basal_weights);
                
                % Write compartments to appropriate variable
                comp1(dend1 <= 1) = apical_comps(1:apical1);
                comp2(dend2 <= 1) = apical_comps(1:apical1);
                comp1(dend1 >= 2) = basal_comps(1:basal1);
                comp2(dend2 >= 2) = basal_comps(1:basal1);                
            end
            
            % Scale gap junction strength based on polynomial fit to Amatai distance-dependence
            params      = [0.0002,-0.0658,7.3211];
            weights     = params(1)*distance(selected).^2 + params(2)*distance(selected) + params(3);
            weights     = weights/max(weights);
            gj_strength = gj_weight(pre_type,post_type)/mean(weights) * weights;
            
            % Save selected to matrix
            gap_junctions{pre_type,post_type} = [row(selected)+size(vertcat(locations{1:pre_type-1}),1),col(selected)+size(vertcat(locations{1:pre_type-1}),1),dend1,comp1,dend2,comp2,gj_strength];
        end
    end
end

save('./Outputs/GJ_Connections.mat','gap_junctions','-v7.3');