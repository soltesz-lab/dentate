
function DGnetwork_4_s_combineconnections(directory)

% Load locations so know how much of each cell type
load('Outputs/Locations.mat')
split           = 50;
connection_M = cell(size(locations,1),size(locations,1));

% Keep track of starting and ending indices
indices = zeros(size(locations,1),2);
for type = 1:size(locations,1)
    switch type
        case 1
            indices(type,1) = 1;
            indices(type,2) = size(locations{type},1);
        otherwise
            indices(type,1) = size(vertcat(locations{1:type-1}),1) + 1;
            indices(type,2) = size(vertcat(locations{1:type}),1);
    end
end
files_i = ceil(indices/split);

for type = 1:size(locations,1)
    start   = files_i(type,1);
    stop    = files_i(type,2);
    
    % Load in all files for given type
    M = cell(files_i(type,2)-files_i(type,1)+1,1);
    file_counter = 1;
    for i = start:stop
        if mod(i,100) == 0
            display(i)
        end
        file    = sprintf('%s/Connections/%i.mat',directory,i);
        load(file);
        M{file_counter}    = connections;
        file_counter = file_counter + 1;
    end 
    all_connections = vertcat(M{:});
    clearvars M
    
    % Select total from all possible weighted connections
    for post = 1:length(locations)
        connection_M{type,post} = vertcat(all_connections{:,post});
    end
end

save('./Outputs/Syn_Connections.mat','connection_M','-v7.3');
        