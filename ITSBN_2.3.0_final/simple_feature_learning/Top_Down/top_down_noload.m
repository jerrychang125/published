%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Uses the model struct in the workspace to visualize the filters in pixel
% space from models above by placing a single
% one in each of the top feature maps independtly and then reconstructing 
% downwards. This is done from the
% top model that is defined in the model struct's fullmodelpath field. All other
% layers will be loaded for their last trained epoch.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @outparam \li \e layer the top layer of the model.
% \li \e recon_z# the top layer's feature maps.
% \li \e pooled_indices# the top layer and below layers indices of Max pooling
% (if used).
% \li \e model# the model struct for each layer containing all parameters of the layers.
% Particularly norm_types, norm_sizes, xdim, ydim, filter_size, z0_filter_size,
% TRAIN_Z0, conmats, num_feature_maps
% \li \e F# the filters for a given layer.
% \li \e z0# the z0 maps for a given layer.
% \li \e COMP_THREADS the number of computation threads you want to use.
%
% @top_down_file @copybrief top_down_noload.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear recon_y1; % Since this is resized below the next itreation would fail if this is not cleared.

%%%%%%%%%%%%%%%%%%%%%%%%%%%model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define the z2 you want to reconstruct
% The third dimension of z2 can be ignored since we only want to
% reconstruct a single set of image planes, y1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the size of the top layer (for one training sample)
% if(exist('zsample','var'))
    eval(strcat('top_size = size(recon_z',num2str(layer),');'));
%     top_size = size(zsample);
% else
%     top_size = size(zsample2);
% end

% Get the middle index to place the point the top feature maps.
% middlez_x = floor(model.xdim/2+model.filter_size(model.layer)/2)
% middlez_y = floor(model.ydim/2+model.filter_size(model.layer)/2)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get middles of each feature maps at the top layer.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch(model.norm_types{layer+1})
    case 'Max'  % Max Pooling
        % If pooled at top, then filter adjustment is on unpooled features
        middlez_x = floor(top_size(1)/model.norm_sizes{layer+1}(1));
        middlez_y = floor(top_size(2)/model.norm_sizes{layer+1}(2));
    case 'Avg'  % Average Pooling
        % If pooled at top, then filter adjustment is on unpooled features
        middlez_x = floor(top_size(1)/model.norm_sizes{layer+1}(1));
        middlez_y = floor(top_size(2)/model.norm_sizes{layer+1}(2));      
    case 'None'
        % Get the middle index to place the point the top feature maps.
        middlez_x = floor(model.xdim/2+model.filter_size(model.layer)/2);
        middlez_y = floor(model.ydim/2+model.filter_size(model.layer)/2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Place a single point in each of the top layer feature maps.
for top_feature_map=1:model.num_feature_maps(layer)
%     fprintf('Processing feature map: %d out of %d\n',top_feature_map,model.num_feature_maps(layer));
    
    % Make the map zero to start.
    eval(strcat('recon_z',num2str(layer),' = zeros(top_size,',char(39),'single',char(39),');'));
    
    %Place the single piont in each z2 filter map.
    eval(strcat('recon_z',num2str(layer),'(middlez_x,middlez_y,top_feature_map) = 1;'));
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Reconstruct from the top down.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Do the layer by layer reconstructions from the top (ending with
    % recon_z0 as the input image).
    top_down_core
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Save a copy of the pixel space for each feature map pass.
    recon_y1(:,:,:,top_feature_map) = recon_z0;
end


% Cut off the unneeded size of the pixel space reconstructions
% Get the center coordinates of the reconstructed image planes.
middlex = floor(size(recon_y1,1)/2);
middley = floor(size(recon_y1,2)/2);

% For each layer add the filter sizes up.
summed_filter_sizes = sum(model.filter_size(1:layer));
% Add one for each layer (to be safe) then take half.
summed_filter_sizes = floor((summed_filter_sizes)/2);

% Get start an end indices for each image plane to do the crop.
startx = middlex-summed_filter_sizes;
starty = middley-summed_filter_sizes;
endx = middlex+summed_filter_sizes;
endy = middley+summed_filter_sizes;

% Crop the reconstructions.
recon_y1 = recon_y1(startx:endx,starty:endy,:,:);
