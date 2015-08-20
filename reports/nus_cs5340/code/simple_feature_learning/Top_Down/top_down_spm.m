%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Visualizes the dictionary elements of the spatial pyramid matching kernel's
% dictionary in pixel space. Since the dictionary is built from top layer 
% feature activation, they must be passed down through the model to be visualized
% in pixel space. This is done from the
% top model that is defined in the model struct's fullmodelpath field. All other
% layers will be loaded for their last trained epoch.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inparam 
% \li \e dict_name the name of the dictionary to load
% \li \e pyramid_pool_type the type of pooling of feature maps used to build the descriptors.
% \li \e pyramid_pool_size the size of pooling region used to build the descriptors.
% \li \e model is set in here with the following fields: PLOT_RESULTS (if and how often to plot results),
% SAVE_RESULTS (if and how often to save results), comp_threads (number of 
% computation threads), fullmodelpath (where to load the Deconv Net),
% fulldatapath (where to get the dictionary from), fullsavepath (where to save
% the results to), machine (either ' local_server' or ' laptop').
%
% @top_down_file @copybrief top_down_spm.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get the gui parameters (or set them here if not using gui.m).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('gui_has_set_the_params.mat')
% If the previously trained model was on a different machine (different
% paths) then this will convert them to the current machine.

dict_name = 'dictionary_200.mat';
% For Layer 1
pyramid_pool_type = 'Abs_Avg';  % type of pooling used from pixels to grid in the my_pyramid.m
pyramid_pool_size = [4 4];
model.PLOT_RESULTS = 1;
model.SAVE_RESULTS = 0;
model.comp_threads = 4;
% Where the descriptors lie.
model.fullmodelpath = 'E:/Jeffrey/Documents/MATLAB/Matthew/old/Results/fruit_100_100/Run_1/epoch5_layer4.mat';
% Where the dictionary is.
model.fulldatapath = 'E:/Jeffrey/Documents/MATLAB/Matthew/old/Results/fruit_100_100/Run_1';
model.fullsavepath = model.fullmodelpath;
model.machine = ' local_server';
%%%%%%%%%


% model = convert_paths(model);
guimodel = model; % Save so you can plot whatever happened last.
maxNumCompThreads(model.comp_threads);
COMP_THREADS = model.comp_threads;
% Get the only parameters set in guimodel that we need.
PLOT_RESULTS = model.PLOT_RESULTS;
SAVE_RESULTS = model.SAVE_RESULTS;
fullsavepath = model.fullsavepath;
fullmodelpath = model.fullmodelpath;
fulldatapath = model.fulldatapath;
machine = model.machine;
% Path to the top layer of the model
topmodelpath = remove_dot_mat(model.fullmodelpath);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Checks how many epochs are in the fullmodelpath (after .mat is removed)
if(strcmp(topmodelpath(end-8:end-8),'h')) %Single digit epochs
    startpath = topmodelpath(1:end-8);
elseif(strcmp(topmodelpath(end-9:end-9),'h')) % Double digit epochs
    startpath = topmodelpath(1:end-9);
else % Triple digit epochs
    startpath = topmodelpath(1:end-10);
end

% Loads the models and the pooled layer indices as well.
load_lower_layers

% % Load the top layer of the model.
load(topmodelpath);
% top_size = size(z);
% model1.norm_types{2} = 'None';
clear z z0 F
% Have to set this so it know's which experiment is running now.
model.expfile = ' top_down';

top_model
% model1.norm_sizes = model2.norm_sizes;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(strcat(fulldatapath,'/',dict_name))
num_maps = top_model.num_feature_maps(top_model.layer);
dictionary  = dictionary';
% Reshape by spliting into feature maps.
dictionary = reshape(dictionary,size(dictionary,1)/num_maps,num_maps,size(dictionary,2));
% Make into the required patches patchSize x patchSize x num_maps x
% dictSize
dictionary = reshape(dictionary,sqrt(size(dictionary,1)),sqrt(size(dictionary,1)),size(dictionary,2),size(dictionary,3));
    % Here you unpool before proceeding with the reconstructions.
    switch(pyramid_pool_type)
        case 'Max' % Discrete or Probablistic  Max Pooling
            % Puts them all in the first place (didn't save indices).
        dictionary = reverse_max_pool(dictionary,ones(size(dictionary)),pyramid_pool_size);
        case 'Avg' % Average Pooling
        dictionary = reverse_avg_pool(dictionary,ones(size(dictionary)),pyramid_pool_size);
        case 'Abs_Avg' % Average Pooling
        dictionary = reverse_avg_pool(dictionary,ones(size(dictionary)),pyramid_pool_size);      
        case 'None'
    end
    
    
    f = figure(55);
    sdispmaps(dictionary);
set(f,'Name','Dictionary Feature Maps (after unpooling)');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display all original filters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(PLOT_RESULTS>0 || SAVE_RESULTS>0)
    for layer=1:top_layer
        slayer = num2str(layer);
        % Make a new figure.
        eval(strcat('f = figure(20',slayer,');')); clf;
        eval(strcat('sdispfilts(F',slayer,',model.conmats{',slayer,'});'));
        set(f,'Name',strcat('Layer ',num2str(layer),' Original Filters'));
        drawnow;
        cursize = get(f,'Position');
        screensize = get(0,'Screensize');
        set(f,'Position',[800,screensize(4)-cursize(4)-100,cursize(3),cursize(4)])
        drawnow;
        if(SAVE_RESULTS>0)
            hgsave(f,strcat(fullsavepath,'original_filters_layer',num2str(layer),'.fig'));
            if(PLOT_RESULTS==0)
                close(f) % Only plotted it in order to save it.
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If running on the laptop then don't use IPP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(strcmp(model.machine,' laptop'))
    LAPTOP = 1;
else
    LAPTOP = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get middles of each feature maps at the top layer.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch(top_model.norm_types{top_layer+1})
    case 'Max'  % Max Pooling
        % If pooled at top, then filter adjustment is on unpooled features
        middlez_x = floor(top_size(1)/top_model.norm_sizes{top_layer+1}(1));
        middlez_y = floor(top_size(2)/top_model.norm_sizes{top_layer+1}(2));
    case 'Avg'  % Average Pooling
        % If pooled at top, then filter adjustment is on unpooled features
        middlez_x = floor(top_size(1)/top_model.norm_sizes{top_layer+1}(1));
        middlez_y = floor(top_size(2)/top_model.norm_sizes{top_layer+1}(2));
    case 'None'
        % Get the middle index to place the point the top feature maps.
        middlez_x = floor(top_model.xdim/2+top_model.filter_size(top_model.layer)/2);
        middlez_y = floor(top_model.ydim/2+top_model.filter_size(top_model.layer)/2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Place the descriptor cluster center in middle of top maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for top_feature_map=1:size(dictionary,4)  % Project each element of the dictionary.
    fprintf('Processing feature map: %d out of %d\n',top_feature_map,size(dictionary,4));
    
    % Make the map zero to start.
    eval(strcat('recon_z',num2str(top_layer),' = zeros(top_size,',char(39),'single',char(39),');'));
    
    %Place the dictionary element near the middle.
    dictelem = dictionary(:,:,:,top_feature_map);
    eval(strcat('recon_z',num2str(top_layer),'(middlez_x-floor(size(dictelem,1)/2)+1:middlez_x+floor(size(dictelem,1)/2),',...
        'middlez_y-floor(size(dictelem,2)/2)+1:middlez_y+floor(size(dictelem,2)/2),:) = dictelem;'));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Reconstruct from the top down.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    layer=top_layer; % Prepare for top_down_core
    
    % Do the layer by layer reconstructions from the top (ending with
    % recon_z0 as the input image).
    top_down_core
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Save a copy of the pixel space for each feature map pass.
    recon_y1(:,:,:,top_feature_map) = recon_z0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Crop the reconstructions around the filter's pooling regions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
before_crop = recon_y1;
% Get the center coordinates of the reconstructed image planes.
middlex = floor(model1.xdim/2);
middley = floor(model1.ydim/2);

% Have to take into account the filter+1 or -1 here I think.
summed_filter_sizes = 0;
for i=1:top_layer
    if(i > 1)
        switch(top_model.norm_types{i+1})
            case 'Max'
                summed_filter_sizes = summed_filter_sizes + ...
                    (top_model.filter_size(i)-1)*top_model.norm_sizes{i+1}(1);
            case 'Avg'
                summed_filter_sizes = summed_filter_sizes + ...
                    (top_model.filter_size(i)-1)*top_model.norm_sizes{i+1}(1);
            case 'None'
                summed_filter_sizes = summed_filter_sizes + ...
                    top_model.filter_size(i);
                
        end
    else % For layer 1, don't double sizes even if subsampled.
        summed_filter_sizes = summed_filter_sizes + ...
            top_model.filter_size(i);
    end
end
% For each layer add the filter sizes up.
%     summed_filter_sizes = sum(top_model.filter_size(1:top_layer));
%     % Add one for each layer (to be safe) then take half.
summed_filter_sizes = floor((summed_filter_sizes)/2);

% Get start an end indices for each image plane to do the crop.
startx = max(middlex-summed_filter_sizes,1);
starty = max(middley-summed_filter_sizes,1);
endx = min(middlex+summed_filter_sizes,model1.xdim);
endy = min(middley+summed_filter_sizes,model1.ydim);

% Crop the reconstructions.
recon_y1 = recon_y1(startx:endx,starty:endy,:,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Undo the normalization on the input images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch(model1.norm_types{1})
    case 'Max'  % Max Pooling
        recon_y1 = reverse_prob_max_pool(recon_y1,pooled_indices0,model1.norm_sizes{1},[model1.orig_xdim+model1.filter_size(1)-1 model1.orig_ydim+model1.filter_size(1)-1]);
    case 'None'
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the pixel space filters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(PLOT_RESULTS>0 || SAVE_RESULTS>0)
    % Plot the final pixel space reconstruction from the top down pass
    eval(strcat('f = figure(25',num2str(top_layer),');')); clf;
    sdispims(recon_y1);
    set(f,'Name',strcat('Layer ',num2str(layer),' Pixel Space Filters'));
    drawnow;
    cursize = get(f,'Position');
    screensize = get(0,'Screensize');
    if(screensize(4)<1200)
        set(f,'Position',[1750,30,cursize(3),cursize(4)])
    else
        set(f,'Position',[1750,900,cursize(3),cursize(4)])
    end
    drawnow;
    if(SAVE_RESULTS>0)
        hgsave(f,strcat(fullsavepath,'pixel_filters_layer',num2str(top_layer),'.fig'));
        % Also save the pixel space filters.
        pixF = recon_y1;
        save(strcat(fullsavepath,'pixel_filters_layer',num2str(top_layer),'.mat'),'pixF');
        if(PLOT_RESULTS==0)
            close(f) % Only plotted it in order to save it.
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
