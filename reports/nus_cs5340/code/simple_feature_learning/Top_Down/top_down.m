%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Visualizes the filters in pixel space from models above by placing a single
% one in each of the top feature maps independtly and then reconstructing 
% downwards. This is done from the
% top model that is defined in the model struct's fullmodelpath field. All other
% layers will be loaded for their last trained epoch.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @outparam \li \e gui_has_set_the_params the model structure for the
% experiemnt's parameters.
%
% @top_down_file @copybrief top_down.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup the model parameters here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except SKIP_LOADING_PARAMS
try
    if(exist('SKIP_LOADING_PARAMS','var'))
        error('Skipping the loading of the the gui_has_set_the_params.mat file')
    else
        load('gui_has_set_the_params.mat')
    end
    
catch ME1
    fprintf('No model parameters set, use defaults set in train.m\n');
    
    clear all
    

    %%%%%%%%%%%%%%%%%
    % All the model struct defaults are set in the following file.
    % Make any desired changes within that file.
    %
    % Important: need to specify a train model in model.fullmodelpath.
    % Important: need to specify a model.fullsavepath where you want to save to
    %%%%%%%%%%%%%%%%%
    set_defaults_here
    %%%%%%%%%%%%%%%%%
end
%%%%%%
%% Additional things to do with the model (do not modify)
model.expfile = ' top_down';
model = backwards_compatible(model);
guimodel = model; % Save so you can plot whatever happened last.
maxNumCompThreads(model.comp_threads);
COMP_THREADS = model.comp_threads;
% Get the only parameters set in guimodel that we need.
PLOT_RESULTS = model.PLOT_RESULTS;
SAVE_RESULTS = model.SAVE_RESULTS;
fullsavepath = model.fullsavepath;
% this is where the top model comes from.
fullmodelpath = model.fullmodelpath;
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
%% Place a single point in the middle of each top layer feature map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for top_feature_map=1:top_model.num_feature_maps(top_layer)
    fprintf('Processing feature map: %d out of %d\n',top_feature_map,model.num_feature_maps(top_layer));
    
    % Make the map zero to start.
    eval(strcat('recon_z',num2str(top_layer),' = zeros(top_size,',char(39),'single',char(39),');'));
    
    %Place the single piont in each z2 filter map.
    eval(strcat('recon_z',num2str(top_layer),'(middlez_x,middlez_y,top_feature_map) = 1;'));
    
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
        set(f,'Position',[1750,30,cursize(3)*2,cursize(4)*2])
    else
        set(f,'Position',[1750,900,cursize(3)*2,cursize(4)*2])
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
