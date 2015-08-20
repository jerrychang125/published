%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Samples the feature maps to reconstruct the samples in pixel space for the 
% distribution of feature map activations from the training images. This uses
% sample_z_map.m to do the sampling. The top model is determiend by loading
% gui_has_set_the_params.mat All other
% layers will be loaded for their last trained epoch.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @outparam \li \e gui_has_set_the_params the model structure for the
% experiemnt's parameters.
%
% @top_down_file @copybrief top_down_sampling.m
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
model.expfile = ' top_down_sampling';
model = backwards_compatible(model);
guimodel = model;
maxNumCompThreads(model.comp_threads);
COMP_THREADS = model.comp_threads;
% Get the only variables from guimodel that we may need.
PLOT_RESULTS = model.PLOT_RESULTS;
SAVE_RESULTS = model.SAVE_RESULTS;
fullsavepath = model.fullsavepath;
fullmodelpath = model.fullmodelpath;
machine = model.machine;
% Get the top layer of the model.
topmodelpath = remove_dot_mat(model.fullmodelpath);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the layers (and select correct pooled/unpooled top features)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Checks how many epochs are in the fullmodelpath (after .mat is removed)
if(strcmp(topmodelpath(end-8:end-8),'h')) %Single digit epochs
    startpath = topmodelpath(1:end-8);
elseif(strcmp(topmodelpath(end-9:end-9),'h')) % Double digit epochs
    startpath = topmodelpath(1:end-9);
else % Triple digit epochs
    startpath = topmodelpath(1:end-10);
end

load_lower_layers

load(topmodelpath)

switch(top_model.norm_types{top_layer+1})
    case 'Max'  % Max Pooling
        eval(strcat('top_z = pooled_maps',num2str(top_layer),';'));
    case 'None'
        % % Save the top_layer z maps for the sampling
        top_z = z;
end
% Have to set this so it know's which experiment is running now.
model.expfile = ' top_down_sampling';
clear z z0 F
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
if(strcmp(model.machine,' laptop'))
    LAPTOP = 1;
else
    LAPTOP = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Randomly sample from the distribution of trained feature maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for top_feature_map=1:1
    
    fprintf('Processing feature map: %d out of %d\n',top_feature_map,model.num_feature_maps(top_layer));
    
    % Get the probabilities at each feature map.
    eval(strcat('recon_z',num2str(top_layer),' = sample_z_map(top_model.num_feature_maps(top_layer),top_z(:,:,:,top_feature_map),0);'));
    % Sort these probabilities.
    eval(strcat('[sorted,ind] = sort(-recon_z',num2str(top_layer),'(:));'));
    % Keep only the first 100 highest probabilities (samples).
    eval(strcat('recon_z',num2str(top_layer),'(ind(100:end)) = 0;'));
    
    
    %Place the single piont in each z2 filter map.
    %     eval(strcat('recon_z',num2str(top_layer),'(middlez_x,middlez_y,top_feature_map) = 1;'));
    
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

%
% % Cut off the unneeded size of the pixel space reconstructions
% % Get the center coordinates of the reconstructed image planes.
% middlex = floor(model1.xdim/2);
% middley = floor(model1.ydim/2);
%
% % For each layer add the filter sizes up.
% summed_filter_sizes = sum(top_model.filter_size(1:top_layer));
% % Add one for each layer (to be safe) then take half.
% summed_filter_sizes = floor((summed_filter_sizes)/2);
%
% % Get start an end indices for each image plane to do the crop.
% startx = middlex-summed_filter_sizes;
% starty = middley-summed_filter_sizes;
% endx = middlex+summed_filter_sizes;
% endy = middley+summed_filter_sizes;
%
% % Crop the reconstructions.
% recon_y1 = recon_y1(startx:endx,starty:endy,:,:);

if(PLOT_RESULTS>0 || SAVE_RESULTS>0)
    % Plot the final pixel space reconstruction from the top down pass
    eval(strcat('f = figure(25',num2str(top_layer),');')); clf;
    sdispims(recon_y1);
    title(strcat('Layer: ',num2str(top_layer),' Sampled'))
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
        hgsave(f,strcat(fullsavepath,'sampled_filters_layer',num2str(top_layer),'.fig'));
        if(PLOT_RESULTS==0)
            close(f) % Only plotted it in order to save it.
        end
    end
end


% % Save the pixel space results to a .fig file.
% if(SAVE_RESULTS)
%     savepath = strcat(fullsavepath,'sampled_filters_layer',num2str(top_layer),'.fig');
%     eval('ssaveims(recon_y1,savepath);');
% end

