%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% The entry point to train all desired layers of a Deconvolutional Network.
% All the parameters of the model can be set here in a 'model' struct.
% Alternatively, these parameters can be saved to a file title
% 'gui_has_set_the_params.mat' either by hand or using a gui (not included).
% Every parameters including the paths to the images you want to train on and
% the save directory are included in model and must be modified at the top of
% this file.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inparam \li \e model the main structure with all the parameters of the model
% must be set within this file. There are many fields to set. Read the comments
% within the file for more information.
%
% @training_file @copybrief train.m
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
    %%%%%%%%%%%%%%%%%
    set_defaults_here
    %%%%%%%%%%%%%%%%%
end
%%%%%%
%% Additional things to do with the model (do not modify)
model.expfile = ' train';
model = backwards_compatible(model);
guimodel = model;
mkdir2(model.fullsavepath);

% Set number of computation threads to use.
maxNumCompThreads(model.comp_threads);
%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare for Layer 1: Get Data Ready
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Overwrite contrast normalize to 0 sinze not needed if using z0.
if(model.TRAIN_Z0 == 1)
    model.CONTRAST_NORMALIZE = 0; % THE ONLY REASON FOR Z0 IS THIS.
end

% Load images from .mat file.
if(exist(strcat(guimodel.fulldatapath,'original_images.mat'),'file'))
    fprintf('Using precomputed images.\n')
    load(strcat(guimodel.fulldatapath,'original_images.mat'))
else
    [original_images,mn,sd,xdim,ydim] = CreateImages(model.fulldatapath,model.CONTRAST_NORMALIZE,model.ZERO_MEAN,model.COLOR_IMAGES,model.SQUARE_IMAGES);
    % Makes the output scaled to roughly [-1,1]
    original_images = rescale_all_n1_1(original_images);
end

% Store the original image sizes in each model (in case pooling was done on
% the original images).
model.orig_xdim = size(original_images,1);
model.orig_ydim = size(original_images,2);

% Convert to single precision.
original_images = single(original_images);

% Make sure the first layer input maps make sense (regarless of color at
% this point).
model.num_input_maps(1) = size(original_images,3);
if(size(model.conmats{1},1) ~= model.num_input_maps(1))
    model.conmats{1} = model.conmats{1}(1:model.num_input_maps(1),:);
end

% Save the images passed in.
% Jie Fu
% if(guimodel.SAVE_RESULTS > 0)
%     save(strcat(guimodel.fullsavepath,'training_images.mat'),...
%         'original_images','mn','sd');
% end

if(guimodel.SAVE_RESULTS > 0)
    save(strcat(guimodel.fullsavepath,'training_images.mat'),...
        'original_images');
end

% For the first layer the input maps are the original images.
y = original_images;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Normalize the data if needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If pooling is required, apply it to the feature maps (y). Then save the results.
% It is at train_layer+1 since there could be normalization on the input first.
% Note: original_images are always their original size.
switch(guimodel.norm_types{1})
    case 'Max'  % Max Pooling
        [pooled_maps,pooled_indices] = max_pool(original_images,[guimodel.norm_sizes{1}(1) guimodel.norm_sizes{1}(2)]);
        
        if(guimodel.SAVE_RESULTS>0)
            % Save the 4 results of the max_pool function.
            save(strcat(guimodel.fullsavepath,'layer0_pooling.mat'),...
                'pooled_maps','pooled_indices');
        end
        % Copy the variables over for the next iteration.
        y = single(pooled_maps);
        pooled_indices0 = pooled_indices;
        clear pooled_maps pooled_indices
    case 'Avg'  % Average Pooling
        [pooled_maps,pooled_indices] = avg_pool(original_images,[guimodel.norm_sizes{1}(1) guimodel.norm_sizes{1}(2)]);
        
        if(guimodel.SAVE_RESULTS>0)
            % Save the 4 results of the max_pool function.
            save(strcat(guimodel.fullsavepath,'layer0_pooling.mat'),...
                'pooled_maps','pooled_indices');
        end
        % Copy the variables over for the next iteration.
        y = single(pooled_maps);
        pooled_indices0 = pooled_indices;
        clear pooled_maps pooled_indices
    case 'None'
        y = original_images;
        pooled_indices0 = [];
        % Note: shouldn't have to setup the next layer's sizes because the
        % number of input maps are the same, just their dimensions change
        % which is figured out at the start of the training code.
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loop over each layer you want to train.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for train_layer=1:model.num_layers
    
    % Specify which layer you are learning for saving the results.
    model.layer = train_layer;
    
    % don't use z0 maps in higher layers.
    if(train_layer>1)
        model.TRAIN_Z0 = 0;
    end
    
    % xdim and ydim are the size of the input layers which are vectorized into
    fprintf('Training Layer %d of a %d-Layer Model\n',model.layer,model.num_layers);
    fprintf('Number of Input Maps = %d, Number of Feature Maps = %d\n',model.num_input_maps(train_layer),model.num_feature_maps(train_layer))
    fprintf('The connectivity map is:\n');
    %     disp(model.conmats{model.layer})   % Display the matrix
    
    % Save the model as model# where # is the layer.
    eval(strcat('model',num2str(train_layer),'=model;'));
    % Since training, initialize F1 and z1 == 0 to start.
    eval(strcat('F',num2str(train_layer),' = 0;'));
    eval(strcat('z0',num2str(train_layer),' = 0;'));
    % Initialize to empty pooling indices for the first layer.
    eval(strcat('pooled_indices',num2str(train_layer),' = [];'));
    
    % A string of parameters to pass to each layer.
    modelargs = '';
    % Construct the modelargs string.
    for layer=train_layer:-1:1
        modelargs = strcat(modelargs,',','model',num2str(layer));
        modelargs = strcat(modelargs,',','F',num2str(layer));
        modelargs = strcat(modelargs,',','z0',num2str(layer));
        modelargs = strcat(modelargs,',','pooled_indices',num2str(layer));
    end
    % Get rid of the first ',' that is in the string.
    modelargs = modelargs(2:end);
    % Add the input_map (y), original images and
    modelargs = strcat(modelargs,',pooled_indices0,y,original_images');
    
    
    % The returned y here is the feature maps (called y so that for the
    % next layer it is the input maps).
    eval(strcat('[F',num2str(train_layer),',y,z0',num2str(train_layer)',...
        ',y_tilda',num2str(train_layer),',model',num2str(train_layer),'] = train_recon_layer(',modelargs,');'));
    
    % Reset the modelargs sring.
    modelargs = '';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Normalization Procedure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % If pooling is required, apply it to the feature maps (y). Then save the results.
    % It is at train_layer+1 since there could be normalization on the input first.
    switch(guimodel.norm_types{train_layer+1})
        case 'Max' % Max Pooling
            fprintf('\n\nMax Subsampling between layers \n\n');
            
            [pooled_maps,pooled_indices] = max_pool(y,[guimodel.norm_sizes{train_layer+1}(1) guimodel.norm_sizes{train_layer+1}(2)]);
            
            if(guimodel.SAVE_RESULTS>0)
                % Save the 4 results of the max_pool function.
                save(strcat(guimodel.fullsavepath,'layer',num2str(train_layer),'_pooling.mat'),...
                    'pooled_maps','pooled_indices','-v7.3');
            end
            % Copy the variables over for the next iteration.
            y = single(pooled_maps);
            % Keep track of each layer's pooled indices.
            eval(strcat('pooled_indices',num2str(train_layer),' = pooled_indices;'));
            clear pooled_maps pooled_indices
        case 'Avg' % Average pooling
            fprintf('\n\nAverage Subsampling between layers \n\n');
            
            [pooled_maps,pooled_indices] = avg_pool(y,[guimodel.norm_sizes{train_layer+1}(1) guimodel.norm_sizes{train_layer+1}(2)]);
            if(guimodel.SAVE_RESULTS>0)
                % Save the 4 results of the max_pool function.
                save(strcat(guimodel.fullsavepath,'layer',num2str(train_layer),'_pooling.mat'),...
                    'pooled_maps','pooled_indices','-v7.3');
            end
            % Copy the variables over for the next iteration.
            y = single(pooled_maps);
            % Keep track of each layer's pooled indices.
            eval(strcat('pooled_indices',num2str(train_layer),' = pooled_indices;'));
            clear pooled_maps pooled_indices
            
    end
    
    
    
    % Makes feature map [-1,1]
    y = rescale_all_n1_1(y);
    
    % Note: shouldn't have to setup the next layer's sizes because the
    % number of input maps are the same, just their dimensions change
    % which is figured out at the start of the training code.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end

% Visualize the upper layer filters in pixel space.
if(guimodel.PLOT_RESULTS || guimodel.SAVE_RESULTS)
top_down_last
end


