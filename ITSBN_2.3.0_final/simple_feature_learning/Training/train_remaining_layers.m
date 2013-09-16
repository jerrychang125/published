%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% The entry point to train layers on top of a previously trained Deconvolutional Network.
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
% @training_file @copybrief train_remaining_layers.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the gui's parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
load('set_parameters.mat')
% If the previously trained model was on a different machine (different
% paths) then this will convert them to the current machine.
model = convert_paths(model);
guimodel = model;
maxNumCompThreads(model.comp_threads);
% Save th epath to the highest previously trained model.
fullmodelpath = guimodel.fullmodelpath;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the layers below the previously trained top layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load all the layers below the top layer as model#,z0#,F#
load_lower_layers

% Load the last good layer (model,y,z,z0,F)'s
% Note: this overwrites the model structure!!!
% This is where the input maps for the nex layer are loaded.
load(guimodel.fullmodelpath)

% Make the old model a new structure.
oldmodel = model;
model = guimodel;
model.expfile = 'train'; % For saving the figures.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Combine the new gui parameters with the oldmodel parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get where it was left off.
layer = oldmodel.layer;

% Copy the array up to where it left off.
model.maxepochs(1:layer) = oldmodel.maxepochs(1:layer);
model.filter_size(1:layer) = oldmodel.filter_size(1:layer);
model.num_feature_maps(1:layer) = oldmodel.num_feature_maps(1:layer);
model.num_input_maps(1:layer) = oldmodel.num_input_maps(1:layer);
model.lambda(1:layer) = oldmodel.lambda(1:layer);
model.alpha(1:layer) = oldmodel.alpha(1:layer);
model.kappa(1:layer) = oldmodel.kappa(1:layer);
for i=1:layer
    model.conmat_types{i} = oldmodel.conmat_types{i};
    model.conmats{i} = oldmodel.conmats{i};
    model.norm_types{i} = oldmodel.norm_types{i};
    model.norm_sizes{i} = oldmodel.norm_sizes{i};
end
% The pooling after previous top can change so don't copy.
% model.norm_types{layer+1} = oldmodel.norm_types{layer+1};
% model.norm_sizes{layer+1} = oldmodel.norm_sizes{layer+1};
% LAYER 1 SPECIFIC THINGS (taken from model1)
model.tag = model1.tag;
model.VARIANCE_THRESHOLD = model1.VARIANCE_THRESHOLD;
model.CONTRAST_NORMALIZE = model1.CONTRAST_NORMALIZE;
model.COLOR_IMAGES = model1.COLOR_IMAGES;
model.TRAIN_Z0 = model1.TRAIN_Z0;
model.layer = model1.layer;
model.fulldatapath = model1.fulldatapath;
start_layer = layer;
model.UPDATE_INPUT = model1.UPDATE_INPUT;
model.lambda_input = model1.lambda_input;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copy lower layers to new save directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This copies all the layer below and including the fullmodelpath model
% specified in the gui (which is what you want to train above).
copy_lower_layers(guimodel);
% Setup the next layer to have the correct epochs for the previous layer below.
model.maxepochs(layer) = get_highest_epoch(guimodel.fullsavepath,layer);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the images that were used for training previously.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(model.COLOR_IMAGES)
    [original_images,good_ind,xdim,ydim] = CreateColorImages(model.fulldatapath,model.CONTRAST_NORMALIZE,model.VARIANCE_THRESHOLD);
else
    [original_images,good_ind,xdim,ydim] = CreateGrayImages(model.fulldatapath,model.CONTRAST_NORMALIZE,model.VARIANCE_THRESHOLD);
end
% Store the original image sizes in each model (in case pooling was done on
% the original images).
model.orig_xdim = xdim;
model.orig_ydim = ydim;
original_images = single(original_images); % make them single
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% load_lower_layers sets up the pooled_indices0 now.

% % Don't normalize the original_images since they are always at their
% % original sizes when passed into train_recon_layer.
% % Here just get the pooled_indices0 setup.
% %%%%%%%%%%
% %% Normaliztion of the input image (get the pooled_indices0)
% %%%%%%%%%%
% % If pooling is required, apply it to the feature maps (y). Then save the results.
% % It is at train_layer+1 since there could be normalization on the input first.
% switch(model.norm_types{1})
%     case 'Max'  % Max Pooling
%                 % If there was pooling done previously then load the pooling file.
%         if(strcat(pooldir,'/layer0_pooling.mat'),'file')
%             load(strcat(pooldir,'/layer0_pooling.mat'))
%             pooled_indices0 = pooled_indices;
%         else % Make them empty if there was no pooling done.
%             fprintf('Trying to load pooling for top layer but does not exist.\n Recomputing pooling after top layer now.')
%             [~,pooled_indices0] = max_pool(original_images,[model.norm_sizes{top_layer+1}(1) model.norm_sizes{top_layer+1}(2)]);
%         end
%
%         % Copy the variables over for the next iteration.
%         original_images = single(pooled_maps);
%         clear pooled_maps pooled_size original_size
%
%         % Note: shouldn't have to setup the next layer's sizes because the
%         % number of input maps are the same, just their dimensions change
%         % which is figured out at the start of the training code.
%     case 'None'
%         pooled_indices0 = [];
% end
% %%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Normalize maps of the previous top layer (input maps for next layer)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If there was pooling done previously then that should have been loaded by
% This pooling has to have the same size as the new one from the gui.
if(exist(strcat('pooled_maps',num2str(top_layer)),'var') && ...
        all(model.norm_sizes{top_layer+1} == oldmodel.norm_sizes{top_layer+1}) &&...
    strcmp(model.norm_types{top_layer+1},oldmodel.norm_sizes{top_layer+1}))
  'using previous pooled_maps'  
    eval(strcat('y = pooled_maps',num2str(top_layer),';'));
% if(exist(strcat(pooldir,'/layer',num2str(top_layer),'_pooling.mat'),'file'))
%     load(strcat(pooldir,'/layer',num2str(top_layer),'_pooling.mat'))
%     eval(strcat('pooled_maps',num2str(top_layer),' = pooled_maps;'));
%     y = single(pooled_maps);
%     clear pooled_maps pooled_indices
else % Make them empty if there was no pooling done.
    'new pooling of the top layer now'
    fprintf('Trying to load pooling for top layer but does not exist.\n Recomputing pooling after top layer now.')
    switch(model.norm_types{top_layer+1})
        case 'Max'  % Max Pooling
            % z is the top layer previously trained feature maps.
            [pooled_maps,pooled_indices] = max_pool(z,[model.norm_sizes{top_layer+1}(1) model.norm_sizes{top_layer+1}(2)]);
            
            eval(strcat('pooled_indices',num2str(top_layer),' = pooled_indices;'));
            eval(strcat('pool_maps',num2str(top_layer),' = pooled_maps;'));
            
            % Save the 4 results of the max_pool function.
            % This has to be saved here so the sizes are correct going
            % forwards for the pooled_indices.
            if(guimodel.SAVE_RESULTS>0)
                save(strcat(guimodel.fullsavepath,'layer',num2str(top_layer),'_pooling.mat'),...
                    'pooled_maps','pooled_indices');
            end
            % Copy the pooled feature maps from previous top layer to the input for the next layer to be trained.
            y = single(pooled_maps);
            clear pooled_maps pooled_indices
        case 'Avg'  % Max Pooling
            % z is the top layer previously trained feature maps.
            [pooled_maps,pooled_indices] = avg_pool(z,[model.norm_sizes{top_layer+1}(1) model.norm_sizes{top_layer+1}(2)]);
            
            eval(strcat('pooled_indices',num2str(top_layer),' = pooled_indices;'));
            eval(strcat('pool_maps',num2str(top_layer),' = pooled_maps;'));
            
            % Save the 4 results of the max_pool function.
            % This has to be saved here so the sizes are correct going
            % forwards for the pooled_indices.
            if(guimodel.SAVE_RESULTS>0)
                save(strcat(guimodel.fullsavepath,'layer',num2str(top_layer),'_pooling.mat'),...
                    'pooled_maps','pooled_indices');
            end
            % Copy the pooled feature maps from previous top layer to the input for the next layer to be trained.
            y = single(pooled_maps);
            clear pooled_maps pooled_indices            
            
        case 'None'
            %% Copy the last model's feature maps into y as input to next layer.
            clear y
            y = z;
    end
end

% Normalize this previous pooled feature map.
    y = svm_rescale2(y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train Additional Layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for train_layer=start_layer+1:model.num_layers
    
    
    % Specify which layer you are learning for saving the results.
    model.layer = train_layer;
    model.layer
    
    % Make sure that the number of computation threads is set by the gui.
    model.comp_threads = guimodel.comp_threads;
    model.DISPLAY_ERRORS= guimodel.DISPLAY_ERRORS;
    
    % Do you want to train the z0 contrast normalizing feature maps as well?
    % Since training above layer 1, always avoid the z0 map.
    model.TRAIN_Z0 = 0;
    
    fprintf('Training Layer %d of a 2-Layer Model\n',model.layer);
    fprintf('Number of Input Maps = %d, Number of Feature Maps = %d\n',model.num_input_maps(layer),model.num_feature_maps(layer))
    fprintf('The connectivity map is:');
    
    % Create the connectivity matrix.
%     disp(model.conmats{model.layer})   % Display the matrix
    
    
    % Save the model as model# where # is the layer.
    eval(strcat('model',num2str(train_layer),'=model;'));
    % Since training, initialize F1 and z1 == 0 to start.
    eval(strcat('F',num2str(train_layer),' = single(0);'));
    eval(strcat('z0',num2str(train_layer),' = single(0);'));
    % Initialize to empty pooling indices for the first layer.
    eval(strcat('pooled_indices',num2str(train_layer),' = [];'));
    
    
    % A string of parameters to pass to each layer.
    modelargs = '';
    % Construct the modelargs string.
    for layer=train_layer:-1:1
        modelargs = strcat(modelargs,',','model',num2str(layer));
        modelargs = strcat(modelargs,',','single(F',num2str(layer),')');
        modelargs = strcat(modelargs,',','single(z0',num2str(layer),')');
        modelargs = strcat(modelargs,',','pooled_indices',num2str(layer));
    end
    % Get rid of the first ',' that is in the string.
    modelargs = modelargs(2:end);
    % Add the input_map (y), original images and
    modelargs = strcat(modelargs,',pooled_indices0,y,original_images');
    
    % Train the layers.
    eval(strcat('[F',num2str(train_layer),',y,z0',num2str(train_layer)',...
        ',y_tilda',num2str(train_layer),',model',num2str(train_layer),'] = train_recon_layer(',modelargs,');'));
    
    % Reset the modelargs sring.
    modelargs = '';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Normalization Procedure (After each Layer)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % If pooling is required, apply it to the feature maps (y). Then save the results.
    % It is at train_layer+1 since there could be normalization on the input first.
    switch(guimodel.norm_types{train_layer+1})
        case 'Max'  % Max Pooling
            [pooled_maps,pooled_indices] = max_pool(y,[guimodel.norm_sizes{train_layer+1}(1) guimodel.norm_sizes{train_layer+1}(2)]);
            
            % Save the 4 results of the max_pool function.
            save(strcat(guimodel.fullsavepath,'layer',num2str(train_layer),'_pooling.mat'),...
                'pooled_maps','pooled_indices');
            
            % Copy the variables over for the next iteration.
            y = single(pooled_maps);
            % Keep track of each layer's pooled indices.
            eval(strcat('pooled_indices',num2str(train_layer),' = pooled_indices;'));
            clear pooled_maps pooled_indices
    end
    
    % Note: shouldn't have to setup the next layer's sizes because the
    % number of input maps are the same, just their dimensions change
    % which is figured out at the start of the training code.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end

