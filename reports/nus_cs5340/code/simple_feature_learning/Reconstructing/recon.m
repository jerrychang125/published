%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% The entry point to reconstruct/denoise an image using the desired layers 
% of a previously trained Deconvolutional Network.
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
% @recon_file @copybrief recon.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
%% Setup the model parameters here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the parameters for the experiment from a file.
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
model.expfile = ' recon';
% If the previously trained model was on a different machine (different
% paths) then this will convert them to the current machine.
% model = convert_paths(model);
model = backwards_compatible(model);
guimodel = model;
maxNumCompThreads(model.comp_threads);
% Save the parameters you need.
PLOT_RESULTS = model.PLOT_RESULTS;
SAVE_RESULTS = model.SAVE_RESULTS;
fullsavepath = model.fullsavepath;
fullmodelpath = model.fullmodelpath;
fulldatapath = model.fulldatapath;
machine = model.machine;
% expfile = guimodel.expfile(2:end);
expfile = 'recon';
topmodelpath = str2mat(remove_dot_mat(fullmodelpath));
% Get the top layer (assuming it is not more than 10).
top_layer = str2num(topmodelpath(end:end));
% Noise types
noisetype = model.noisetype;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BACKWARDS COMPATIBILITY (do not modify anything below)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
guimodel = backwards_compatible(guimodel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the model's for each layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This requires model to be set/loaded and loads all the layers below the model specified.
% guimodel is returned (the input model undisturbed).
load_lower_layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the inputs to layer 1.
% Check if there is already saved noisy images.
if(exist(strcat(guimodel.fulldatapath,'original_images.mat'),'file') &&...
        exist(strcat(guimodel.fulldatapath,'noisy_images.mat'),'file'))
    fprintf('Using precomputed noisy image.\n')
    load(strcat(guimodel.fulldatapath,'original_images.mat'))
    load(strcat(guimodel.fulldatapath,'noisy_images.mat'))
else
    [original_images,mn,sd,xdim,ydim] = CreateImages(guimodel.fulldatapath,0,guimodel.ZERO_MEAN,guimodel.COLOR_IMAGES,guimodel.SQUARE_IMAGES);    
    original_images = rescale_all_n1_1(original_images);
%     
%     if(guimodel.COLOR_IMAGES)
%         [original_images,mn,sd,xdim,ydim] = CreateColorImages(guimodel.fulldatapath,0,guimodel.ZERO_MEAN,guimodel.COLOR_IMAGES,guimodel.SQUARE_IMAGES);
%         original_images = single(original_images);
%         if(size(F1,3)==1) % If trained on gray images.
%             for i=1:3 % Copy the one plane into each color plane.
%                 F1(:,:,i,:) = F1(:,:,1,:);
%             end
%             model1.num_input_maps(1) = 3;
%         end
%     else
%         [original_images,mn,sd,xdim,ydim] = CreateGrayImages(guimodel.fulldatapath,0,guimodel.ZERO_MEAN,guimodel.COLOR_IMAGES);
%         original_images = single(original_images);
%         if(size(F1,3)==3) % was trained on color images.
% %             F1 = mean(F1,3); % then just take the average over the color planes
% %             model1.num_input_maps(1) = 1;
%             original_images = repmat(original_images,[1 1 3 1]);
%         end
%     end
    
    % If the images were contrast normalized, then don't use z0.
    if(guimodel.CONTRAST_NORMALIZE)
        guimodel.TRAIN_Z0 = 0;
    else
        guimodel.TRAIN_Z0 = 1;  % This is the default case.
    end
    
    
    % Initialize the noisy image to zeroes.
%     noisy_images = zeros(size(original_images),'single');
    noisy_images = original_images;
    
    switch guimodel.noisetype
        case 'None (Reconstruct)'
            noisy_images = original_images;
        case 'Random'
            noisy_images = original_images + (randn(size(original_images),'single')*0.25);
            noisy_images = max(noisy_images,min(original_images(:)));
            noisy_images = min(noisy_images,max(original_images(:)));
        case 'Random Gray'
            noise = (randn(size(original_images(:,:,1,:)),'single')*0.25);
            for i=1:size(original_images,3) % Same applied ot each color (gray noise).
                noisy_images(:,:,i,:) = original_images(:,:,i,:)+noise;
            end
        case 'Half Clean Half Random'
            sizeO = size(original_images,4);
            original_images = repmat(original_images,[1,1,1,2]);
            noisy_images = original_images;
            for image=1:sizeO
            noisy_images(:,:,:,sizeO+image) = original_images(:,:,:,sizeO+image) + (randn(size(original_images,1),size(original_images,2),size(original_images,3),'single')*0.25);
            end            
        case 'Energy of City vs Fruit'
            [city_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Images/city_100_100/',0,guimodel.ZERO_MEAN);
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Images/fruit_100_100/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,city_images,fruit_images);
            noisy_images = original_images;
        case 'Energy of Scenes'
            [city_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/street/10/',0,guimodel.ZERO_MEAN);
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/coast/10/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,city_images,fruit_images);            
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/forest/10/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,original_images,fruit_images);            
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/highway/10/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,original_images,fruit_images);            
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/inside_city/10/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,original_images,fruit_images);            
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/mountain/10/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,original_images,fruit_images);            
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/open_country/10/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,original_images,fruit_images);            
            [fruit_images,good_ind,xdim,ydim] = CreateColorImages('/misc/FergusGroup/zeiler/Datasets/Scenes/tallbuilding/10/',0,guimodel.ZERO_MEAN);
            original_images = cat(4,original_images,fruit_images);
            noisy_images = original_images;            
        case 'Scattered Lines'
            noisy_images = scattered_lines(original_images,10,1,50,0.5);
        case 'Scattered Boxes'
            noisy_images = scattered_boxes(original_images,5,5,100,0.5);
        case 'Scattered Crosses'
            noisy_images = scattered_crosses(original_images,5,1,50,0.1);
        case 'Rotate Image'
            noisy_images = imrotate(original_images,10);
        case 'Blur Image'
            h = fspecial('gaussian',[11 11],3);
            clear noisy_images
            for i=1:size(original_images,3)
                noisy_images(:,:,i) = conv2(original_images(:,:,i),h,'valid');
            end
            original_images = original_images(6:end-5,6:end-5,:);
        case 'Residual Image'
            'Computing residual images'
            noisy_images = CreateResidualImages(original_images);
        case 'Repmat Random'
            original_images = repmat(original_images,[1,1,1,6]);
            for image=1:size(original_images,4)
            noisy_images(:,:,:,image) = original_images(:,:,:,image) + (randn(size(original_images,1),size(original_images,2),size(original_images,3),'single')*(image-1)*0.05);
            end
    end
    
end

if(guimodel.PLOT_RESULTS>0)
    f = figure(801); clf;
    sdispims(original_images);
    set(f,'Name','Original Real Image');
    cursize = get(f,'Position');
    set(f,'Position',[0 100 cursize(3) cursize(4)])
    if(SAVE_RESULTS>0)
        mkdir2(strcat(fullsavepath,expfile))
        hgsave(f,strcat(fullsavepath,expfile,'/original_images_beforeCN.fig'));
        if(PLOT_RESULTS==0)
            close(f) % Only plotted it in order to save it.
        end
    end
end

if(guimodel.PLOT_RESULTS>0)
    f = figure(802); clf;
    sdispims(noisy_images);
    set(f,'Name','Noisy Real Input Image');
    cursize = get(f,'Position');
    set(f,'Position',[0 100+cursize(4) cursize(3) cursize(4)])
    if(SAVE_RESULTS>0)
        hgsave(f,strcat(fullsavepath,expfile,'/noisy_images_beforeCN.fig'));
        if(PLOT_RESULTS==0)
            close(f) % Only plotted it in order to save it.
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save the original and noisy images matrices as they were.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(guimodel.SAVE_RESULTS>0)
            mkdir2(strcat(fullsavepath,expfile))
    save(strcat(fullsavepath,expfile,'/original_images.mat'),'original_images')
    save(strcat(fullsavepath,expfile,'/noisy_images.mat'),'noisy_images')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Contrast Normalize (if needed)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If the images were contrast normalized, then don't use z0.
if(guimodel.CONTRAST_NORMALIZE)
    'Now Contrast Normalizing the input images.'
    guimodel.TRAIN_Z0 = 0;
    % Since noise is applied to real images only.
    % If contrast normalizing then you must apply it here.
    % The laplacian passed images are the second arguement of this.
    [resIo,original_images] = CreateResidualImages(original_images);
    [resI,noisy_images] = CreateResidualImages(noisy_images);
%     f = figure(10003)
%     sdispims(resIo);
%     set(f,'Name','Residual of Original Image')
%     f = figure(10000);
%     sdispims(resI);
%     set(f,'Name','Residual of Noisy Image')
%     f = figure(10001);
%     sdispims(resI+noisy_images);
%     set(f,'Name','Residual + Noisy Image')
%     f = figure(10002)
%     sdispims(resI+original_images);
%     set(f,'Name','Residual + Original Image')
    %% Could train z0 maps on the residual image.
    
    
    if(guimodel.PLOT_RESULTS>0)
        f = figure(803); clf;
        sdispims(original_images);
        set(f,'Name','Original (CN) Image');
        cursize = get(f,'Position');
        set(f,'Position',[850-2*cursize(3) 30 cursize(3) cursize(4)])
        
        if(SAVE_RESULTS>0)
            hgsave(f,strcat(fullsavepath,expfile,'/original_images_afterCN.fig'));
            if(PLOT_RESULTS==0)
                close(f) % Only plotted it in order to save it.
            end
        end
    end
    if(guimodel.PLOT_RESULTS>0)
        f = figure(804); clf;
        sdispims(noisy_images);
        set(f,'Name','Noisy (CN) Image');
        cursize = get(f,'Position');
        set(f,'Position',[850-3*cursize(3) 30 cursize(3) cursize(4)])
        
        if(SAVE_RESULTS>0)
            hgsave(f,strcat(fullsavepath,expfile,'/noisy_images_afterCN.fig'));
            if(PLOT_RESULTS==0)
                close(f) % Only plotted it in order to save it.
            end
        end
    end
else
%     guimodel.TRAIN_Z0 = 1;  % This is the default case.
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional Noise Added to Contrast Normalized Images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    switch guimodel.noisetype
        case 'Erase Corners'
            noisy_images = erase_corners(original_images,0.005);
        case 'Erase Corner Boxes'
            noisy_images = erase_corner_boxes(original_images,0.008,1,0,0.05);
        case 'Erase Half of Image'
            noisy_images = erase_half_of_image(original_images);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Normalize the noisy_image data if needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If pooling is required, apply it to the feature maps (y). Then save the results.
% It is at train_layer+1 since there could be normalization on the input first.
% Note: original_images are always their original size.
switch(model1.norm_types{1})
    case 'Max'  % Max Pooling
        [pooled_maps,pooled_indices0] = max_pool(noisy_images,model1.norm_sizes{1});
        
        % Copy the variables over for the next iteration.
        y = single(pooled_maps);
        clear pooled_maps
    case 'Avg'  % Average Pooling
        [pooled_maps,pooled_indices0] = avg_pool(noisy_images,model1.norm_sizes{1});
        
        % Copy the variables over for the next iteration.
        y = single(pooled_maps);
        clear pooled_maps        
    case 'None'
        % Setup the first layer input image which is just the niosy image.
        y = noisy_images;
        pooled_indices0 = [];
        % Note: shouldn't have to setup the next layer's sizes because the
        % number of input maps are the same, just their dimensions change
        % which is figured out at the start of the training code.
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Infer up for each layer (store reconstruction in y_tilda##)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each layer, have to infer z# fist (the y for the next layer).
for recon_layer=1:top_layer
    % COPY GUI MODEL PARAMETERS TO THE CURRENT LAYER MODEL (passed in).
    % Set the expfile so saves to proper directory.
    eval(strcat('model',num2str(recon_layer),'.expfile = guimodel.expfile;'))
    % Set number of computation threads from the gui.
    eval(strcat('model',num2str(recon_layer),'.comp_threads = guimodel.comp_threads;'));
    % Set the maxepochs from the gui.
    eval(strcat('model',num2str(recon_layer),'.maxepochs = guimodel.maxepochs;'));
    eval(strcat('model',num2str(recon_layer),'.min_iterations = guimodel.min_iterations;'));    
    eval(strcat('model',num2str(recon_layer),'.UPDATE_FILTERS_IN_BATCH = guimodel.UPDATE_FILTERS_IN_BATCH;'));
    eval(strcat('model',num2str(recon_layer),'.USE_IPP = guimodel.USE_IPP;'));    
    % Set the saving parameters (changeable in gui)
    eval(strcat('model',num2str(recon_layer),'.PLOT_RESULTS = guimodel.PLOT_RESULTS;'));
    eval(strcat('model',num2str(recon_layer),'.SAVE_RESULTS = guimodel.SAVE_RESULTS;'));
    eval(strcat('model',num2str(recon_layer),'.DISPLAY_ERRORS = guimodel.DISPLAY_ERRORS;'));
    eval(strcat('model',num2str(recon_layer),'.fullsavepath = guimodel.fullsavepath;'));
    eval(strcat('model',num2str(recon_layer),'.fullmodelpath = guimodel.fullsavepath;'));
    eval(strcat('model',num2str(recon_layer),'.fulldatapath = guimodel.fulldatapath;'));
    % Set the noise type and image preprocessing (changeable in gui)
    eval(strcat('model',num2str(recon_layer),'.noisetype = guimodel.noisetype;'));
    eval(strcat('model',num2str(recon_layer),'.CONTRAST_NORMALIZE = guimodel.CONTRAST_NORMALIZE;'));
    eval(strcat('model',num2str(recon_layer),'.COLOR_IMAGES = guimodel.COLOR_IMAGES;'));
    eval(strcat('model',num2str(recon_layer),'.ZERO_MEAN = guimodel.ZERO_MEAN;'));
    % Set training parameters (changeable in gui)
    eval(strcat('model',num2str(recon_layer),'.lambda = guimodel.lambda;'));
    eval(strcat('model',num2str(recon_layer),'.Binitial = guimodel.Binitial;'));
    eval(strcat('model',num2str(recon_layer),'.Bmultiplier = guimodel.Bmultiplier;'));
    eval(strcat('model',num2str(recon_layer),'.betaT = guimodel.betaT;'));
    eval(strcat('model',num2str(recon_layer),'.alpha = guimodel.alpha;'));
    eval(strcat('model',num2str(recon_layer),'.beta_norm = guimodel.beta_norm;'));
    eval(strcat('model',num2str(recon_layer),'.kappa = guimodel.kappa;'));
    eval(strcat('model',num2str(recon_layer),'.alphaF = guimodel.alphaF;'));
    % If you want to updated the images at each iteration
    eval(strcat('model',num2str(recon_layer),'.UPDATE_INPUT = guimodel.UPDATE_INPUT;'));
    eval(strcat('model',num2str(recon_layer),'.lambda_input = guimodel.lambda_input;'));
    % Set z0 parameters (changeable in gui).
    eval(strcat('model',num2str(recon_layer),'.psi = guimodel.psi;'));
    eval(strcat('model',num2str(recon_layer),'.z0_filter_size = guimodel.z0_filter_size;'));
    if(recon_layer == 1) % Only train z0 maps on the first layer.
        eval(strcat('model',num2str(recon_layer),'.TRAIN_Z0 = guimodel.TRAIN_Z0;'));
    end
    % Save the size of the original images (for undoing pooling)
    eval(strcat('model',num2str(recon_layer),'.orig_xdim = size(original_images,1);'));
    eval(strcat('model',num2str(recon_layer),'.orig_ydim = size(original_images,2);'));
    % If you want to update the feature maps with the pooling/unpooling
    eval(strcat('model',num2str(recon_layer),'.SUBSAMPLING_UPDATES = guimodel.SUBSAMPLING_UPDATES;'));
    eval(strcat('model',num2str(recon_layer),'.norm_types = guimodel.norm_types;'));
    eval(strcat('model',num2str(recon_layer),'.norm_sizes = guimodel.norm_sizes;'));
    
    %%%%%%%
    %% Make sure each model is backwards compatible.
    %%%%%%%
    eval(strcat('model',num2str(recon_layer),' = backwards_compatible(model',num2str(recon_layer),');'));    
    %%%%%%%

    

    
    % A string of parameters to pass to each layer.
    modelargs = '';
    % Construct the modelargs string.
    for layer=recon_layer:-1:1
        modelargs = strcat(modelargs,',','model',num2str(layer));
        modelargs = strcat(modelargs,',','F',num2str(layer));
        modelargs = strcat(modelargs,',','z0',num2str(layer));
        modelargs = strcat(modelargs,',','pooled_indices',num2str(layer));        
    end
    % Get rid of the first ',' that is in the string.
    modelargs = modelargs(2:end);
    % Add the input_map (y), original image and the noisy image.
    %     modelargs = strcat(modelargs,',y,original_images,')
    modelargs = strcat(modelargs,',pooled_indices0,y,original_images');
    
    fprintf('Reconstructing Layer %d of a %d-Layer Model\n',recon_layer,top_layer);
    
    % Call the recon_layer with the modelargs parameters.
    % The z0# for the layer is saved.
    % y are the returned (inferred) feature maps (input maps for next layer)
    eval(strcat('[blahF,y,z0',num2str(recon_layer),',y_tilda',num2str(recon_layer),...
        ',model',num2str(recon_layer),'] = train_recon_layer(',modelargs,');'))
    
    % Reset the modelargs sring.
    modelargs = '';
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Normalization Procedure to get the indices need for input to next layer
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % If pooling is required, apply it to the feature maps (y). Then save the results.
    % It is at train_layer+1 since there could be normalization on the input first.
    switch(model1.norm_types{recon_layer+1})
        case 'Max' % Max Pooling
            [pooled_maps,pooled_indices] = max_pool(y,model1.norm_sizes{recon_layer+1});

            if(guimodel.SAVE_RESULTS>0)
                       save(strcat(guimodel.fullsavepath,'layer',num2str(recon_layer),'_pooling.mat'),...
                'pooled_maps','pooled_indices','-v7.3');
            end
            
            % Copy the variables over for the next iteration.
            y = single(pooled_maps);
            % Keep track of each layer's pooled indices.
            eval(strcat('pooled_indices',num2str(recon_layer),' = pooled_indices;'));
            clear pooled_maps pooled_indices
        case 'Avg' % Average Pooling
            [pooled_maps,pooled_indices] = avg_pool(y,model1.norm_sizes{recon_layer+1});

            if(guimodel.SAVE_RESULTS>0)
                        save(strcat(guimodel.fullsavepath,'layer',num2str(recon_layer),'_pooling.mat'),...
                'pooled_maps','pooled_indices','-v7.3');
            end
            
            % Copy the variables over for the next iteration.
            y = single(pooled_maps);
            % Keep track of each layer's pooled indices.
            eval(strcat('pooled_indices',num2str(recon_layer),' = pooled_indices;'));
            clear pooled_maps pooled_indices            
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Special Plotting for Comparisons
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if(guimodel.PLOT_RESULTS>0 || guimodel.SAVE_RESULTS > 0)
%     %%%%%%%%%
%     %% Compile the images together for plotting.
%     start = size(original_images,4);
%     comp = zeros([size(original_images(:,:,:,1)) (start*(2+top_layer))],'single');
%     for i=1:size(original_images,4)
%         comp(:,:,:,i) = original_images(:,:,:,i);
%     end
%     for i=1:size(noisy_images,4)
%         comp(:,:,:,i+start) = noisy_images(:,:,:,i);
%     end
%     for lay=1:top_layer
%         for i=1:size(noisy_images,4)
%             eval(strcat('comp(:,:,:,i+start*(1+lay)) = y_tilda',num2str(lay),'(:,:,:,i);'));
%         end
%     end
%     
%     f = figure(10); clf;
%     sdispims(comp);
%     set(f,'Name','Original,Noisy,Layer 1,2,... Reconstruction (top to bottom)')
%     if(SAVE_RESULTS>0)
%         hgsave(f,strcat(fullsavepath,expfile,'/reconstructions.fig'));
%         if(PLOT_RESULTS==0)
%             close(f) % Only plotted it in order to save it.
%         end
%     end
%     %%%%%%%%%%
%     
%     
%     
%     %%%%%%%%%%
%     %% Error Maps
%     start = size(original_images,4);
%     comp = zeros([size(original_images,1) size(original_images,2) 1 (start*(2+top_layer))],'single');
%     for i=1:size(noisy_images,4) % Noisy reconstruction error.
%         comp(:,:,:,i) = sqrt(sum((original_images(:,:,:,i)-noisy_images(:,:,:,i)).^2,3));
%     end
%     for i=1:size(noisy_images,4)   % Harris corners.
%         comp(:,:,:,i+start) = cornermetric(mean(original_images(:,:,:,i),3));
%     end
%     for lay=1:top_layer % Reconstruction errors of each layer.
%         for i=1:size(noisy_images,4)
%             eval(strcat('comp(:,:,:,i+start*(1+lay)) = sqrt(sum((y_tilda',num2str(lay),'(:,:,:,i) - original_images(:,:,:,i)).^2,3));'));
%             eval(strcat('E',num2str(lay),'(:,:,:,i) = comp(:,:,:,i+start*(1+lay));')); % save the errors.
%         end
%     end
%     for lay=2:top_layer % subtraction of errors compared to lay1
%         for i=1:size(noisy_images,4)
%             eval(strcat('comp(:,:,:,i+start*(2+top_layer+lay-2)) = (E',num2str(lay),'(:,:,:,i) - E1(:,:,:,i));'));
%         end
%     end
%     
%     f = figure(11); clf;
%     sdispims(comp);
%     colormap default
%     set(f,'Name','Noisy Error,Corners,Lay 1,2,... Errors, E2-E1,E3-E1,... (top to bottom)')
%     if(SAVE_RESULTS>0)
%         hgsave(f,strcat(fullsavepath,expfile,'/error.fig'));
%         if(PLOT_RESULTS==0)
%             close(f) % Only plotted it in order to save it.
%         end
%     end
%     
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
