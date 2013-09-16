%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% The entry point to train all desired layers of a Deconvolutional Network
% using the GPUmat library and gpu based convolutions.
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
% @training_file @copybrief gpu_train.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%
%% Start the GPU
GPUstart
%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup the model parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the parameters for the experiment.
% Comment out this line if manually setting parameters below.
clear all

load('./GUI/set_parameters.mat')
% If the previously trained model was on a different machine (different
% paths) then this will convert them to the current machine.
model = convert_paths(model);

%%%%%%%%%
% Set parameters here if not using the gui.
if ~exist('model','var')
    
    'no model parameters set, use defaults'
    
    clear all
    
    maxNumCompThreads(1)
    
    
    % The type of experiment you want to run.
    model.exptype = 'Train Full Model';
    % The actual file to run the experiment at. Leave a space before first char
    model.expfile = ' train';
    
    % The tag is used to say Run##tag folder.
    % model.datadirectory = '../Datasets/Images/2020/';
    % model.tag = '_2020';
    model.datadirectory = '../Datasets/Images/city_patches/';
    model.tag = '_city_patches';
    % model.datadirectory = '../Datasets/Images/singles/2020/';
    % model.tag = '_single_2020';
    % model.datadirectory= '../Datasets/Images/singles/nature_02_small/';
    % model.tag = '_single_nature_02_small';
    % model.datadirectory = '../Datasets/Images/small/';
    % model.tag = '_small';
    % model.datadirectory = '../Datasets/Images/new_100_100/';
    % model.tag = '_new_100_100';
    % model.datadirectory = '../Datasets/Images/notnature_100_100/';
    % model.tag = '_notnature_100_100';
    % model.datadirectory = '../Datasets/Images/patch_256_256/';
    % model.tag = '_patch_256_256';
    % model.datadirectory = '../Datasets/Images/120_160/';
    % model.tag = '_120_160';
    % model.datadirectory = '../Datasets/Images/240_320/';
    % model.tag = '_240_320'
    % Parameters for the image creation.
    model.VARIANCE_THRESHOLD = 0; % real number
    model.CONTRAST_NORMALIZE = 1; % binary
    model.COLOR_IMAGES = 1; % binary
    % Threshold for the gradients.
    model.grad_threshold = 0.01;
    % Size of each filter
    model.filter_size = 7;
    % Number of feature maps, z,  to learn
    model.num_feature_maps = 9;
    % Make the z0 filter size the same
    model.z0_filter_size = model.filter_size;
    % Sparsity Weighting
    model.lambda = 2;
    % This adds the lambda = lambda + ramp_lambda_amount
    model.RAMP_LAMBDA = 0;
    model.ramp_lambda_amount = 0.5;
    % For the dummy variables, here are the parameters for ramping beta
    model.Binitial = 10^-5; %start value
    model.Bmultiplier = 10; %beta = beta*Bmultiplier each iteration
    model.betaT = 12;       %number of iterations, also used as T in IRLS as the number of updates to the values of z and F overall (outer loop)
    model.RAMP_DOWN_AND_UP = 0; %switch to beta=beta/Bmultiplier after half way
    % The alpha-norm on the dummy regularizer variables
    model.alpha = 1;
    % The alpha-normalization on the F's.
    model.alphaF = 2;
    % The coefficient on the filter L2 regularization
    model.kappa = 0;
    % The coefficient on the gradient(z0) term when training z0.
    model.psi = 1;
    % Number of epochs total through the training set.
    model.maxepochs = 25;
    % Number of iterations to run minimize
    model.min_iterations = 2;
    % If this is set to 1, then randomize the order in which the images are
    % selecdted from the training set. Random order changes at each epoch.
    % If this is set to 0, then they are picked in order.
    model.RANDOM_IMAGE_ORDER = 1;
    % Filters updates after all samples update z.
    model.UPDATE_FILTERS_IN_BATCH = 0;
    % Save the results when set greater than 0. This is number of epochs
    % between which to save.
    model.SAVE_RESULTS = 5;
    % Set this to 1 to show the resulting z,F,original images, etc. plots.
    model.PLOT_RESULTS = 1;
    
else % Parameters are set in the model structure
    
    % Shouldn't have to do anything here.
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%
%% Set number of computation threads to use.
maxNumCompThreads(model.comp_threads);
%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare for Layer 1: Get Data Ready
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the inputs to layer 1.
% if(model.COLOR_IMAGES)
%     [y,good_ind,xdim,ydim] = CreateColorImages(model.fulldatapath,model.CONTRAST_NORMALIZE,model.VARIANCE_THRESHOLD);
% else
%     [y,good_ind,xdim,ydim] = CreateGrayImages(model.fulldatapath,model.CONTRAST_NORMALIZE,model.VARIANCE_THRESHOLD);
% end
% the first two dimensions of y1 are the x and y dimensions of each image.
% the 3rd dimension of y is the number of input maps.
% the 4th dimensions of y is the number of training samples.

%Normalize the input to the first layer to be zero mean and unit variance.
% model.mean_y1 = mean(y1(:));
% y1 = y1-model.mean_y1; % Shift up by the smallest negative number to make all numbers positive.
% model.std_y1 = std(y1(:));
% y1 = y1./model.std_y1;  % Scale them so max element goes to 1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Loop over each layer you want to train.
for train_layer=1:model.num_layers
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Train Layer 1: Everything is planar going into/out of the train_layer
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Specify which layer you are learning for saving the results.
    model.layer = train_layer;
    
    if(train_layer == 1)
        % Overwrite contrast normalize to 0 sinze not needed if using z0.
        if(model.TRAIN_Z0 == 1)
            model.CONTRAST_NORMALIZE = 0; % THE ONLY REASON FOR Z0 IS THIS.
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Perpare for Layer 1: Get Data Ready
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get the inputs to train_layer 1.
        if(model.COLOR_IMAGES)
            [original_images,good_ind,xdim,ydim] = CreateColorImages(model.fulldatapath,model.CONTRAST_NORMALIZE,model.VARIANCE_THRESHOLD);
        else
            [original_images,good_ind,xdim,ydim] = CreateGrayImages(model.fulldatapath,model.CONTRAST_NORMALIZE,model.VARIANCE_THRESHOLD);
        end
        original_images = GPUsingle(original_images);
        % For the first layer the input maps are the original images.
        y = original_images;
    else % Set to zero for other train_layers.
        model.TRAIN_Z0 = 0;
    end
    
    
    
    % xdim and ydim are the size of the input layers which are vectorized into
    fprintf('Training Layer %d of a %d-Layer Model\n',model.layer,model.num_layers);
    fprintf('Number of Input Maps = %d, Number of Feature Maps = %d\n',model.num_input_maps(train_layer),model.num_feature_maps(train_layer))
    fprintf('The connectivity map is:\n');
    %% Create the connectivity matrix.
    % C(j,k) of this connectivity matrix is 1 if there is a connection from
    % feature map k down to input map j.
    disp(model.conmats{model.layer})   % Display the matrix
    
    %     % Trains a single layer in the heirarchy.
    %     % y is the input and parameters are set in model.
    %     % y is also the output since that is the input for the next layer.
    %     [F,y,z0] = train_layer(model,y);
    %     % The return values are in planar form.
    
    % Save the model as model# where # is the layer.
    eval(strcat('model',num2str(train_layer),'=model;'));
    % Since training, initialize F1 and z1 == 0 to start.
    eval(strcat('F',num2str(train_layer),' = 0;'));
    eval(strcat('z0',num2str(train_layer),' = 0;'));
    
    % A string of parameters to pass to each layer.
    modelargs = '';
    % Construct the modelargs string.
    for layer=train_layer:-1:1
        modelargs = strcat(modelargs,',','model',num2str(layer));
        modelargs = strcat(modelargs,',','F',num2str(layer));
        modelargs = strcat(modelargs,',','z0',num2str(layer));
    end
    % Get rid of the first ',' that is in the string.
    modelargs = modelargs(2:end);
    % Add the input_map (y), original images and
    modelargs = strcat(modelargs,',y,original_images,',char(39),'train',char(39));
    
    
    eval(strcat('[F',num2str(train_layer),',y,z0',num2str(train_layer)',...
        ',y_tilda',num2str(train_layer),'] = gpu_train_recon_layer(',modelargs,');'));
    
    % Reset the modelargs sring.
    modelargs = '';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end





