%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is where you setup all the defaults in the model struct. If you are not
% using a GUI to set them, this is the place to modify the model parameters.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @defaults_file @copybrief set_parameters_here.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set Model Parameters/Defaults
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The number of computation threads to use
model.comp_threads = 4;

%%%%%%%%%%
%% IPP Libraries (use for speed!)
%%%%%%%%%%
% Whether or not to use IPP libraries (much faster if you can use this but requires the IPPConvsToolbox from www.matthewzeiler.com/software/).
model.USE_IPP = 1;
if(~exist(strcat('ipp_conv2.',mexext),'file'))
    fprintf('You do not have the compiled versions of the IPP Convolutions Toolbox therefore reverting to slower MATLAB only implementation.\n')
    model.USE_IPP = 0;
end
%%%%%%%%%%


%%%%%%%%%%
%% Plotting and Saving Directories
%%%%%%%%%%
% If >0 then results will be plotted every PLOT_RESULTS many epochs in the various forms. 
% (set to zero for speed if you just want to save the results instead of view online).
% Note: the plots lock to different locations on the screen initially but can be
% dragged anywhere afterwards. This locking was designed for an (1920x1080) or
% higher resolution screen.
model.PLOT_RESULTS = 1;
% If >0 then results will be saved every SAVE_RESULTS many epochs in the format epoch#_layer#.mat
model.SAVE_RESULTS = 5;
% If >0 then error function evaluation will be printed out (faster if you don't) every DISPLAY_ERROR many epochs.
% THIS IS AUTOMATICALLY SET WHEN RECONSTRUCTING as it is needed to display the reconstructed results as well.
model.DISPLAY_ERRORS = 0;

% Where to save the results (will be appended to save various things if SAVE_RESULTS>1)
% Note this is based on the current directory by defuault, you'll likely want to change this.
model.fullsavepath = './Results/fruit_100_100/Run_1/';
% Path to a folder containing only image files (and maybe other folder but it 
% will not be checked recursively) for training or to reconstruct.
model.fulldatapath = './Datasets/Images/';
% Set this to the last folder
model.tag = 'fruit_100_100';
% model.fulldatapath = '/Datasets/Images/city_100_100/';
% model.tag = 'city_100_100';
% Example for reconstructing.
% model.fulldatapath = '/Datasets/Images/singles/test1/';

% Used for Reconstruction: this is a path to a previously trained model used
% for reconstructing an image (this is a file but don't both with the .mat extension).
% Note: this is ignored during training.
model.fullmodelpath = './Results/fruit_100_100/Run_1/epoch5_layer1.mat';
%%%%%%%%%%



%%%%%%%%%%
%% Preprocessing
%%%%%%%%%%
% Parameters for the image creation/preprocessing.
model.ZERO_MEAN = 1;          % binary (subtract mean or not)
model.CONTRAST_NORMALIZE = 1; % binary (constrast normalize images or not)
model.COLOR_IMAGES = 'rgb';   % string: 'gray', 'rgb', 'ycbcr', 'hsv'
model.SQUARE_IMAGES = 1;      % binary (square images or not)
%%%%%%%%%%

%%%%%%%%%%
%% Training Adjustments
%%%%%%%%%%
% Filters updates after all samples update z (doesn't work well).
model.UPDATE_FILTERS_IN_BATCH = 0;
% Batch size used in training (leave at 1 as batching doesn't work well).
model.batch_size = 2;

% If this is set to 0, then they are picked in order (1 is best).
model.RANDOM_IMAGE_ORDER = 1;

% Number of epochs per layer through the training set (5 is usually sufficient).
model.maxepochs = [7 1 1 5];
% Number of steps of conjugate gradient used when updating filters and feature maps at each iteration (2 is best).
model.min_iterations = 2;
% Threshold for the gradients.
model.grad_threshold = 0.01;

% For Yann's inference scheme, if you want to train the 1st layer initially.
model.LAYER1_FIRST = 0;
%%%%%%%%%%

%%%%%%%%%%
%% Model Structure
%%%%%%%%%%
% Number of layers total in the model you want to train. Note: below you will
% see many variables defined for a 4 layer model. This was just for convenience
% and only the first num_layers of them are used for the layers. 4 was just
% choosen as a reasonable size as well, but the code will accept any number of
% layer (though not tested) if these arrays of parameters below are extended.
model.num_layers =2;
% Size of each filters in each layer (assumes square). (7 is good).
model.filter_size = [7 7 9 9];
% Number of feature maps in each layer. (this is the defualt).
% model.num_feature_maps = [128 100 500 500];
% model.num_feature_maps = [50 1275 500 500];
% model.num_feature_maps = [45 1035 500 500];
% model.num_feature_maps = [40 820 500 500];
% model.num_feature_maps = [35 630 500 500];
% model.num_feature_maps = [30 465 500 500];
% model.num_feature_maps = [25 325 400 500];
% model.num_feature_maps = [20 210 250 250];
% model.num_feature_maps = [15 120 250 150];
 model.num_feature_maps = [9 45 150 150];
% Number of input maps in the first layer (do not modify this).
if(strcmp(model.COLOR_IMAGES,'ycbcr') || strcmp(model.COLOR_IMAGES,'rgb') || strcmp(model.COLOR_IMAGES,'hsv'))
    num_input_maps = 3;
else
    num_input_maps = 1;
end
% Number of input maps in all layers (do not modify this).
model.num_input_maps = [num_input_maps model.num_feature_maps(1) model.num_feature_maps(2) model.num_feature_maps(3)];
% The default types for the connectivity matrix (from cvpr2010)
model.conmat_types = {'Full','Singles and All Doubles','Random Doubles','Random Doubles'};
% Initialize connectivity matrices to the desired defaults.
[model] = update_conmats(model);
%%%%%%%%%%

%%%%%%%%%%
%% Learning parameters
%%%%%%%%%%
% Reconstruction term lambda*||sum(conv(z,f)) - y||^2 weighting for each layer (1 works well).
model.lambda = [1 1 1 1];
% The alpha-norm on the auxilary variable kappa*|x|^alpha for each layer.
model.alpha = [0.8 0.8 1 1];
% A coefficient on the filter L2 regularization (leave at 1, not in papers)
model.kappa = [1 1 1 1];

% The regeme over the continutaion variable, beta.
model.Binitial = 1;     %start value
model.Bmultiplier = 10; %beta = beta*Bmultiplier each iteration
model.betaT = 6;        %number of iterations
%%%%%%%%%%%

%%%%%%%%%%%
% z0 map parameters, still fairly experimental.
%%%%%%%%%%%
% Make the z0 filter size the same as the first layer's by default.
model.z0_filter_size = model.filter_size(1);
% The coefficient on the gradient(z0) term when training z0. (this may vary).
model.psi = 1;
% Check to determine if you even want to train the z0 map (while training z and f).
model.TRAIN_Z0 = 0;
%%%%%%%%%%%

%%%%%%%%%%
%% Experimental Features (do not modify)
%%%%%%%%%%
% Noise, leave this as 'none' for no noise.
model.noisetype = 'None (Reconstruct)';
% If you want to update the y variable when reconstructing.
model.UPDATE_INPUT = 0;
% Lambda for the reconstruction error of the updated and input images.
model.lambda_input = 1;

% Normalization type for each layer and the input image (first one).
model.norm_types = {'None','None','None','None','None'};
% The size of the pooling regions.
model.norm_sizes = {[2 2],[2 2],[2 2],[2 2],[2 2]};
% If you want to loop and unloop the pooling operation at each iteration.
model.SUBSAMPLING_UPDATES = 0;
%%%%%%%%%%

%%%%%%%%%%
%% Not Used (do not modify, kept in for backwards compatibility).
%%%%%%%%%%
% This adds the lambda = lambda + ramp_lambda_amount at each iteration.
model.RAMP_LAMBDA = 0;
model.ramp_lambda_amount = 0.5;
model.RAMP_DOWN_AND_UP = 0; %switch to beta=beta/Bmultiplier after half way
% The beta-norm on the dummy variable - feature map clamping.
model.beta_norm = 2;
% The alpha-normalization on the F's.
model.alphaF = 2;
%%%%%%%%%%

%%%%%%%%%%
%% GUI Specific fields (not included so do not modify)
%%%%%%%%%%
% The type of experiment you want to run (used by the gui).
model.exptype = 'Train Layer By Layer';
% The actual file to run the experiment at. Leave a space before first char
model.expfile = ' train';
% The location where the job was run from. Leave a space before first char.
model.machine = ' laptop';

% The dataset directory (where to load images from for training/reconstruction).
model.datadirectory = '/Datasets/Images/fruit_100_100/';
% model.tag = 'fruit_100_100';
% model.datadirectory = '/Datasets/Images/city_100_100/';
% model.tag = 'city_100_100';
% model.datadirectory = '/Datasets/Images/singles/test1/';

% The model directory (used when loading a previous model).
model.modeldirectory = '/Results/train/city_100_100/color_filters_9_45/Run_0/epoch25_layer1';
% Where to save the results.
model.savedirectory = '/Results/+';
%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%