%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a combined function to both train or reconstruct images
% in the pixel space. The type of operation is determined by model.exptype
% Reconstructs a single layer (specified by model.layer) of the model
% based on the input y. y is the input maps of model.layer which maybe be
% the feature maps of a layer below. The feature maps are inferred from
% this and then used to reconstruct a y' reconstruction of the input maps.
% Inputs: The inputs are passed in varargin (a cell array) because when
% reconstructing from higher layers, all the lower layers need to be passed
% in to reconstruct down to the lowest level.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @training_file @copybrief train_recon_layer.m
% @recon_file @copybrief train_recon_layer.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief train_recon_layer.m
%
% @param varargin should be of the following format: (example for 2 layers)
% \li \e model2 struct with parameters for top layer
% \li \e F2 third layer filters or []
% \li \e z02 third layer z0 maps or []
% \li \e pooled_inpdices2 the indices from Max pooling after L2 (usually not every used) to allow reconstructions
% \li \e model1 struct with parameters for layer 1
% \li \e F1 layer 1 filters
% \li \e z02 fist layer z0 maps or []
% \li \e pooled_inpdices1 the indices from Max pooling after L1 (usually notevery used) to allow reconstructions
% \li \e pooled_inpdices0 the indices from Max pooling on the image (usually not every used) to allow reconstructions
% \li \e y the input maps for the given layer (may be noisy if denoising)
% \li \e original_images the clean images (will be identical to y when training on clean images)
%
% @retval F the learned (or previous if reconstructing) filters.
% @retval z the inferred feature maps.
% @retval z0 the inferred z0 feature maps (or [] if not used).
% @retval recon_images the reconstructed images (required DISPLAY_ERROR to be
% set).
% @retval model some fields of the model structure will be updated within (with
% xdim, ydim, and errors for example).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [F,z,z0,recon_images,model] = train_recon_layer(varargin)


% Seed the ramdon numbers based on time.
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read in the variables arguments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The layer to be inferred is the first on input.
layer = varargin{1}.layer;
% Set up the top layer's variables.
model = varargin{1};
F = varargin{2};
% z0 = varargin{3};

maxNumCompThreads(model.comp_threads);
COMP_THREADS = model.comp_threads;

% Get the model,F,z0 triples
% The actual model that is being inferred is stored in model,F,z0.
% In addition, all the layers are stored in model#,z#,z0#
for i=1:layer
    eval(strcat('model',num2str(layer-i+1),'=varargin{(i-1)*4+1};'));
    eval(strcat('F',num2str(layer-i+1),'=varargin{(i-1)*4+2};'));
    eval(strcat('z0',num2str(layer-i+1),'=varargin{(i-1)*4+3};'));
    % Note this may not be set during training for the top layer.
    eval(strcat('pooled_indices',num2str(layer-i+1),'=varargin{(i-1)*4+4};'));
end

% Set the pooling indices on the input image.
pooled_indices0 = varargin{end-2};
% Get the input maps for this layer. (for teh first layer this is the same
% as the noisy input image planes).
y = varargin{end-1};
% Get the original image.
original_images = varargin{end};
% train_recon is either 'train' or 'recon' depending on what you want to do
% train_recon = varargin{end};
%train_recon = 'train';

% Setup sizes properly (since input image maybe be different than what was
% used during training of this model.
model.xdim = size(y,1); % Assuming square images.
eval(strcat('model',num2str(layer),'.xdim = size(y,1);'));
xdim = model.xdim;
model.ydim = size(y,2); % Assuming square images.
eval(strcat('model',num2str(layer),'.ydim = size(y,2);'));
ydim = model.ydim;
model.num_input_maps(layer) = size(y,3); % Set the number of input maps here.

input_maps = model.num_input_maps(model.layer);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters. DO NOT CHANGE HERE. Change in the gui.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Size of each filter
filter_size = model.filter_size(model.layer);
% Number of filters to learn
num_feature_maps = model.num_feature_maps(model.layer);

% Number of epochs total through the training set.
maxepochs = model.maxepochs(model.layer);
% Number of iterations to run minimize
min_iterations = model.min_iterations;
% Theshold to control the gradients
% grad_threshold = model.grad_threshold;

% Sparsity Weighting
lambda = model.lambda(model.layer);
UPDATE_INPUT = model.UPDATE_INPUT;
lambda_input = model.lambda_input;
% Use 0.01, batch, and put lambda = lambda + 0.1 to see that the first filters just take patches.
% RAMP_LAMBDA = model.RAMP_LAMBDA;
% ramp_lambda_amount = model.ramp_lambda_amount;
% Dummy regularization weighting. This starts at beta=Binitial, increases
% by beta*Bmultiplier each iteration until T iterations are done.
Binitial = model.Binitial;
Bmultiplier = model.Bmultiplier;
betaT = model.betaT;
beta = Binitial;
% % See if ramping the beta back down after half of T's iterations helps.
% RAMP_DOWN_AND_UP = model.RAMP_DOWN_AND_UP;
% The alpha-norm on the dummy regularizer variables
alpha = model.alpha(model.layer);

% The alpha-norm on the filters
% alphaF = model.alphaF;
% The coefficient on the sparsity term in for each layer.
kappa = model.kappa;
%XXXXXXXXXXXXXXXXXXXXXXXXXX
% Not implemented yet
% The normalization on the ||z-x|| dummy variable term.
% beta_norm = model.beta_norm;
%XXXXXXXXXXXXXXXXXXXXXXXXXX

% Connectivity Matrix, this is a input_maps by num_feature_maps matrix.
connectivity_matrix = model.conmats{model.layer};

% If the  z0 map is used at this layer.
TRAIN_Z0 = model.TRAIN_Z0;

% If this is set to 1, then randomize the order in which the images are
% selecdted from the training set. Random order changes at each epoch.
% If this is set to 0, then they are picked in order.
RANDOM_IMAGE_ORDER = model.RANDOM_IMAGE_ORDER;
% Fitlers are set after each batch while z's are every image sample.
UPDATE_FILTERS_IN_BATCH = model.UPDATE_FILTERS_IN_BATCH;
% No point in batching data when not updating the filters in batch.
if(UPDATE_FILTERS_IN_BATCH == 0)
    batch_size = size(y,4);
else
    batch_size = model.batch_size;
end

% For saving the results.
layer = model.layer;

% Set this to 1 if you want to store the Filter matrices.
SAVE_RESULTS = model.SAVE_RESULTS;
fullsavepath = model.fullsavepath;
% If you want to plot the results.
PLOT_RESULTS = model.PLOT_RESULTS;



% Setup for saving the figures.
switch(model.expfile)
    case ' train_remaining_layers'
        expfile = 'train';
        train_recon = 'train';
    case 'train_remaining_layers'
        expfile = 'train';
        train_recon = 'train';
    case ' train'
        expfile = model.expfile;
        train_recon = 'train';
    case 'train'
        expfile = model.expfile;
        train_recon = 'train';
    case ' recon'
        expfile = model.expfile;
        train_recon = 'recon';
    case 'recon'
        expfile = model.expfile;
        train_recon = 'recon';
    otherwise
        expfile = model.expfile;
        train_recon = 'train';
end

if(strcmp(expfile(1),' '))
    expfile = expfile(2:end);
end

% If you want ot use the IPP libraries or not.
USE_IPP = model.USE_IPP;

% If reconstructing always show the errors.
if(strcmp(train_recon,'recon'))
    DISPLAY_ERRORS = 1;
else
    DISPLAY_ERRORS = model.DISPLAY_ERRORS;
end

% Makes directory to save figures in.
if(SAVE_RESULTS>0)
    mkdir2(strcat(fullsavepath,expfile))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing of the images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine the size of the inputs.
% Just use one batch because things are done in order anyways.
% batch_size = size(y,4);
% This is 3 for color images, but may be more for higher level
% representations.
% input_maps = size(y,2);

% This calculates the number of mini-batches, this will be 1.
number_of_batches = max(floor(size(y,4)/batch_size),1);

% Ignore the batches that would have less than full batch_size
% y = y(:,:,1:input_maps,1:batch_size*number_of_batches);
y_input = y;

noisy_SNR = zeros(1,size(y,4),'single');
% Compute SNR values again noisy image (y) and original image
if(layer==1)
    % Check if the input image was normalized.
    switch(model.norm_types{1})
        case 'Max'  % Max Pooling
            % Reverse the max pooling.
            for i=1:size(y,4)
                noisy_SNR(i) = compute_snr(original_images(:,:,:,i),reverse_max_pool(y(:,:,:,i),pooled_indices0(:,:,:,i),model.norm_sizes{1},[model.orig_xdim model.orig_ydim]));
            end
        case 'Avg'  % Avereage Pooling
            % Reverse the max pooling.
            for i=1:size(y,4)
                noisy_SNR(i) = compute_snr(original_images(:,:,:,i),reverse_avg_pool(y(:,:,:,i),pooled_indices0,model.norm_sizes{1},[model.orig_xdim model.orig_ydim]));
            end
        case 'None'
            for i=1:size(y,4) % For each image
                noisy_SNR(i) = compute_snr(original_images(:,:,:,i),y(:,:,:,i));
            end
    end
else
    for i=1:size(y,4) % For each image
        noisy_SNR(i) = 0;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the input images (or maps for higher layers).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(PLOT_RESULTS>0)
    %     if(layer==1)
    f = figure(100+model.layer); clf;
    if(input_maps ==1 || input_maps == 3)
        sdispims(y);
    else
        sdispmaps(y);
    end
    set(f,'Name',strcat('Layer ',num2str(model.layer),' Input Maps'));
    drawnow
    cursize = get(f,'Position');
    screensize = get(0,'Screensize');
    if(screensize(4) < 1200)
        set(f,'Position',[850-cursize(3),30,cursize(3),cursize(4)])
    else
        set(f,'Position',[850-cursize(3),900,cursize(3),cursize(4)])
    end
    drawnow;
    
    if(strcmp(train_recon,'recon'))
        f = figure(100); clf;
        if(input_maps ==1 || input_maps == 3)
            sdispims(original_images);
        else
            sdispmaps(original_images);
        end
        set(f,'Name',strcat('Layer ',num2str(model.layer),' Input Maps'));
        drawnow
        cursize2 = get(f,'Position');
        screensize = get(0,'Screensize');
        if(screensize(4) < 1200)
            set(f,'Position',[850-cursize(3)-cursize2(3),30,cursize2(3),cursize2(4)])
        else
            set(f,'Position',[850-cursize(3)-cursize2(3),900,cursize2(3),cursize2(4)])
        end
        drawnow;
    end
    
    if(SAVE_RESULTS>0)
        %         if(strcmp(train_recon,'recon'))
        %             hgsave(f,strcat(fullsavepath,'input_maps_layer',num2str(layer),'.fig'));
        %         else
        mkdir2(strcat(fullsavepath,expfile));
        hgsave(f,strcat(fullsavepath,expfile,'/input_maps_layer',num2str(layer),'.fig'));
        %         end
        if(PLOT_RESULTS==0)
            close(f) % Only plotted it in order to save it.
        end
    end
    %     end
    
    f = figure(150+model.layer); clf;
    hist(y(:),1000);
    title(strcat('Num elements in y: ',num2str(numel(y)),' Max bin size: ',num2str(max(hist(y(:),1000)))));
    set(f,'Name',strcat('Layer ',num2str(model.layer),' Input Histogram'));
    screensize = get(0,'Screensize');
    set(f,'Position',[0,screensize(4)-400,400,300])
    drawnow;
    if(SAVE_RESULTS>0)
        %         if(strcmp(train_recon,'recon'))
        %             hgsave(f,strcat(fullsavepath,'input_hist_layer',num2str(layer),'.fig'));
        %         else
        hgsave(f,strcat(fullsavepath,expfile,'/input_hist_layer',num2str(layer),'.fig'));
        if(PLOT_RESULTS==0)
            close(f) % Only plotted it in order to save it.
        end
        %         end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAINING PHASE SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(strcmp(train_recon,'train'))
    % Initialize the F matrix to random number (0,1)
    % This F matrix stores all the filters as columns.
    % There are filter_size*filter_size by input_maps by num_feature_maps
    F = randn(filter_size,filter_size,input_maps,num_feature_maps,'single');
    % Normalize the planes of F
    F = plane_normalize(F); % XXX
end


% Initialize the feature maps to be random values.
% Thus z is xdim+/- x ydim+/- x num_feature_maps x num_samples
z =zeros((xdim+filter_size-1),(ydim+filter_size-1),num_feature_maps,size(y,4),'single');

% If using the z0 for the given layer.
if(TRAIN_Z0)
    z0_filter_size = model.z0_filter_size;
    psi = model.psi;
    % If more than just z0=0, then can use that z0.
    if(ndims(varargin{3})<3)
        z0 = 0.01*randn((xdim+z0_filter_size-1),(ydim+z0_filter_size-1),input_maps,size(y,4),'single');
    else % Just use the z0 passed in.
        z0 = varargin{3};
    end
else % Just set z0=0 if not used.
    z0=0;
    z0_filter_size = 1;
    psi = 0;
end
% Get rid of the input parameters (to save memory)
clear varargin;

% Introduce the dummy variables w, same size as z.
w = z;

% Initialize a matrix to store the reconstructed images.
recon_images = zeros(size(original_images),'single');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%
%% Video stuff
%%%%%%%%%%%%%%%
frame_num = 1; % For original filters.
pframe_num = 1; % For pixel space filters.
VIDEO_FRAMES = 0;

if(layer==1)
    if(PLOT_RESULTS ||SAVE_RESULTS || VIDEO_FRAMES)
        % Display the filters
        f = figure(200+model.layer); clf;
        sdispfilts(F,model.conmats{model.layer});
        set(f,'Name',strcat('Layer ',num2str(model.layer),' Filters'));
        drawnow
        if(PLOT_RESULTS==0)
            cursize = get(f,'Position');
            screensize = get(0,'Screensize');
            set(f,'Position',[800,screensize(4)-cursize(4)-100,cursize(3),cursize(4)])
        end
        drawnow;
        
        if(SAVE_RESULTS>0 || VIDEO_FRAMES)
            %% Setup a path to save to /fullsavepath/train/epoch##/layer#/
            savepath = strcat(fullsavepath,expfile,'/epoch',num2str(0),'/layer',num2str(layer));
            mkdir2(savepath);
            
            
            if(VIDEO_FRAMES)
                videopath = strcat(fullsavepath,expfile,'/video_layer',num2str(layer));
                mkdir2(videopath);
                title(sprintf('Frame %d',frame_num));
                if(frame_num<10)
                    saveas(f,strcat(videopath,'/fig000',num2str(frame_num)),'png');
                elseif(frame_num<100)
                    saveas(f,strcat(videopath,'/fig00',num2str(frame_num)),'png');
                elseif(frame_num<1000)
                    saveas(f,strcat(videopath,'/fig0',num2str(frame_num)),'png');
                elseif(frame_num<10000)
                    saveas(f,strcat(videopath,'/fig',num2str(frame_num)),'png');
                end
                frame_num = frame_num+1;
            end
            
            hgsave(f,strcat(savepath,'/original_filters.fig'));
            
            
            if(PLOT_RESULTS==0)
                close(f) % Only plotted it in order to save it.
            end
        end
    end
else
    if(PLOT_RESULTS || SAVE_RESULTS || VIDEO_FRAMES)
        
        % Display the pixel space filters for higher layers.
        if(layer > 1)
            if(TRAIN_Z0)
                eval(strcat('z0',num2str(layer),' = z0sample;'));
            end
            eval(strcat('F',num2str(layer),' = F;'));
            
            savepath = strcat(fullsavepath,expfile,'/epoch',num2str(0),'/layer',num2str(layer));
            mkdir2(savepath);
            
            eval(strcat('recon_z',num2str(layer),' = z(:,:,:,1);'));
            
            
            top_down_noload  % Generate the pixel space filters.
            f = figure(250+model.layer); clf;
            sdispims(recon_y1);
            set(f,'Name',strcat('Layer ',num2str(model.layer),' Pixel Filters'));
            drawnow
            if(PLOT_RESULTS==0)
                cursize = get(f,'Position');
                screensize = get(0,'Screensize');
                if(screensize(4)<1200)
                    set(f,'Position',[1750,30,cursize(3),cursize(4)])
                else
                    set(f,'Position',[1750,900,cursize(3),cursize(4)])
                end
            end
            drawnow;
            if(SAVE_RESULTS>0 || VIDEO_FRAMES)
                
                if(VIDEO_FRAMES)
                    videopath = strcat(fullsavepath,expfile,'/video_layer',num2str(layer));
                    mkdir2(videopath);
                    title(sprintf('Frame %d',pframe_num));
                    if(frame_num<10)
                        saveas(f,strcat(videopath,'/pfig000',num2str(pframe_num)),'png');
                    elseif(frame_num<100)
                        saveas(f,strcat(videopath,'/pfig00',num2str(pframe_num)),'png');
                    elseif(frame_num<1000)
                        saveas(f,strcat(videopath,'/pfig0',num2str(pframe_num)),'png');
                    elseif(frame_num<10000)
                        saveas(f,strcat(videopath,'/pfig',num2str(pframe_num)),'png');
                    end
                    pframe_num = pframe_num+1;
                end
                
                hgsave(f,strcat(savepath,'/pixel_filters.fig'));
                
                if(PLOT_RESULTS==0)
                    close(f) % Only plotted it in order to save it.
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear the error matrices (especially for reconstruction)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.update_noise_rec_error = [];
model.pix_noise_rec_error = [];
model.pix_clean_rec_error = [];
model.pix_clean_SNR_error = [];
model.pix_update_rec_error = [];
model.reg_error = [];
model.beta_rec_error = [];
model.unscaled_total_energy = [];
model.scaled_total_energy = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop through the entire training set this number of times.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for epoch = 1:maxepochs
    tic
    % Make the training images appear at ramdon by permuting their indexes
    if RANDOM_IMAGE_ORDER
        permbatchindex = randperm(size(y,4));
    else
        permbatchindex = 1:size(y,4);
    end
    
    total_image_num = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop over mini-batches.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for batch = 1:number_of_batches
        
        % Get the start and end index for the given batch.
        start_of_batch = (batch-1)*batch_size+1;
        % The last batch should contain all the remaining images.
        if(batch == number_of_batches)
            end_of_batch = length(permbatchindex);
        else
            end_of_batch = batch*batch_size;
        end
        batch_indices = permbatchindex(start_of_batch:end_of_batch);
        
        % Keep track of the errors for the whole batch.
        image_num = 0;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Go through each image in the batch.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for sample=batch_indices
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Setups each sample's variables
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            image_num = image_num+1; % Count the number of images processed.
            total_image_num = total_image_num+1; % For not just the batch.
            
            % Get only those values for the current training sample.
            zsample = z(:,:,:,sample); % imagesize x num_feature_maps matrix
            ysample = y(:,:,:,sample); % imagesize column.
            wsample = w(:,:,:,sample);
            if(TRAIN_Z0)
                z0sample = z0(:,:,:,sample);
            else
                z0sample = 0;
            end
            
            if(mod(epoch,DISPLAY_ERRORS)~=0)
                fprintf('Layer: %1d, Epoch: %2d, Batch %2d, Image: %2d (%d/%d) ',model.layer,epoch,batch,sample,total_image_num,size(y,4));
            else
                fprintf('Layer: %1d, Epoch: %2d, Batch %2d, Image: %2d (%d/%d) ',model.layer,epoch,batch,sample,total_image_num,size(y,4));
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Beta Regeme
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for beta_iteration = 1:betaT
                if beta_iteration == 1
                    beta = Binitial;
                elseif beta_iteration == betaT
                    fprintf('.\n')
                    beta = beta*Bmultiplier;
                else
                    fprintf('.')
                    beta = beta*Bmultiplier;
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Update the w values based on the current sample of z.
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                wsample(:) = solve_image(zsample(:),beta,alpha,kappa(layer)); % Generic code that does all the below.
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Update Feature Maps
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if(~USE_IPP)
                    zsample = fast_infer(min_iterations,zsample,wsample,ysample,F,z0sample,z0_filter_size,lambda,beta,connectivity_matrix,TRAIN_Z0);
                else
                    zsample = ipp_infer(min_iterations,zsample,wsample,ysample,F,z0sample,z0_filter_size,lambda,beta,connectivity_matrix,TRAIN_Z0,COMP_THREADS);
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Update z0 Feature Maps
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if(TRAIN_Z0)
                    if(~USE_IPP)
                        z0sample(:) = fast_z0(min_iterations,zsample,ysample,F,z0sample,z0_filter_size,lambda,connectivity_matrix,psi);
                    else
                        z0sample(:) = ipp_z0(min_iterations,zsample,ysample,F,z0sample,z0_filter_size,lambda,connectivity_matrix,psi,COMP_THREADS);
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Subsample and Inverse Subsample Feature Maps
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(model.SUBSAMPLING_UPDATES)
                switch(model.norm_types{layer+1})
                    case 'Max'  % Pool and unpool the current feature maps
                        fprintf('Max pooling and unpooling\n');
                        zsample = loop_max_pool(zsample,model.norm_sizes{layer+1});
                    case 'Avg'  % Pool and unpool the current feature maps
                        fprintf('Avg pooling and unpooling\n');
                        zsample = loop_avg_pool(zsample,model.norm_sizes{layer+1});
                    case 'None'
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Update Filters
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(strcmp(train_recon,'train'))
                if UPDATE_FILTERS_IN_BATCH == 0
                    if((model.CONTRAST_NORMALIZE == 1 || model.COLOR_IMAGES == 0) && model.layer == 1)
                        filter_min = min_iterations;
                        model.min_iterations = filter_min;
                    else
                        filter_min = min_iterations*2;
                        model.min_iterations = filter_min;
                    end
                    
                    if(~USE_IPP)
                        F = fast_learn_filters(min_iterations,zsample,ysample,F,z0sample,z0_filter_size,lambda,connectivity_matrix,TRAIN_Z0);
                    else
                        F = ipp_learn_filters(filter_min,zsample,ysample,F,z0sample,z0_filter_size,lambda,connectivity_matrix,TRAIN_Z0,COMP_THREADS);
                    end
                    % Normalize the columns
                    F = plane_normalize(F);
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Copy the updated z's for the sample back into the z and w.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            z(:,:,:,sample) = zsample;
            w(:,:,:,sample) = wsample;
            if(TRAIN_Z0)
                z0(:,:,:,sample) = z0sample;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Plot several figures (this is messy and may cause bugs, comment
            % out if needed).
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if((PLOT_RESULTS>0 && (mod(epoch,PLOT_RESULTS)==0 || epoch==maxepochs)) ||...
                    (SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size)||...
                    VIDEO_FRAMES)
                
                if(SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size)
                    %% Setup a path to save to /fullsavepath/train/epoch##/layer#/
                    savepath = strcat(fullsavepath,expfile,'/epoch',num2str(epoch),'/layer',num2str(layer));
                    mkdir2(savepath);
                end
                
                if(strcmp(train_recon,'train'))
                    
                    if((SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size) ||...
                            (PLOT_RESULTS && (mod(epoch,PLOT_RESULTS)==0 || epoch==maxepochs) && ((UPDATE_FILTERS_IN_BATCH && image_num==batch_size) || UPDATE_FILTERS_IN_BATCH==0)) ||...
                            VIDEO_FRAMES)
                        % Display the filters
                        f = figure(200+model.layer); clf;
                        sdispfilts(F,model.conmats{model.layer});
                        set(f,'Name',strcat('Layer ',num2str(model.layer),' Filters'));
                        drawnow
                        if((epoch==1 && image_num == 1) || (PLOT_RESULTS==0))
                            cursize = get(f,'Position');
                            screensize = get(0,'Screensize');
                            set(f,'Position',[800,screensize(4)-cursize(4)-100,cursize(3),cursize(4)])
                        end
                        drawnow;
                        
                        if((SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size)||...
                                VIDEO_FRAMES)
                            % Make new path directory.
                            savepath = strcat(fullsavepath,expfile,'/epoch',num2str(epoch),'/layer',num2str(layer));
                            mkdir2(savepath);
                            
                            if(VIDEO_FRAMES)
                                videopath = strcat(fullsavepath,expfile,'/video_layer',num2str(layer));
                                mkdir2(videopath);
                                title(sprintf('Frame %d',frame_num));
                                if(frame_num<10)
                                    saveas(f,strcat(videopath,'/fig000',num2str(frame_num)),'png');
                                elseif(frame_num<100)
                                    saveas(f,strcat(videopath,'/fig00',num2str(frame_num)),'png');
                                elseif(frame_num<1000)
                                    saveas(f,strcat(videopath,'/fig0',num2str(frame_num)),'png');
                                elseif(frame_num<10000)
                                    saveas(f,strcat(videopath,'/fig',num2str(frame_num)),'png');
                                end
                                frame_num = frame_num+1;
                            end
                            hgsave(f,strcat(savepath,'/original_filters.fig'));
                            
                            if(PLOT_RESULTS==0)
                                close(f) % Only plotted it in order to save it.
                            end
                        end
                        %                     end
                        %
                        %                     % Display the pixel space filters for higher layers.
                        %                     if(PLOT_RESULTS || image_num==batch_size || VIDEO_FRAMES)
                        
                        
                        if(layer > 1)
                            if(TRAIN_Z0)
                                eval(strcat('z0',num2str(layer),' = z0sample;'));
                            end
                            eval(strcat('F',num2str(layer),' = F;'));
                            
                            
                            
                            % top_down_core below.
                            switch(model.norm_types{layer+1})
                                case 'Max'  % Pool the current feature maps
                                    eval(strcat('[recon_z',num2str(layer),',pooled_indices',num2str(layer),'] = max_pool(zsample,model.norm_sizes{',num2str(layer+1),'});'));
                                case 'Avg'  % Pool the current feature maps
                                    eval(strcat('[recon_z',num2str(layer),',pooled_indices',num2str(layer),'] = avg_pool(zsample,model.norm_sizes{',num2str(layer+1),'});'));
                                case 'None'
                                    eval(strcat('recon_z',num2str(layer),' = zsample;'));
                            end
                            
                            top_down_noload  % Generate the pixel space filters.
                            f = figure(250+model.layer); clf;
                            sdispims(recon_y1);
                            set(f,'Name',strcat('Layer ',num2str(model.layer),' Pixel Filters'));
                            drawnow
                            if((epoch==1 && image_num == 1) || (PLOT_RESULTS==0))
                                cursize = get(f,'Position');
                                screensize = get(0,'Screensize');
                                if(screensize(4)<1200)
                                    set(f,'Position',[1750,30,cursize(3)*2,cursize(4)*2])
                                else
                                    set(f,'Position',[1750,900,cursize(3)*2,cursize(4)*2])
                                end
                            end
                            drawnow;
                            if((SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size) ||...
                                    VIDEO_FRAMES)
                                if(VIDEO_FRAMES)
                                    videopath = strcat(fullsavepath,expfile,'/video_layer',num2str(layer));
                                    mkdir2(videopath);
                                    title(sprintf('Frame %d',pframe_num));
                                    if(frame_num<10)
                                        saveas(f,strcat(videopath,'/pfig000',num2str(pframe_num)),'png');
                                    elseif(frame_num<100)
                                        saveas(f,strcat(videopath,'/pfig00',num2str(pframe_num)),'png');
                                    elseif(frame_num<1000)
                                        saveas(f,strcat(videopath,'/pfig0',num2str(pframe_num)),'png');
                                    elseif(frame_num<10000)
                                        saveas(f,strcat(videopath,'/pfig',num2str(pframe_num)),'png');
                                    end
                                    pframe_num = pframe_num+1;
                                end
                                hgsave(f,strcat(savepath,'/pixel_filters.fig'));
                                
                                if(PLOT_RESULTS==0)
                                    close(f) % Only plotted it in order to save it.
                                end
                            end
                        end
                    end
                end
                
                %                                 if(PLOT_RESULTS || image_num==batch_size)
                %
                %                                     % Display the feature maps (very slow)
                %                                     f = figure(300+model.layer); clf;
                %
                %
                % %                                     switch(model.norm_types{layer+1})
                % %                                         case 'Max'  % Pool the current feature maps
                % %                                             sdispmaps(max_pool(z,model.norm_sizes{layer+1}));
                % %                                         case 'Avg'  % Pool the current feature maps
                % %                                             sdispmaps(avg_pool(z,model.norm_sizes{layer+1}));
                % %                                         case 'None'
                %                                             sdispmaps(z);
                % %                                     end
                %                                     set(f,'Name',strcat('Layer ',num2str(model.layer),' Feature Maps'));
                %                                     drawnow
                %                                     if((epoch==1 && image_num == 1) || (PLOT_RESULTS==0))
                %                                         cursize = get(f,'Position');
                %                                         screensize = get(0,'Screensize');
                %                                         set(f,'Position',[1920-cursize(3),screensize(4)-cursize(4)-100,cursize(3),cursize(4)])
                %                                     end
                %                                     drawnow
                %                                     if(SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size)
                %                                         hgsave(f,strcat(savepath,'/feature_maps.fig'));
                %                                         if(PLOT_RESULTS==0)
                %                                             close(f) % Only plotted it in order to save it.
                %                                         end
                %                                     end
                %                                 end
                
                
                %                                 if(PLOT_RESULTS && image_num==batch_size)
                %
                %                                     f = figure(350+model.layer); clf;
                %                                     hist(z(:),1000);
                %                                     ylim([0 5000]);
                %                                     title(strcat('Numels z: ',num2str(numel(z)),' Max bin: ',num2str(max(hist(z(:),1000))),' #==0:',num2str(length(find(abs(z(:))<1e-07))),'Mean:',num2str(mean(z(:)))));
                %                                     set(f,'Name',strcat('Layer ',num2str(model.layer),' Feature Histogram'));
                %                                     if((epoch==1 && image_num == 1) || (PLOT_RESULTS==0))
                %                                         set(f,'Position',[400,screensize(4)-400,400,300])
                %                                     end
                %                                     drawnow;
                %                                     if(SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size)
                %                                         hgsave(f,strcat(savepath,'/feature_hist.fig'));
                %                                         if(PLOT_RESULTS==0)
                %                                             close(f) % Only plotted it in order to save it.
                %                                         end
                %                                     end
                %                                 end
                
                if(((SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs)) ||...
                        (PLOT_RESULTS && (mod(epoch,PLOT_RESULTS)==0 || epoch==maxepochs))) && image_num == batch_size )
                    
                    if(TRAIN_Z0==1)
                        % Display z0 feature maps (if they are used).
                        f = figure(400+model.layer); clf;
                        if(input_maps ==1 || input_maps == 3)
                            sdispims(z0);
                        else
                            sdispmaps(z0);
                        end
                        set(f,'Name',strcat('Layer ',num2str(model.layer),' z0 Feature Maps'));
                        if((epoch==1 && image_num == 1) || (PLOT_RESULTS==0))
                            cursize = get(f,'Position');
                            screensize = get(0,'Screensize');
                            if(screensize(4)<1200)
                                set(f,'Position',[1750,30,cursize(3),cursize(4)])
                            else
                                set(f,'Position',[1750,900,cursize(3),cursize(4)])
                            end
                        end
                        drawnow;
                        if(SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs) && image_num == batch_size)
                            hgsave(f,strcat(savepath,'/z0feature_maps1.fig'));
                            if(PLOT_RESULTS==0)
                                close(f) % Only plotted it in order to save it.
                            end
                        end
                    end
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Compute Errors
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(mod(epoch,DISPLAY_ERRORS)==0 || UPDATE_INPUT)
                % The name of zsample has to be recon_z# where # is the layer.
                if(TRAIN_Z0)
                    eval(strcat('z0',num2str(layer),' = z0sample;'));
                end
                eval(strcat('F',num2str(layer),' = F;'));
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Reconstruct from the top down.
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Do the layer by layer reconstructions from the top (ending with
                % recon_z0 as the input image).
                
                % Since top_down_core epects to undo pooling at the top layer, turn
                % that one off and then set it back afterwards.
                % The top layer should be pooled here and then unpooled by
                % top_down_core below.
                switch(model.norm_types{layer+1})
                    case 'Max'  % Pool the current feature maps
                        eval(strcat('[recon_z',num2str(layer),',pooled_indices',num2str(layer),'] = max_pool(zsample,model.norm_sizes{',num2str(layer+1),'});'));
                    case 'Avg'  % Pool the current feature maps
                        eval(strcat('[recon_z',num2str(layer),',pooled_indices',num2str(layer),'] = avg_pool(zsample,model.norm_sizes{',num2str(layer+1),'});'));
                    case 'None'
                        eval(strcat('recon_z',num2str(layer),' = zsample;'));
                end
                top_down_core
                % Need to undo the pooling to get back to the image
                % perhaps.
                % Check if the input image was normalized.
                switch(model.norm_types{1})
                    case 'Max'  % Undo the Max Pooling discretely since you know the indices for the given image.
                        recon_z0 = reverse_max_pool(recon_z0,pooled_indices0(:,:,:,sample),model.norm_sizes{1},[model.orig_xdim model.orig_ydim]);
                    case 'Avg'  % Undo the Average Pooling discretely since you know the indices for the given image.
                        recon_z0 = reverse_avg_pool(recon_z0,pooled_indices0,model.norm_sizes{1},[model.orig_xdim model.orig_ydim]);
                    case 'None'
                end
                % recon_z0 is the result of the top down pass so save that in
                % the correct location to be compared to origin_image.
                recon_images(:,:,:,sample) = recon_z0;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Update y and plot and save
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if(strcmp(train_recon,'train'))
                    if(UPDATE_INPUT && layer==1) % This has to be set to updated the input images at each iteration.
                        % This uses the layer below.
                        % Linear combo of reconstruction and noisy input.
                        eval(strcat('ysample = (lambda_input*y_input(:,:,:,sample) + lambda*recon_z',num2str(model.layer-1),')/(lambda_input+lambda);'));
                        y(:,:,:,sample) = ysample;
                        
                        %                                                 % For blurred images
                        %                         blur = fspecial('gaussian',[11 11],3);
                        %                         for color=1:size(y,3)
                        %                             ysample = real(ifft2((fft2(lambda_input*y_input(:,:,color,sample)).*conj(fft2(blur,size(y,1),size(y,2)))+lambda*fft2(recon_z0(:,:,color)))./(lambda_input*conj(fft2(blur,size(y,1),size(y,2))).*fft2(blur,size(y,1),size(y,2))+lambda)));
                        %                             y(:,:,color,sample) = ysample;
                        %
                        %                         end
                        
                        
                        %
                        %                         % For cutting off half the image.
                        %                         D = cat(2,zeros(size(y,1),floor(size(y,2)/4),size(y,3),'single'),ones(size(y,1),ceil(3*size(y,2)/4),size(y,3),'single'));
                        %                         allOnes = ones(size(ysample),'single');
                        %                         eval(strcat('ysample = (lambda_input*y_input(:,:,:,sample) + lambda*recon_z',num2str(model.layer-1),')./(lambda_input*D+lambda*allOnes);'));
                        %                         y(:,:,:,sample) = ysample;
                        
                        
                        if((PLOT_RESULTS>0 && (mod(epoch,PLOT_RESULTS)==0 || epoch==maxepochs)) ||...
                                (SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs)))
                            
                            if(PLOT_RESULTS || image_num==batch_size)
                                
                                f = figure(500); clf;
                                if(size(y,3)==1 || size(y,3)==3)
                                    sdispims(y(:,:,:,sample));
                                else
                                    sdispmaps(y(:,:,:,sample));
                                end
                                set(f,'Name','Updated y Image');
                                if((epoch==1 && image_num == 1) || (PLOT_RESULTS==0))
                                    
                                    cursize = get(f,'Position');
                                    set(f,'Position',[850 30 cursize(3) cursize(4)])
                                end
                                if(SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs))
                                    savepath = strcat(fullsavepath,expfile,'/epoch',num2str(epoch),'/layer',num2str(layer));
                                    mkdir2(savepath);
                                    hgsave(f,strcat(savepath,'/updated_y_images.fig'));
                                    if(PLOT_RESULTS==0)
                                        close(f) % Only plotted it in order to save it.
                                    end
                                end
                            end
                        end
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Compute and Store Errors
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Check if the input image was normalized.
                switch(model.norm_types{layer})
                    case 'Max' % Haven't implemented max pooling reversal for the errors yet.
                        % Since in top_down_core unpooling was done, now
                        % have to pool again to compare.
                        eval(strcat('recon_error = sqrt(sum(sum(sum((max_pool(recon_z',num2str(layer-1),',model.norm_sizes{',num2str(layer),'})-y_input(:,:,:,sample)).^2))));'));
                        model.pix_noise_rec_error(epoch,sample) = recon_error;
                        
                        eval(strcat('recon_error = sqrt(sum(sum(sum((max_pool(recon_z',num2str(layer-1),',model.norm_sizes{',num2str(layer),'})-y(:,:,:,sample)).^2))));'));
                        model.pix_update_rec_error(epoch,sample) = recon_error;
                    case 'Avg' % Haven't implemented average pooling reversal for the errors yet.
                        % Since in top_down_core unpooling was done, now
                        % have to pool again to compare.
                        eval(strcat('recon_error = sqrt(sum(sum(sum((avg_pool(recon_z',num2str(layer-1),',model.norm_sizes{',num2str(layer),'})-y_input(:,:,:,sample)).^2))));'));
                        model.pix_noise_rec_error(epoch,sample) = recon_error;
                        
                        eval(strcat('recon_error = sqrt(sum(sum(sum((avg_pool(recon_z',num2str(layer-1),',model.norm_sizes{',num2str(layer),'})-y(:,:,:,sample)).^2))));'));
                        model.pix_update_rec_error(epoch,sample) = recon_error;
                    case 'None'
                        % Compute error versus the input maps (recon_z# where # is the layer below.
                        eval(strcat('recon_error = sqrt(sum(sum(sum((recon_z',num2str(layer-1),'-y_input(:,:,:,sample)).^2))));'));
                        model.pix_noise_rec_error(epoch,sample) = recon_error;
                        
                        % Layer 1's error versus the updated pixel space images.
                        eval(strcat('recon_error = sqrt(sum(sum(sum((recon_z',num2str(layer-1),'-y(:,:,:,sample)).^2))));'));
                        %                 fprintf(' :1: Updated y Total error: %4.1f, Recon error (all planes): %4.1f, Reg error: %4.1f Inp error: %4.1f\n',recon_error+reg_error+upd_rec_error,recon_error,reg_error,upd_rec_error);
                        model.pix_update_rec_error(epoch,sample) = recon_error;
                end
                % Updated y versus the noisy input reconstruction error
                % (same for Layers 1 and 2).
                upd_rec_error = sqrt(sum(sum(sum((y_input(:,:,:,sample)-y(:,:,:,sample)).^2))));
                model.update_noise_rec_error(epoch,sample) = upd_rec_error;
                
                % Compute regularization error.
                reg_error = sum(abs(zsample(:))); % Compare with L1 norm.
                %                 fprintf(' :1: Pix Noise Total error: %4.1f, Recon error (all planes): %4.1f, Reg error: %4.1f Inp error: %4.1f\n',recon_error+reg_error+upd_rec_error,recon_error,reg_error,upd_rec_error);
                model.reg_error(epoch,sample) = reg_error;
                
                % Layer 1's error versus the clean pixel space images.
                recon_error = sqrt(sum(sum(sum((recon_images(:,:,:,sample)-original_images(:,:,:,sample)).^2))));
                SNR_error = compute_snr(original_images(:,:,:,sample),recon_images(:,:,:,sample));
                fprintf(' ::: Pix Clean Total error: %4.1f, Recon error: %4.1f, Reg error: %4.1f SNR: %4.4f / Noisy: %4.4f = %3.1f x\n',...
                    recon_error+reg_error,recon_error,reg_error,...
                    SNR_error,noisy_SNR(sample),SNR_error/noisy_SNR(sample));
                model.pix_clean_rec_error(epoch,sample) = recon_error;
                model.pix_clean_SNR_error(epoch,sample) = SNR_error;
                
                % Compute Beta reconstruction error.
                recon_error = sqrt(sum(sum(sum((w(:,:,:,sample)-z(:,:,:,sample)).^2))));
                model.beta_rec_error(epoch,sample) = recon_error;
                
                % Compute the sum of each term (without coefficients).
                model.unscaled_total_energy(epoch,sample) = model.pix_update_rec_error(epoch,sample)+model.reg_error(epoch,sample)+upd_rec_error+model.beta_rec_error(epoch,sample);
                fprintf(' ::: Energy Function Total: %4.1f Lay Reg: %4.1f Update Error: %4.1f Lay Rec Upd: %4.1f Lay Rec Beta: %4.1f\n',...
                    model.unscaled_total_energy(epoch,sample),...
                    model.reg_error(epoch,sample),upd_rec_error,model.pix_update_rec_error(epoch,sample),model.beta_rec_error(epoch,sample))
                
                % Compute the sum of each term (with coefficients)
                model.scaled_total_energy(epoch,sample) = lambda/2*model.pix_update_rec_error(epoch,sample)+kappa(layer)*model.reg_error(epoch,sample)+lambda_input/2*upd_rec_error+(beta/2)/kappa(layer)*model.beta_rec_error(epoch,sample);
                fprintf(' ::: Scaled Energy F Total: %4.1f Lay Reg: %4.1f Update Error: %4.1f Lay Rec Upd: %4.1f Lay Rec Beta: %4.1f\n',...
                    model.scaled_total_energy(epoch,sample),...
                    kappa(layer)*model.reg_error(epoch,sample),lambda_input*upd_rec_error,lambda/2*model.pix_update_rec_error(epoch,sample),(beta/2)*model.beta_rec_error(epoch,sample))
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Display reconstructed Image planes.
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if((PLOT_RESULTS>0 && (mod(epoch,PLOT_RESULTS)==0 || epoch==maxepochs)) ||...
                        (SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs)))
                    
                    if(PLOT_RESULTS || image_num==batch_size)
                        
                        f = figure(500+model.layer); clf;
                        sdispims(recon_images);
                        set(f,'Name',strcat('Layer ',num2str(model.layer),' Reconstructions'));
                        drawnow
                        if((epoch==1 && image_num == 1) || (PLOT_RESULTS==0))
                            cursize = get(f,'Position');
                            screensize = get(0,'Screensize');
                            if(screensize(4)<1200)
                                set(f,'Position',[850+cursize(3)*model.layer,30,cursize(3),cursize(4)])
                            else
                                set(f,'Position',[850+cursize(3)*model.layer,900,cursize(3),cursize(4)])
                            end
                        end
                        if(SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs))
                            %% Setup a path to save to /fullsavepath/train/epoch##/layer#/
                            savepath = strcat(fullsavepath,expfile,'/epoch',num2str(epoch),'/layer',num2str(layer));
                            mkdir2(savepath);
                            hgsave(f,strcat(savepath,'/recon_images.fig'));
                            if(PLOT_RESULTS==0)
                                close(f) % Only plotted it in order to save it.
                            end
                        end
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end % This is the end of the batch, so update filters after this.
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Update Filters in a batch.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(strcmp(train_recon,'train'))
            if UPDATE_FILTERS_IN_BATCH
                fprintf('Updating Filters in Batch\n');
                
                if(~USE_IPP)
                    F = fast_batch_learn_filters(min_iterations,z(:,:,:,batch_indices),y(:,:,:,batch_indices),...
                        F,z0batch,z0_filter_size,lambda,connectivity_matrix,TRAIN_Z0);
                else
                    if(TRAIN_Z0)
                        z0batch = z0(:,:,:,batch_indices);
                    else
                        z0batch = [];
                    end
                    F = ipp_batch_learn_filters(min_iterations,z(:,:,:,batch_indices),y(:,:,:,batch_indices),...
                        F,z0batch,z0_filter_size,lambda,connectivity_matrix,TRAIN_Z0,COMP_THREADS);
                end
                % Normalize the columns
                F = plane_normalize(F);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Compute the average errors over the batch
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % These will be saved to the model struct for later viewing.
        if(mod(epoch,DISPLAY_ERRORS)==0)
            
            fprintf('\nLayer: %1d, Epoch: %2d, Batch:    \n',layer,epoch);
            
            fprintf(' ::: Pix Clean Total error: %4.1f, Recon error: %4.1f, Reg error: %4.1f SNR: %4.4f / Noisy: %4.4f\n',...
                mean(model.pix_clean_rec_error(epoch,:),2)+mean(model.reg_error(epoch,:),2),...
                mean(model.pix_clean_rec_error(epoch,:),2),mean(model.reg_error(epoch,:),2),...
                mean(model.pix_clean_SNR_error(epoch,:),2),mean(noisy_SNR));
            
            fprintf(' ::: Energy Function Total: %4.1f Lay Reg: %4.1f Update Error: %4.1f Lay Rec Upd: %4.1f Lay Rec Beta: %4.1f\n',...
                mean(model.pix_update_rec_error(epoch,:),2)+mean(model.reg_error(epoch,:),2)+mean(model.update_noise_rec_error(epoch,:),2)+mean(model.beta_rec_error(epoch,:),2),...
                mean(model.reg_error(epoch,:),2),mean(model.update_noise_rec_error(epoch,:),2),mean(model.pix_update_rec_error(epoch,:),2),mean(model.beta_rec_error(epoch,:),2))
            
            %                 fprintf('                               ');
            fprintf(' ::: Scaled Energy F Total: %4.1f Lay Reg: %4.1f Update Error: %4.1f Lay Rec Upd: %4.1f Lay Rec Beta: %4.1f\n',...
                lambda/2*mean(model.pix_update_rec_error(epoch,:),2)+kappa(layer)*mean(model.reg_error(epoch,:),2)+lambda_input*mean(model.update_noise_rec_error(epoch,:),2)+...
                (beta/2)*mean(model.beta_rec_error(epoch,:),2),...
                kappa(layer)*mean(model.reg_error(epoch,:),2),lambda_input/2*mean(model.update_noise_rec_error(epoch,:),2),lambda/2*mean(model.pix_update_rec_error(epoch,:),2),...
                (beta/2)*mean(model.beta_rec_error(epoch,:),2))
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Save Results: Can be used to infer the z's for a given image.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (SAVE_RESULTS>0 && (mod(epoch,SAVE_RESULTS)==0 || epoch==maxepochs))
        % Save the filter matrix and model if it exists.
        if(TRAIN_Z0==1)
            save(strcat(fullsavepath,'epoch',int2str(epoch),'_layer',int2str(layer),'_z0.mat'),...
                'F','z','z0','y','model','-v7.3');
        else
            z0 = 0;
            save(strcat(fullsavepath,'epoch',int2str(epoch),'_layer',int2str(layer),'.mat'),...
                'F','z','z0','y','model','-v7.3');
        end
        
        % Also save the pooled layers at given epochs so you can stop
        % training at any time.
        switch(model.norm_types{layer+1})
            case 'Max'  % Pool and unpool the current feature maps
                [pooled_maps,pooled_indices] = max_pool(z,model.norm_sizes{layer+1}); %#ok<NASGU>
                % Save the 4 results of the max_pool function.
                save(strcat(fullsavepath,'layer',num2str(layer),'_pooling.mat'),...
                    'pooled_maps','pooled_indices','-v7.3');
            case 'Avg'  % Pool and unpool the current feature maps
                [pooled_maps,pooled_indices] = avg_pool(z,model.norm_sizes{layer+1}); %#ok<NASGU>
                % Save the 4 results of the max_pool function.
                save(strcat(fullsavepath,'layer',num2str(layer),'_pooling.mat'),...
                    'pooled_maps','pooled_indices','-v7.3');
            case 'None'
        end
        
        
        
    end
    t=toc;
    fprintf('Time For One Epoch: %f\n',t)
    fprintf('----------------------------------------------------------------------------------------------\n');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


% Plots various surfaces over epochs and other things.
%plot_surfaces

end
