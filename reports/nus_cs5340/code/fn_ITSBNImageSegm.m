function [] = ITSBNImageSegm(filename_input,...
                                          imagename, imageext,...
                                          C,...
                                          iteration_max,...
                                          exp_number,...
                                          cpt_spread_factor,...
                                          added_diag_cpt,...
                                          epsilon_F)
% ITSBN on 2D image of any size
% This algorithm is superpixel-based. Inputs at leaf nodes are feature
% vectors extracted from the corresponding superpixels. The irregular tree
% structure is created by the hierarchical superpixel program, e.g.
% Quickshift, turbopixel, UCM. 


% #########################################################################
% ############ Tree Structure Image Segmentation ########################
% ========================================================================
% Mohammad Akbari, NGS,
% Shahab Ensafi, Soc,
% Fu Jie, NGS,
% National University of Singapore
% {Akbari, shahab.ensafi, jie.fu} @nus.edu.sg
% Thanks from Li Cheng, Kittipat Kampa and Matthew Zeiler
% ########################################################################

addpath('./Gaussian2Dplot'); % add 2D Gaussian plot 

%% ######################################################################## 
% ############################ DEFAULT PARAMETERS #########################
% =========================================================================
% This parameters are indeed for users to determine, especially the
% parameters C, iteration_max and experiment number. However, if the user
% chooses to discard the parameters, the program will put the default
% parameters as follows:
% #########################################################################
if nargin < 9 epsilon_F = 1e-4; end
if nargin < 8 added_diag_cpt = 0.001; end
if nargin < 7 cpt_spread_factor = 0.3; end % 0 < cpt_spread_factor < 1, the bigger, the more uniform initial CPT is
if nargin < 6 exp_number = 666; end
if nargin < 5 iteration_max = 10; end
if nargin < 4 C = 0; end % number of classes/mixtures in GMM
    
% #########################################################################
% ######################### make a storage folder #########################
% =========================================================================
% # All the results, .mat file and all figures from a test image will be stored in this folder
% ####################################
[s,mess,messid] = mkdir('./',imagename); % make a new folder
addpath(['./',imagename]);
disp(mess);


% #########################################################################
% ################## some auxilialry constant for stability ###############
% =========================================================================
% # The parameters here are for stability of the algorithm. Only advanced
% user are reccomended to change them. 
% #########################################################################
added_renormalize_cpt_factor = 1e-6;
added_message_stabilizer = 1e-4;
Sigma_c_loading_factor = 1e-3;


% #########################################################################
% ######################## List of figure ID ############################
% =========================================================================
% # Here is where we can define the ID of each plotted figure
% ########################################################################
fig_initial_CPT = 101;
fig_CPT_each_level = 102;
fig_posterior_each_level = 103;
fig_GMM_feature_space = 104;
fig_final_GMM_feature_space = 105;
fig_cost_function = 106;
fig_GMM_segmentation = 107;
fig_TSBN_segmentation = 108;
% ============== END OF USER DEFINE PARAMETERS ==============



%% ########################################################################
% ######################## Prepare data for segmentation #################
% =========================================================================
% We load tree structure from the previous process, together with feature
% vectors extracted from each superpixel
% ########################################################################
load(filename_input); % load tree structure package
load([imagename,'_feature']); % load image feature

% read the image in
img_RGB = imread([imagename,imageext]);


% ########################################################################
% ########### Pick FEATURES to use in the segmentation ###################
% =========================================================================
% The features are listed as following:
% averaged RGB: 1-3
% averaged CIE Lab: 4-6
% averaged box DCT: 7-16
% averaged centerbox DCT: 17-26
% averaged 3-level Complex wavelet transform: 27 - 35
% ########################################################################
% It is recommended to use RGB, CIE Lab and 3-level wavelet
Y = Y(:,[1:6,27:35]); % recommended setting


% ########################################################################
% ######################## Standardize the data ########################
% =========================================================================
% We process the feature vector such that, for each dimension, the mean is
% set to 0 and the std is at 1. This way we can take any kind of feature, and we
% don't have to concern about the dynamic range of each dimension at all.
% ########################################################################
Y = Y - repmat( mean(Y,1), length(Y), 1);
Y = Y./repmat(std(Y,1), length(Y), 1); % standardize the normal data
% Y = Y/255; % standardize the pixel data
figure(fig_GMM_feature_space); hold on; plot(Y(:,1),Y(:,2),'.-','Color',0.7*[1 1 1]); daspect([1 1 1]);


% ########################################################################
% ####### Prepare the number of levels and points in the dataset #########
% ########################################################################
N = size(Y,1); % number of the points in the dataset
L = n_level+2; % #level from H(0) - H{L-1}. 
% In this version 2.1.0, nodes from H(0) are not included in the
% network structure Z, but H(0) still remains in a lot of important
% parameters, for instance, H{}, CPT{}. However, we don't input any
% parameter from H(0) when building bnet1 though.
K = size(Z,1); % number of nodes in the structure, not including the phantom node





%% ########################################################################
%  ######################---- FITTING GMM -----############################
% =========================================================================
% Each feature vector extracted from finest-scale superpixel (level H0) is
% clustered using Gaussian mixture model (GMM). The number of the class is
% determined by user, otherwise determined by BIC. The mean and covariance
% of each cluster is stored and will be used in the next process when
% calculating ITSBN. More specifically, We use the resulting mean and covariance here as an
% initial point of ITSBN.
% ########################################################################
tic;
D = size(Y,2); % dimensionality
options = statset('Display','final','MaxIter',500);

if C == 0 % We will use BIC to decide the number of clutster
    BIC_winner = -1e+9;
    for C = 3:5
        gmm_obj_tmp = gmdistribution.fit(Y,C,'Regularize',1e-4,'Replicates',20,'Options',options);
        if gmm_obj_tmp.BIC > BIC_winner
            gmm_obj = gmm_obj_tmp;
            BIC_winner = gmm_obj_tmp.BIC;
        end
    end
else % if C is determined by the user
    gmm_obj = gmdistribution.fit(Y,C,'Regularize',1e-4,'Replicates',20,'Options',options);
end

mu_c = gmm_obj.mu; % zeros(C,D);
Sigma_c = gmm_obj.Sigma; % zeros(D,D,C);
Lambda_c = zeros(D,D,C);
for c = 1:C
    Lambda_c(:,:,c) = Sigma_c(:,:,c)^(-1);
end
disp(['GMM spend ',num2str(toc),' sec']);
% ------------------------


%% ########################################################################
% ##################### INITIAL STRUCTURE ######################
% =========================================================================
% Normally, in this version of code, we'll get the initial structure from another code function fn_makeTreeFromLabel.m
% However, at this point, you can also change the structure manually, For
% example,
% -- YOU CAN CHANGE THE STRUCTURE ---
% Z(9,:) = 0; Z(9,5) = 1;
% Z(14,:) = 0; Z(14,6) = 1;
% -----------------------------------
% The structure Z is (child x parent), however, the format we are going to
% input to the BNET is dag (parent x child).
% ########################################################################

dag = Z(:,1:(end-1))';

% --- visualize the tree structure ---
if N < 100
    figure(665); title('initial structure'); draw_graph(dag);
%     print('-depsc','-r200',['initial_structure_',num2str(exp_number),'.eps']);
    print('-djpeg','-r100',['initial_structure_',num2str(exp_number),'.jpg']);
%     h = gcf; saveas(h,['initial_structure_',num2str(exp_number),'.fig'])
movefile(['./initial_structure_',num2str(exp_number),'.jpg'], ['./',imagename]);
end
% ------------------------------------




%% ########################################################################
%  ###################---- MAKE CPT for ITSBN -----########################
% =========================================================================
% In this work, we assume that the CPT is shared within the same level for algorithmic robustness, 
% however, CPTs from different levels can be, and in most cases are,
% different. The CPT cellarray is in the following format:
% - CPT is a (Lx2) cell array. L is the number of levels in the ITSBN.
% Remember that we have level H0 - H(L-1), totally L levels. The number of 
% column, 2, is
% - col#1: store the number indicating the level l. You may figure out that CPT{l+1,1} = l; 
% which makes the notion of col#1 seem redundant, but we still store l there for our conveniene.
% - col#2: store the CPT matrix of that level. Note that there is no
% restrictions of the dimension of the CPT, so it means we can make any arbitrary dimension as we wish.
% For example, we can CPT for level l and l+1 to be rxs and sxt
% respectively.
% - Also note that, we have "+1" in the subscript of CPT...what does it means?
% As you may know that our paper run the level from l=0 to L-1, but matlab
% subscript starts from 1. Therefore, we map l=0 to 1, hence we write
% CPT{l+1} for the level Hl.
% - Spread factor for each CPT. This is done by adding a w-weighted random matrix
% whose entry is between 0-w to the original CPT. This way the CPT is
% spreaded away from diagonal, which means uncertainty in the label field.
% The weight w is the variable cpt_spread_factor in the code. 
% cpt_spread_factor = 0.9 means very unspecific CPT, while
% cpt_spread_factor = 0.01 means very specific CPT. 
% ########################################################################

CPT = cell(L,2); % initialize the CPT cell array

if C == 2
    CPT{0+1,1} = 0; % in 2.1.0 we don't have H(0)
    CPT{0+1,2} = eye(C);
    for l = 1:(L-2)
        CPT{l+1,1} = l;
        cpt = eye(C)+ cpt_spread_factor *rand(C,C); % --- spread CPT ----
        cpt = cpt./repmat(sum(cpt,1),C,1); % normalizing each column
        CPT{l+1,2} = cpt;
    end
    CPT{L-1+1,1} = L-1;
    CPT{L-1+1,2} = ones(1, C)/C;
else
    % ------- CPT 3 class ---------
    if C == 3
        % --- spread CPT ----
        CPT{0+1,1} = 0; % in 2.1.0 we don't have H(0)
        CPT{0+1,2} = eye(C);
        for l = 1:(L-2)
            CPT{l+1,1} = l;
            cpt = eye(C)+ cpt_spread_factor *rand(C,C);
            cpt = cpt./repmat(sum(cpt,1),C,1);
            CPT{l+1,2} = cpt;
        end
        CPT{L-1+1,1} = L-1;
        CPT{L-1+1,2} = ones(1, C)/C;
    else % C is not either 2 or 3
        % -------- CPT of other C-class ---------
        CPT{0+1,1} = 0; % in 2.1.0 we don't have H(0)
        CPT{0+1,2} = eye(C);
        for l = 1:(L-2)
            CPT{l+1,1} = l;
            cpt = eye(C)+ cpt_spread_factor *rand(C,C); % 0.9 --> very unspecific CPT
            cpt = cpt./repmat(sum(cpt,1),C,1);
            CPT{l+1,2} = cpt;
        end
        CPT{L-1+1,1} = L-1;
        CPT{L-1+1,2} = ones(1, C)/C;
    end
end

% =======================plot the initial CPT =============================
figure(fig_initial_CPT); title('initial CPT')
tmp_cnt = 1;
for l = (L-1):-1:1
    subplot(L-1, 1, tmp_cnt); imagesc(CPT{l+1,2} );
    colormap('gray'); caxis([0 1]); colorbar;
    set(gca, 'XTick', [1:C],'XTickLabel',1:C);
    ylabel(['H^',num2str(l)]);
    daspect([1 1 1]);
    tmp_cnt = tmp_cnt + 1;
end



%% #######################################################################
% ########################################################################
% ########################################################################
% ############################ MAIN PROGRAM ##############################
% ########################################################################
% ########################################################################
% ########################################################################
% ========================================================================
% Here we use Bayes-Net Toolbox (BNT) to infer the hidden nodes. However,
% BNET does not fully support our parameter learning for ITSBN, so we use BNT 
% just to calculate the expectation at each node, then use those expectations
% to update parameters using EM-ITSBN outside the BNT.
% ########################################################################
tic;

% ----- make a Bayesian network from structure and parameters
X_index = 1:K; % version 2.1.0 indices of hidden nodes
discrete_nodes = [ X_index ]; % version 2.1.0
node_sizes = C*ones(1,K); % in this case we fix the node size to C for every node in the structure
bnet1 = mk_bnet(dag, node_sizes, 'discrete', discrete_nodes);
disp(['mk_bnet ',num2str(toc),' sec']);


F_old = -1e+9; % The initial value of cost function is assigned with a huge negative value
F = F_old + 1; % init
iteration = 1;
figure(fig_CPT_each_level); figure(fig_posterior_each_level);


while F - F_old > epsilon_F && iteration <= iteration_max
    
    
    disp(['iteration: ',num2str(iteration)]);
    close(fig_CPT_each_level); close(fig_posterior_each_level);
    
    
    
    % =================== classification for GMM =========================
    % In this section, we unsupervisedly segment each superpixel
    % independently using GMM. 
    % Later on, we will will use the mean and covariance in this process as
    % the intial value for ITSBN.
    % ====================================================================
    script_GMMCalculatePosterior;
    
    
    
    % ===================== CONSTRUCT ITSBN ==============================
    % In this section, we will construct a tree-structure bayesian networks
    % using BNT. The summary is following:
    % 1) We prepare and update the CPT of the ITSBN
    % 2) Prepare soft and hard evidence for the ITSBN
    % 3) Make an inference engine out of those prepared information
    % The inference engine will be used later when needing to calculate the
    % posterior marginal or the expectation of each node in the network
    % ================== put/update CPTs to the network ==================
    % # add CPT to nodes in level H(1) to H(L-2)
    % ====================================================================
    script_constructITSBN;
    
    
    
    % =============== calculating the cost function F() =================
    % This section, we calculate the cost function F which is the
    % expectation of the complete log-likelihood. The value of F is used to
    % indicate when to stop the algorithm.
    % There are 2 terms involving in the cost function:
    % 1) term#1: log-likelihood term which is modeled using the multivariate
    % Gaussian. 
    % 2) term#2: The tree-structured prior term which involves image-class
    % label.
    % The 2 terms will be combined into the cost function F at the end
    % ===================================================================
    script_calculateCostFuncF;
    
    
    %=================== PARAMETER LEARNING for ITSBN =====================
    % In this section, we learn the parameters mu_c, Sigma_c and phi_lvu
    % in order to update the ITSBN. The update equations are not very stable,
    % so I add some regularizers to them. 
    % =====================================================================
    script_ITSBNParameterLearning;
    
    
    % ===== Plot CPT, distribution at each node, segmentation results =====
    % In this version we moved the plot routine outside as a script so that
    % the main code is readable
    % =====================================================================
    script_plottingResult;

    
end

% ########################################################################
% ########################## SAVE THE RESULTS ############################
% The segmentation result is saved to a cell array called "segs" which will
% be used in evaluation process
% =========================================================================
% ########################################################################

[seg_result] = fn_displaySegmentationResult(class_result, Y_index', node_label_hier{1,3});
segs = cell(1,1);
segs{1,1} = seg_result;
save(imagename,'segs');
movefile(['./',imagename,'.mat'], ['/home/student1/MATLABcodes/BSR/bench/data/segs_bot_ITSBN']);
% =========================================================================
    
    
% ########################################################################
% ########################## PLOT FINAL RESULT ############################
% Here, we will plot the final clusters in the feature space (2D though),
% and also the cost function F too.
% =========================================================================
% ########################################################################

% ================ plot the FINAL classification result ==================
figure(fig_final_GMM_feature_space); hold on; daspect([1 1 1]); title('final classification result');
plot(Y(:,1),Y(:,2),'.-','Color',0.7*[1 1 1]); daspect([1 1 1]);
for c = 1:C
    class_index = class_result == c;
    d_plot = Y(class_index,:);
    color_index = mod(c,length(color_cls))+1;
    plot(d_plot(:,1),d_plot(:,2),'o','Color',color_cls(color_index));
    plot(mu_c(c,1),mu_c(c,2),'x','Color',color_cls(color_index));
    h = plotGaussianContour2D(mu_c(c,:)',Sigma_c(:,:,c),1,color_cls(color_index),300);
end
% print('-depsc','-r200',['final_classification result_',num2str(exp_number),'.eps']);
print('-djpeg','-r100',['final_classification result_',num2str(exp_number),'.jpg']);
movefile(['./final_classification result_',num2str(exp_number),'.jpg'], ['./',imagename]);


% =================== print the cost function curve =======================
figure(fig_cost_function);
%     print('-depsc','-r200',['cost_function_',num2str(exp_number),'.eps']);
    print('-djpeg','-r100',['cost_function_',num2str(exp_number),'.jpg']);
%     h = gcf; saveas(h,['cost_function_',num2str(exp_number),'.fig'])
movefile(['./cost_function_',num2str(exp_number),'.jpg'], ['./',imagename]);
disp(['The results (tree structure) are saved in the folder /',imagename]);

% --- remove the path
rmpath(['./',imagename]);