% === IMAGE SEGMENTATION USING GMM =====
% [I_label, I_post] = fn_imgSegmentationGMM(imagename);
% Take an image name imagename as the input, the function will output the
% resulting segmentation I_label and its maximum posterior probability
% I_post. The default values of cluster numbers and weights will be 2 and [3 3 3
% 1 1] respectively
% [I_label, I_post] = fn_imgSegmentationGMM(imagename, C, W);
% [I_label, I_post] = fn_imgSegmentationGMM(imagename, C);
% 
% ----- Example -----------
% imagename = '108073';
% C = 3; W = [5 5 5 1 1];
% [I_label, I_post] = fn_imgSegmentationGMM(imagename, C, W);
% ========================================


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

function [I_label, I_post] = fn_imgSegmentationGMM(imagename, imageext, C, W)

if nargin < 4 W = [3 3 3 1 1]; end % W := [L a b x y] scale of each dimension
if nargin < 3 C = 3; end % C: number of class

[s,mess,messid] = mkdir('./',imagename); % make a new folder
disp(mess);

% import an image
img_RGB = imread([imagename '/' imagename,imageext]);
Ncol = size(img_RGB,2);
Nrow = size(img_RGB,1);

% convert to Lab color space
cform = makecform('srgb2lab');
img_LAB = applycform(img_RGB,cform);

% normalize the color pixel
I = img_LAB;
I = double(I)/255;
% figure; imagesc(I); title('image plotted in Lab color space');
Y = reshape(I(:),Ncol*Nrow,[]); % align the image pixels by NxD

% create pixel location
[xI,yI] = meshgrid(1:Ncol,1:Nrow); 
% figure; imagesc(xI); daspect([1 1 1]); % 4test
% figure; imagesc(yI); daspect([1 1 1]); % 4test

% normalize the location
xI = xI(:); yI = yI(:);
xI = xI/max(xI,[],1);
yI = yI/max(yI,[],1);

% === Compose feature vectors ====
Y = [Y xI yI];
Y = Y(:,W~=0); % pick non-zero weights
W = W(W~=0); % pick non-zero weights

% ==== standardize the data ======
Y = Y - repmat( mean(Y,1), length(Y), 1);
Y = Y./repmat(std(Y,1), length(Y), 1); % standardize the normal data

% --- scale each dimension ----
W = W/sum(W,2)*10; % scale by 10 to prevent underflow of any dimension
Y = Y.*repmat(W,size(Y,1),1);
% figure; imagesc(Y); % 4test

%% ---- fitting GMM -----
tic;
D = size(Y,2); % dimensionality
options = statset('Display','final','MaxIter',500);
gmm_obj = gmdistribution.fit(Y,C,'Regularize',1e-4,'Replicates',1,'Options',options);
disp(['GMM spend ',num2str(toc),' sec']);
% ------------------------

%  ------ classification using MAP ---------------------------------------
gmm_posterior = posterior(gmm_obj,Y);
[max_postr, class_result] = max(gmm_posterior,[],2);
% --------------------------------------------------------------------

% ---- plot the label ----
I_label = reshape(class_result,Nrow,Ncol,[]);
I_post = reshape(max_postr,Nrow,Ncol,[]);

I_tmp = double(label2rgb(I_label))/255; % convert label to RGB 
I_result = I_tmp;
% ---- incorporate the posterior as brightness of the result -------- 
I_result(:,:,1) = I_tmp(:,:,1).*I_post;
I_result(:,:,2) = I_tmp(:,:,2).*I_post;
I_result(:,:,3) = I_tmp(:,:,3).*I_post;
figure; imagesc(I_result); title(['resulting segmentation using GMM with C=',num2str(C),'W=',num2str(W)]); axis equal off tight; colorbar;
print('-djpeg','-r200',[imagename,'_segm_gmm_class',num2str(C),'_LAB_XY_normalized_weighted.jpg']);
movefile(['./',imagename,'_segm_gmm_class',num2str(C),'_LAB_XY_normalized_weighted.jpg'],['./',imagename]);

% ----- overlay the label on the original image ----------
I_gray = double(rgb2gray(img_RGB))/255; 
I_gray = I_gray+1; I_gray = I_gray/max( max(I_gray,[],1),[],2); % fade the gray scale a bit for proper display
% figure; imagesc(I_gray); axis equal off tight; colormap gray; caxis([0 1]);
I_overlay = I_tmp;
% ---- overlay the label image on the gray scale image -------- 
I_overlay(:,:,1) = I_tmp(:,:,1).*I_gray;
I_overlay(:,:,2) = I_tmp(:,:,2).*I_gray;
I_overlay(:,:,3) = I_tmp(:,:,3).*I_gray;
figure; imagesc(I_overlay); title(['resulting segmentation using GMM with C=',num2str(C),'W=',num2str(W)]); axis equal off tight; colorbar;
print('-djpeg','-r200',[imagename,'_overlay_segm_gmm_class',num2str(C),'_LAB_XY_normalized_weighted.jpg']);
movefile(['./',imagename,'_overlay_segm_gmm_class',num2str(C),'_LAB_XY_normalized_weighted.jpg'],['./',imagename]);

% ===== plot 2 images in the same Windows ======
% figure; 
% subplot(2,1,1); imagesc(I_overlay); title(['resulting segmentation using GMM with C=',num2str(C),'W=',num2str(W)]); axis equal off tight; colorbar;
% subplot(2,1,2); imagesc(I_result); title(['resulting segmentation using
% GMM with C=',num2str(C),'W=',num2str(W)]); axis equal off tight; colorbar;

disp(['The results (tree structure) are saved in the folder /',imagename]);