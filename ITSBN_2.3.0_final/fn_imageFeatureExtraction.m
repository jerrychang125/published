function [] = fn_imageFeatureExtraction(imagename, imageext)


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
% note for using feature learning the corresponding package should be
% installed

% ========== for test ===============
% close all
% clear all
% clc
% ===== user input =====
% imagename = '302022'; % flowershirt
% imagename = '326085'; % wildcat
% imagename = '326025'; % wildcat2
% ======================
% ===================================


[s,mess,messid] = mkdir(['./',imagename]);
disp(mess);
addpath(['./',imagename]);
% load([imagename,'_img_label_hier']);
load([imagename,'_package']);

% ################################################
% ##### Define the feature vector ################
% ################################################
avg_RGB = [1:3]; D_RGB = 3;
avg_LAB = [4:6]; D_LAB = 3;
box_DCT = [7:16]; D_DCT = 10;
centerbox_DCT = [17:26]; D_centerDCT = 10;
avg_DWT = [27:35]; D_avgDWT = 9;
feature_learning =[36:80]; D_feature_learning = 45;

D_feature = D_RGB +D_LAB +D_DCT +D_centerDCT +D_avgDWT +D_feature_learning;

% ########################################################################
% ########################################################################
% ########################################################################
% ######################## feature extraction ############################
% ########################################################################
% ########################################################################
% ########################################################################
% import an image
img_RGB = imread([imagename,imageext]);
Ncol = size(img_RGB,2);
Nrow = size(img_RGB,1);

% Feature#1: RGB ------ normalize the color pixel
I = img_RGB;
I = double(I)/255;
I_RGB = reshape(I(:),Ncol*Nrow,[]); % align the image pixels by Nx3, N: # of pixels in the image

% Feature#2: LAB ------ normalize the color pixel
cform = makecform('srgb2lab');
img_LAB = applycform(img_RGB,cform);
% normalize the color pixel
I = img_LAB;
I = double(I)/255;
I_LAB = reshape(I(:),Ncol*Nrow,[]); % align the image pixels by NxD, N: # of pixels in the image

% Feature#3 Gray ---
I_GRAY = rgb2gray(img_RGB);
% I = double(I)/255;
% I_GRAY = reshape(I(:),Ncol*Nrow,[]); % align the image pixels by NxD, N: # of pixels in the image
% figure; imagesc(I_GRAY); colormap('gray');

% Feature#4 DWT
% --- wavelet level1 ----
[cA1,cH1,cV1,cD1] = dwt2(I_GRAY,'db1');
% cA=cA1; cH=cH1; cV=cV1; cD=cD1;
% figure; title('DWT level1');
% subplot(2,2,1); imagesc(cA); colormap('gray');
% subplot(2,2,2); imagesc(cH); colormap('gray');
% subplot(2,2,3); imagesc(cV); colormap('gray');
% subplot(2,2,4); imagesc(cD); colormap('gray');

% --- wavelet level2 ----
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db1');
% cA=cA2; cH=cH2; cV=cV2; cD=cD2;
% figure; title('DWT level2');
% subplot(2,2,1); imagesc(cA); colormap('gray');
% subplot(2,2,2); imagesc(cH); colormap('gray');
% subplot(2,2,3); imagesc(cV); colormap('gray');
% subplot(2,2,4); imagesc(cD); colormap('gray');

% --- wavelet level3 ----
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db1');
% cA=cA3; cH=cH3; cV=cV3; cD=cD3;
% figure; title('DWT level3');
% subplot(2,2,1); imagesc(cA); colormap('gray');
% subplot(2,2,2); imagesc(cH); colormap('gray');
% subplot(2,2,3); imagesc(cV); colormap('gray');
% subplot(2,2,4); imagesc(cD); colormap('gray');

% I_feature = [I_RGB I_LAB]; % align the image pixels by NxD, N: # of pixels in the image

% % --- test ----
% I_show = reshape(I_GRAY,Nrow,Ncol,[]);
% figure; imagesc(I_show); colormap('gray');
% % -------------

% ##### average the feature within a superpixel ########
% Y: (# superpixel in H(0)) X dimension of feature vector
% Y_index: (# superpixel in H(0)) X 1 vector store the node index of H(0)
% So, we can tell precisely about which superpixel would have what feature
% vector from (Y_index, Y)
evidence_label = node_label_hier{1,3}; % labels at the H(1)
num_evidence = length(unique(evidence_label)); % number of superpixel in H(0)

Y = zeros(num_evidence, D_feature); % feature storage
Y_index = zeros(num_evidence,1);

% ===== extract and organize the feature ======
I_soft_evidence = unique(evidence_label)'; % List of soft evidence nodes
tmp_cnt = 1;
for i = I_soft_evidence
    % get the region indices of superpixel i
    region_index = (evidence_label==i);
    % get the min, max of x-y coordinate of the region
    [sub_row, sub_col] = ind2sub([Nrow, Ncol], find((region_index==1)));
    % get the bounding box
    row_min = min(sub_row,[],1); row_max = max(sub_row,[],1);
    col_min = min(sub_col,[],1); col_max = max(sub_col,[],1);
    % Get the centroid of the superpixel
    row_center = floor(mean(sub_row,1));
    col_center = floor(mean(sub_col,1));
    
    % ####### Feature #######  average RGB #######
    Y(tmp_cnt,avg_RGB) = mean( I_RGB(region_index(:),:) ,1);

    % average L*a*b
    Y(tmp_cnt,avg_LAB) = mean( I_LAB(region_index(:),:) ,1);
    
%     % ####### Feature ####### first 10 coef of DCT of bounding box ----
%     box_GRAY = I_GRAY(row_min:row_max, col_min:col_max); % figure; imagesc(box_GRAY);
%     % apply DCT to the bounding box
%     dct_coef = dct2(box_GRAY);
%     % use function to generate zigzag pattern for the patch
%     zigzag_index = fn_zigzagPatternGenerator(size(box_GRAY,1), size(box_GRAY,2));
%     zigzag_index = zigzag_index(1:D_DCT); % D_DCT is the # of coefficients
%     I_DCT = dct_coef(zigzag_index);
%     Y(tmp_cnt,box_DCT) = I_DCT;
    
%     % ####### Feature ####### first 10 coef of DCT of center box 15x15
%     row_range = (row_center-8):(row_center+7); row_range = row_range(row_range>=1 & row_range<=Nrow);
%     col_range = (col_center-8):(col_center+7); col_range = col_range(col_range>=1 & col_range<=Ncol);
%     box_GRAY = I_GRAY(row_range, col_range); % figure; imagesc(box_GRAY);
%     % apply DCT to the bounding box
%     dct_coef = dct2(box_GRAY);
%     % use function to generate zigzag pattern for the patch
%     zigzag_index = fn_zigzagPatternGenerator(size(box_GRAY,1), size(box_GRAY,2));
%     zigzag_index = zigzag_index(1:D_centerDCT); % D_DCT is the # of coefficients
%     I_DCT = dct_coef(zigzag_index);
%     Y(tmp_cnt,centerbox_DCT) = I_DCT;
    
    % ####### Feature ####### avg wavelet coefficients
    % DWT level 1: HVD1
    sub_row_level1 = ceil(sub_row/2);
    sub_col_level1 = ceil(sub_col/2);
    ind_level1 = sub2ind(size(cH1), sub_row_level1, sub_col_level1);
    
    % DWT level 2: HVD2
    sub_row_level2 = ceil(sub_row/2^2);
    sub_col_level2 = ceil(sub_col/2^2);
    ind_level2 = sub2ind(size(cH2), sub_row_level2, sub_col_level2);
    
    % DWT level 1: HVD1
    sub_row_level3 = ceil(sub_row/2^3);
    sub_col_level3 = ceil(sub_col/2^3);
    ind_level3 = sub2ind(size(cH3), sub_row_level3, sub_col_level3);
    
    % combine the feature
    Y(tmp_cnt, avg_DWT) = [ mean( cH1(ind_level1).^2, 1),...
                            mean( cV1(ind_level1).^2, 1),...
                            mean( cD1(ind_level1).^2, 1),...
                            mean( cH2(ind_level2).^2, 1),...
                            mean( cV2(ind_level2).^2, 1),...
                            mean( cD2(ind_level2).^2, 1),...
                            mean( cH3(ind_level3).^2, 1),...
                            mean( cV3(ind_level3).^2, 1),...
                            mean( cD3(ind_level3).^2, 1) ];
                              
    
%     % centerBoc DWT
%     row_range = (row_center-8):(row_center+7); row_range = row_range(row_range>=1 & row_range<=Nrow);
%     col_range = (col_center-8):(col_center+7); col_range = col_range(col_range>=1 & col_range<=Ncol);
%     box_GRAY = I_GRAY(row_range, col_range); % figure; imagesc(box_GRAY);
%     % apply DWT to the box
%     [cA1,cH1,cV1,cD1] = dwt2(box_GRAY,'db1');
%     Y(tmp_cnt,centerbox_DWT) = [sqrt(sum(cH1(:).^2)) sqrt(sum(cV1(:).^2)) sqrt(sum(cD1(:).^2))]./(sqrt(sum(cH1(:).^2))+sqrt(sum(cV1(:).^2))+sqrt(sum(cD1(:).^2)));
    
    
    % SIFT
    
% %     % --- for version 2.1.0 ---- get the corresponding feature vector
% %     feature_vectors = I_feature(region_index(:),:);
% %     % mean color of the superpixel
% %     Y(tmp_cnt,:) = mean(feature_vectors,1); % average the feature within the superpixel

%##########################################################################
%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% add some feature from feature learning
load('featurelearning128035');

[tx ty] = size(img_RGB);
[tx1 ty1 tz1] = size(z);


z = z(85:405,5:485,:);
z = reshape(z(:),Ncol*Nrow,[]);

for index2 = 1:tz1
    meanz(index2)= mean( z(region_index(:),index2) ,1);
end

Y(tmp_cnt,feature_learning) = meanz;

%##########################################################################
%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    
    Y_index(tmp_cnt) = i; % the node index, in the ascend order
    tmp_cnt = tmp_cnt + 1;
end
% figure; imagesc(Y);
% Note that Y_index in version 2.1.0 still keeps the node index of level H(1)
% only. In the future we intend to use Y_index to keep all node indices
% from every level.

% ########################################################################
% ########################################################################

% % ==== test the feature extraction by reading back the feature and display ====
% % ###### average RGB image #######
% img_feature = zeros(Ncol*Nrow,3);
% for i = I_soft_evidence
%     region_index = (evidence_label==i);
%     I_read = Y(Y_index==i,avg_RGB);
%     img_feature(region_index,:) = repmat( I_read(1,:), sum(region_index(:), 1),1);
% end
% I_show = reshape(img_feature,Nrow,Ncol,[]);
% figure; imagesc(I_show); title('average RGB'); daspect([1 1 1]); axis off
% print('-djpeg','-r100',['average_RGB.jpg']);
% movefile(['average_RGB.jpg'], ['./',imagename]);
% 
% % ###### average LAB image #######
% img_feature = zeros(Ncol*Nrow,3);
% for i = I_soft_evidence
%     region_index = (evidence_label==i);
%     I_read = Y(Y_index==i,avg_LAB);
%     img_feature(region_index,:) = repmat( I_read(1,:), sum(region_index(:), 1),1);
% end
% I_show = reshape(img_feature,Nrow,Ncol,[]);
% figure; imagesc(I_show); title('average LAB'); daspect([1 1 1]); axis off
% print('-djpeg','-r100',['average_LAB.jpg']);
% movefile(['average_LAB.jpg'], ['./',imagename]);
% 
% % ###### bounding box DCT image #######
% img_feature = zeros(Ncol*Nrow,1);
% for i = I_soft_evidence
%     region_index = (evidence_label==i);
%     I_read = sum(abs(Y(Y_index==i,box_DCT)),2);
%     img_feature(region_index,:) = repmat( I_read(1,:), sum(region_index(:), 1), 1);
% end
% I_show = reshape(img_feature,Nrow,Ncol,[]);
% figure; imagesc(log(I_show)); title('bounding box DCT'); daspect([1 1 1]); axis off
% print('-djpeg','-r100',['bounding_box_DCT.jpg']);
% movefile(['bounding_box_DCT.jpg'], ['./',imagename]);
% 
% % ###### center box DCT image #######
% img_feature = zeros(Ncol*Nrow,1);
% for i = I_soft_evidence
%     region_index = (evidence_label==i);
%     I_read = sum(abs(Y(Y_index==i,centerbox_DCT)),2);
%     img_feature(region_index,:) = repmat( I_read(1,:), sum(region_index(:), 1), 1);
% end
% I_show = reshape(img_feature,Nrow,Ncol,[]);
% figure; imagesc(log(I_show)); title('center box DCT'); daspect([1 1 1]); axis off
% print('-djpeg','-r100',['center_box_DCT.jpg']);
% movefile(['center_box_DCT.jpg'], ['./',imagename]);
% 
% % ###### DWT image #######
% img_feature = zeros(Ncol*Nrow,1);
% for i = I_soft_evidence
%     region_index = (evidence_label==i);
%     I_read = Y(Y_index==i,avg_DWT(3));
%     img_feature(region_index,:) = repmat( I_read(1,:), sum(region_index(:), 1),1);
% end
% I_show = reshape(img_feature,Nrow,Ncol,[]);
% figure; imagesc(I_show); title('DWT'); daspect([1 1 1]); axis off
% print('-djpeg','-r100',['average_DWT.jpg']);
% movefile(['average_DWT.jpg'], ['./',imagename]);

% ########################################################################
% ########################################################################
% ########################################################################

save([imagename,'_feature'], 'Y', 'Y_index'); % in 2.1.0 we don't use 'evidence_label',
% We also move 'Y', 'Y_index' to feature extraction routine

% move the output to the corresponding folder
movefile(['./',imagename,'_feature.mat'], ['./',imagename]);
disp(['The results (tree structure) are saved in the folder /',imagename]);

% --- remove the path ----
rmpath(['./',imagename]);