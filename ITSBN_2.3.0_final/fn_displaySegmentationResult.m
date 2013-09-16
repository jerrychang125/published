% ========================================================================
% resulting plot will show the segment (color) overlaid on the gray-scale
% version of the original image ing_RGB.
% 
% ------------------- parameters ------------------------------------
% class_result: Nsup-D column vector represents class/label, class_result(i)
% is the label of the superpixel/region result_index(i).
% result_index: Nsup-D column vector represents the ID of the region.
% region_field: a Nrow x Ncol matrix -- a map showing which pixel in the image is mapped to which region ID. 
% img_RGB = Nrow x Ncol x 3 matrix read from a color image
% 
% ------ example -----------
% [I_out] = displaySegmentationResult(class_result, result_index, region_field, img_RGB)
% figure; imagesc(I_out);


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


function [I_out] = displaySegmentationResult(class_result, result_index, region_field, img_RGB)

% ===== coloring the patches in the region field =============
I_label = region_field;
for i_tmp = result_index
    I_label(region_field==i_tmp) = class_result(result_index==i_tmp);
end

I_out = I_label;

if nargin > 3

% ---- plot the label ----
I_tmp = double(label2rgb(I_label))/255; % convert label to RGB 
I_tmp = I_tmp + 0.0; % make color softer
I_tmp(I_tmp > 1) = 1; % threshold it
% ----- overlay the label on the original image ----------
I_gray = double(rgb2gray(img_RGB))/255; 
I_gray = I_gray+1; I_gray = I_gray/max( max(I_gray,[],1),[],2); % fade the gray scale a bit for proper display
% figure; imagesc(I_gray); axis equal off tight; colormap gray; caxis([0 1]);
I_overlay = I_tmp;
% ---- overlay the label image on the gray scale image -------- 
I_overlay(:,:,1) = I_tmp(:,:,1).*I_gray;
I_overlay(:,:,2) = I_tmp(:,:,2).*I_gray;
I_overlay(:,:,3) = I_tmp(:,:,3).*I_gray;

I_out = I_overlay;
end


     