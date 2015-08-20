function [] = QuickShiftSingleImage(imagename, imageext, blursize, MaxDist_list)

% ====== USER INPUT ===========
% % ===== input parameters =======
% imagename = '113016';
% blursize = 2;
% MaxDist_list = [10 20 35 50 80 130]; % good for 299091, 108073 105053 61060 299091 232038 41004
% % ==============================

img_label_hier = cell(length(MaxDist_list),3);
[s,mess,messid] = mkdir(['./',imagename]);
disp(mess);

tmp_maxdist = 1;
for maxdist = MaxDist_list
    % --- segmentation parameters ---
    ratio = 0.1;
    kernelsize = 5;
    %     maxdist = 30; % 8-385
    ndists = 10;
    
    filename_output = [imagename,' kernel',num2str(kernelsize),' ratio',num2str(10*ratio),' maxdist',num2str(maxdist),' blursize',num2str(blursize)];
    % =============================
    
    % import an image'
    img_RGB = imread(['./',imagename, '/',imagename,imageext]);
    %img_RGB = imread([imagename,'.jpg']);
    Ncol = size(img_RGB,2);
    Nrow = size(img_RGB,1);
    
    H = fspecial('disk',blursize);
    blurred = imfilter(img_RGB,H,'replicate');
    figure(101); imshow(blurred); title('Blurred Image');
    
    % ---- segmentation only one image -----
    I = blurred;
    [Iseg LABELS MAPS GAPS E] = vl_quickseg(I, ratio, kernelsize, maxdist);
    
    % ----- overlay the label on the original image ----------
    
    I_gray = double(rgb2gray(img_RGB))/255;
    I_gray = I_gray+0.8; I_gray = I_gray/max( max(I_gray,[],1),[],2); % fade the gray scale a bit for proper display
    % figure; imagesc(I_gray); axis equal off tight; colormap gray; caxis([0 1]);
    I_tmp = double(label2rgb(LABELS))/255; % convert label to RGB
    I_overlay = I_tmp;
    % ---- overlay the label image on the gray scale image --------
    I_overlay(:,:,1) = I_tmp(:,:,1).*I_gray;
    I_overlay(:,:,2) = I_tmp(:,:,2).*I_gray;
    I_overlay(:,:,3) = I_tmp(:,:,3).*I_gray;
    
    figure(102); imagesc(I_overlay); daspect([1 1 1]); title([filename_output,' nlabels',num2str(length(unique(LABELS)))]);
    print('-djpeg','-r100',[filename_output,' nlabels',num2str(length(unique(LABELS))),'.jpg']);
    movefile(['./',filename_output,' nlabels',num2str(length(unique(LABELS))),'.jpg'], ['./',imagename]);
    
    % --- update count box ---
    img_label_hier{tmp_maxdist,1} = maxdist;
    img_label_hier{tmp_maxdist,2} = length(unique(LABELS));
    img_label_hier{tmp_maxdist,3} = LABELS;
    
    tmp_maxdist = tmp_maxdist + 1;
end

save([imagename,'_img_label_hier'], 'img_label_hier') 


% move the output to the corresponding folder
movefile(['./',imagename,'_img_label_hier.mat'], ['./',imagename]);
disp(['The results (hierarchical segmentation) are saved in the folder /',imagename]);
