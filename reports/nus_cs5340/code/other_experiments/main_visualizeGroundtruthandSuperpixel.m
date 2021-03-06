close all;
clc;
clear all;

% load a segmentation
imageID = '1_23_s'; imageext = '.bmp';
addpath(['./',imageID]); % folder of the superpixel segment
addpath('/home/student1/MATLABcodes/MSRC_ObjCategImageDatabase_v2/Images'); % original image folder
addpath('/home/student1/MATLABcodes/MSRC_ObjCategImageDatabase_v2/GroundTruth'); % groundtruth image folder
addpath('../'); % add path to the main code


% load/read images
load([imageID,'_img_label_hier']);
Iorg = imread([imageID,imageext]); Irgb = double(Iorg)/255; % original image
Igt = imread([imageID,'_GT',imageext]); Igt = double(Igt)/255; % groundtruth image
Igt = 2*Igt; Igt(Igt>1)=1; % adjust the brightness of the groundtruth image

% make gray scale out of the original image
Igray = sum(Irgb,3)/3;
% Igray = 2*Igray; Igray(Igray>1) = 1; % adjust the brightness

cnt_level = 1;
for l = 4:-1:1 % level Hl
    
    segm_img = img_label_hier{l,3}; % the segmentation image
    
    % make boundary image from the superpixel image
    Ibnd = fn_segment2boundary(segm_img, 'off');
    
    % #####################################################################
    % overlay the original image, the groundtruth and the boundary on the
    % same image
    % #####################################################################
    Iorg_gt_boundary = fn_overlayGTORGBND(Irgb, Igt, Ibnd);
    
    subplot(2,2,cnt_level); imshow(Iorg_gt_boundary);
    title(['superpixel N=',num2str(img_label_hier{l,2})]);
    cnt_level = cnt_level + 1;
end

%     print('-depsc','-r200',[imageID,'_overlay.eps']);
print('-djpeg','-r200',[imageID,'_overlay.jpg']);
%     h = gcf; saveas(h,['cost_function_',num2str(exp_number),'.fig'])
movefile([imageID,'_overlay.jpg'], ['./',imageID]);
