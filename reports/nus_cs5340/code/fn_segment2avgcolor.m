function Irgb_avg = fn_segment2avgcolor(img_RGB,segm_output)

% ========================================================================
% This function make a good segmentation figure for display. The output
% image is the average color within the image-class label with boundary.
% img_RGB: the original image. In fact, this function used the normalized
% color range from 0 - 1.
% segm_output: is a matrix with segmentation label with the same size as
% the original image
% ========================================================================

% === average RGB color within a segment ====
if max(img_RGB(:),[],1) > 1
    img_RGB = double(img_RGB)/255; % normalize the color such that it ranges 0 to 1
end
label_unique = unique(segm_output(:),'rows');
rgb = reshape(img_RGB(:),[],3);
rgb_avg = double(rgb); % a copy
for label_idx = label_unique'
    px_index = segm_output(:) == label_idx;
    rgb_avg(px_index,1) = mean(rgb(px_index,1),1);
    rgb_avg(px_index,2) = mean(rgb(px_index,2),1);
    rgb_avg(px_index,3) = mean(rgb(px_index,3),1);
end
% average color without boundary
Irgb_avg = reshape(rgb_avg(:),size(img_RGB,1),size(img_RGB,2),[]);

% ===== make a clear contour for display =======
detected_contour = fn_segment2boundary(segm_output, 'off');
for l = 1:3 % 3 is R, G and B. We do this layer by layer
    Irgb_avg_tmp = Irgb_avg(:,:,l);
    Irgb_avg_tmp(detected_contour ~= 0) = 1; % make conour white
    Irgb_avg(:,:,l) = Irgb_avg_tmp;
end
% figure; imshow(Irgb_avg);
% print('-depsc','-r200',['segment_gmm_',imageID,'.eps']);
% print('-djpeg','-r200',['segment_gmm_',imageID,'.jpg']);
% print('-depsc','-r200',['segment_itsbn_',imageID,'.eps']);
% print('-djpeg','-r200',['segment_itsbn_',imageID,'.jpg']);
% ================================