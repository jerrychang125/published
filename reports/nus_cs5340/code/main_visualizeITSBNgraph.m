clear all;
clc;
close all;

% ############### Load the image #####################
imagename = '16_28_s';
addpath(['./',imagename]);
load([imagename,'_package']); 

L = n_level+2;

% ########################################################################
% ######## CALCULATE CENTROID FOR EACH SUPERPIXEL IN EACH LEVEL ##########
% ========================================================================
% calculate the centroid of each superpixel
% ########################################################################
centroid = zeros(size(Z,1), 3); % col#1: node label j, col#2 and 3: centroid coordinate in row and column

for l = L-1:-1:1
    
    % load the superpixel indices in level l
    if l == L-1
        % Since superpixel_indx_child does not have the superpixel at level
        % L-1, therefore, we have to make a fake one by doing this. Note
        % that the node in level L-1 is 1.
        superpixel_indx_child = 1+0*node_label_hier{1,3};
    else
        superpixel_indx_child = node_label_hier{l,3};
    end

    % calculate the centroid
    for j = H{l +1, 2};
        superpixel_j_indx = find(superpixel_indx_child == j);
        [superpixel_j_row, superpixel_j_col] = ind2sub( size( superpixel_indx_child), superpixel_j_indx);
        % find centroid in row direction
        centroid_row = median(unique(superpixel_j_row,'rows'),1);
        centroid_row = ceil(centroid_row); % make it integer
        % find centroid in column direction
        centroid_col = median( superpixel_j_col(superpixel_j_row == centroid_row), 1);
        centroid_col = ceil(centroid_col);  % make it integer
        % store the centroid information
        centroid(j,1) = j;
        centroid(j,2) = centroid_row;
        centroid(j,3) = centroid_col;
    end
end
centroid = centroid(centroid(:,1)~=0,:);

% #########################################################################
% ################## VISUALIZE ITSBN IN EACH LAYER #######################
% ========================================================================
% plot the connection in each level
% ########################################################################

% ########################################################################
% plot connectivity in each level 
% ========================================================================
for l = L-3:-1:1
    boundary_child = fn_segment2boundary(node_label_hier{l,3}, 'off');
    boundary_parent = fn_segment2boundary(node_label_hier{l+1,3}, 'off');
    boundary_child(boundary_parent==1) = 2;
    figure; hold on;
    imagesc(boundary_child); daspect([1 1 1]);
    
    for j = H{l +1,2}
        i = find(Z(j,:)==1);
        c_child = centroid(centroid(:,1)==j,[2 3]);
        c_parent = centroid(centroid(:,1)==i,[2 3]);
        % draw a line between each connection
        Y = [c_child(1) c_parent(1)];
        X = [c_child(2) c_parent(2)];
        line(X,Y);
        % plot the centroid of each superpixel
        plot(centroid(centroid(:,1)==j,3), centroid(centroid(:,1)==j,2), 'b.');
        plot(centroid(centroid(:,1)==i,3), centroid(centroid(:,1)==i,2), 'r*');
    end

end

% ########################################################################
% overlay the connectivity on the original image
% ========================================================================
for l = L-3:-1:1
    Irgb = imread([imagename,'.bmp']);
    Irgb_norm = double(Irgb)/255;
    boundary_child = fn_segment2boundary(node_label_hier{l,3}, 'off');
    boundary_parent = fn_segment2boundary(node_label_hier{l+1,3}, 'off');
    
    % overlay the boundary on the image
    img_fade = 0.8;
    Irgb_norm_R = img_fade*Irgb_norm(:,:,1);
    Irgb_norm_G = img_fade*Irgb_norm(:,:,2);
    Irgb_norm_B = img_fade*Irgb_norm(:,:,3);
    
    Irgb_norm_R(boundary_child==1) = 0;
    Irgb_norm_G(boundary_child==1) = 1;
    Irgb_norm_B(boundary_child==1) = 0;
    
    Irgb_norm_R(boundary_parent==1) = 1;
    Irgb_norm_G(boundary_parent==1) = 0;
    Irgb_norm_B(boundary_parent==1) = 0;

    Irgb_norm(:,:,1) = Irgb_norm_R;
    Irgb_norm(:,:,2) = Irgb_norm_G;
    Irgb_norm(:,:,3) = Irgb_norm_B;
%     boundary_child(boundary_parent==1) = 2;
    figure; hold on;
    imagesc(Irgb_norm); daspect([1 1 1]);
    
    for j = H{l +1,2}
        i = find(Z(j,:)==1);
        c_child = centroid(centroid(:,1)==j,[2 3]);
        c_parent = centroid(centroid(:,1)==i,[2 3]);
        % draw a line between each connection
        Y = [c_child(1) c_parent(1)];
        X = [c_child(2) c_parent(2)];
        line(X,Y);
        % plot the centroid of each superpixel
        plot(centroid(centroid(:,1)==j,3), centroid(centroid(:,1)==j,2), 'b.');
        plot(centroid(centroid(:,1)==i,3), centroid(centroid(:,1)==i,2), 'r*');
    end

end





