%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This takes all images in as a matrix and then compues the contrast normalized
% image I and resI=input-I residual image. I is assumed to be squared previously
% if you want squared images.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @image_file @copybrief CreateResidualImages.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief CreateResidualImages.m
%
% @param I input images
%
% @retval resI the residual images as: xdim-4 x ydim-4 x color_channels x
% num_images
% @retval I the contrast normalized images as: xdim-4 x ydim-4 x color_channels x
% num_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [resI,I] = CreateResidualImages(I)

% Author: Matthew Zeiler
% Returns: residual and laplacian passed images as planes in I
% Purpose:
%   Loads images from the y variables, computes laptplacians and returns
%   the results of the y-lapy residual image.

% resI = ones(size(I),'single')-0.5; % Midrange so it doesn't disrupt scaling.
% lapI = ones(size(I,1)-2,size(y,2)-2,size(y,3),size(y,4),'single'); % Midrange so it doesn't disrupt scaling.

% Loop through the number of images
% for image=1:size(y,4)
    
%     conI = zeros(size(y(:,:,:,1)),'single');
%     % Run a laplacian over images to get edge features.
%     h = fspecial('laplacian',0.2);
%     for j=1:size(y,3)  % Each color plane needs to be passed with laplacin
%         lapI(:,:,j,image) = ipp_conv2(single(y(:,:,j,image)),single(h),'valid');
% 
% %         lapI(2:end-1,2:end-1,j,image) = ipp_conv2(single(y(:,:,j,image)),single(h),'valid');
%         resI(2:end-1,2:end-1,j,image) = single(y(2:end-1,2:end-1,j,image))-single(lapI(:,:,j,image));  % Compute the residual image.
%     end
    
    
% end

    CN_I = cell(size(I,4));
    res_I = cell(size(I,4));

        %% Graham/Koray Method for Norb
        k = fspecial('gaussian',[5 5],1.591);
        for image=1:size(I,4)
            fprintf('Contrast Normalizing Image: %d\n',image);
            for j=1:size(I,3)
                dim = double(I(:,:,j,image));
                lmn = conv2(dim,k,'valid');
                lmnsq = conv2(dim.^2,k,'valid');
                lvar = lmnsq - lmn.^2;
                lvar(lvar<0) = 0; % avoid numerical problems
                lstd = sqrt(lvar);
                lstd(lstd<1) = 1;
    
                shifti = floor(size(k,1)/2)+1;
                shiftj = floor(size(k,2)/2)+1;
    
                % since we do valid convolutions
                dim = dim(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1);
                dim = dim - lmn;
                dim = dim ./ lstd;
    
                CN_I{image}(:,:,j) = dim;
                res_I{image}(:,:,j) = double(I(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1,j,image))-...
                    double(CN_I{image}(:,:,j));  % Compute the residual image.
                %             IMG = conI;
            end
        end
        
        
        
        % Now pad them again to ensure they are square.
% This has to be done after contrast normalizing to avoid strong edges on padded
% regions.
clear I PADIMG;  % Size of I may have changed by this point due to CN.
max_size = max(size(CN_I{1},1),size(CN_I{1},2));
I = zeros(max_size,max_size,size(CN_I{1},3),size(CN_I,2));
resI = zeros(max_size,max_size,size(CN_I{1},3),size(CN_I,2));
for image=1:size(CN_I,2)
    [xdim ydim planes] = size(CN_I{image});
    maxdim = max(xdim,ydim);
    PADIMG = zeros(maxdim,maxdim,planes);
    RESIMG = zeros(maxdim,maxdim,planes);
    for plane=1:planes
        tempimg = padarray(CN_I{image}(:,:,plane),[floor((maxdim-xdim)/2) floor((maxdim-ydim)/2)],'pre');
        PADIMG(:,:,plane) = padarray(tempimg,[ceil((maxdim-xdim)/2) ceil((maxdim-ydim)/2)],'post');
        tempimg = padarray(res_I{image}(:,:,plane),[floor((maxdim-xdim)/2) floor((maxdim-ydim)/2)],'pre');
        RESIMG(:,:,plane) = padarray(tempimg,[ceil((maxdim-xdim)/2) ceil((maxdim-ydim)/2)],'post');
    end
    % Store the padded images into a matrix (as they are all the same
    % dimension).
    fprintf('Squaring Image: %d\n',image);
    I(:,:,:,image) = PADIMG;
    resI(:,:,:,image) = RESIMG;
end

% Makes it [-1,1]
% I = svm_rescale2(I);
I = single(I);
resI = single(resI);

end
    
    
