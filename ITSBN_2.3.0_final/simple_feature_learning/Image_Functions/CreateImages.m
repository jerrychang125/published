%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This takes all images from the input folder, converts them to the desired
% colorspace, removes mean/divides by standard deviations (if desired), and
% constrast normalizes the image (if desired). If the images are of different
% sizes, then it will padd them with zeros (after contrast normalizing) to make
% them square (assumes that they all images have the same maximum dimension).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @image_file @copybrief CreateImages.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief CreateImages.m
%
% @param imgs_path to a folder contain only image files (or other folder
% which will be ignored).
% @param CONTRAST_NORMALIZE [optional] binary value indicating whether to contrast
% normalize the images. Defaults to 1.
% @param ZERO_MEAN [optional] binary value indicating whether to subtract the mean and divides by standard deviation (current
% commented out in the code). Defuaults to 1.
% @param COLOR_TYPE [optional] a string of: 'gray','rgb','ycbcr','hsv'. Defaults to 'gray'.
% @param SQUARE_IMAGES [optional] binary value indicating whether or not to square the
% images. This must be used if using different sized images. Even then the max
% dimensions of each image must be the same. Defaults to 0.
%
% @retval I the images as: xdim x ydim x color_channels x num_images
% @retval mn the mean if ZERO_MEAN was set.
% @retval sd the standard deviation if ZERO_MEAN was set.
% @retval xdim the size of the images in x direction.
% @retval ydim the size of the images in y direction.
% @retval resI the (image-contrast normalized image) if CONTRAST_NORMALIZE is
% set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I,mn,sd,xdim,ydim,resI] = CreateImages(imgs_path,CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_TYPE,SQUARE_IMAGES)

% Defaults
if(nargin<5)
    SQUARE_IMAGES = 0;
end

if(nargin<4)
    COLOR_TYPE = 'gray';
end

if(nargin<3)
    ZERO_MEAN = 1;
end

if(nargin<2)
    CONTRAST_NORMALIZE = 1;
end


% For backwards compatibility, revert to grayscale.
if(isnumeric(COLOR_TYPE))
    if(COLOR_TYPE == 1)
        COLOR_TYPE = 'rgb';
    else
        COLOR_TYPE = 'gray';
    end
end

% Cell array listing all files and paths.
subdir = dir(imgs_path);

[blah,files] = split_folders_files(subdir);

% Counter for the image
image = 1;

if(size(files,1) == 0)
    error('No Images in this directory');
end
    

% Loop through the number of files ignoring . and ..
for file=1:length(files)
    % Makes sure not to count subdirectories
    if (files(file).isdir == 0)
        
        % Get the path to the given file.
        img_name = strcat(imgs_path,files(file).name);
        
        fprintf('Loading: %s\n',img_name);
        
        % Load the image file
        rgb_im = double(imread(img_name));
        
        switch(COLOR_TYPE)
            case 'rgb'
                fprintf('Making RGB Image\n');
                IMG = rgb_im;
                % Normalize the RGB values to [0,1] (do not do this on YCbCr!!!!!).
                IMG = cast(IMG,'double')./255;
            case 'ycbcr'
                fprintf('Making YUV Image\n');
                IMG = rgb2ycbcr(double(rgb_im)/255.0);
            case 'hsv'
                fprintf('Making HSV Image\n');
                IMG = rgb2hsv(double(rgb_im)/255.0);
            case 'gray'
                fprintf('Making Gray Image\n');
                % Convert to grayscale
                if (size(rgb_img,3) == 3)
                    IMG = double(rgb2gray(rgb_img));
                else
                    IMG = double(rgb_img);
                end
        end
        
        %% If you wnat to convert to YCbCr space.
        %                 IMG = rgb2ycbcr(double(rgb_im)/255);
        
        
        % Original image dimensions
        [xdim,ydim] = size(IMG(:,:,1));
        
        
        
        % Not used anymore.
        good_ind = 0;
        
        % Put the image in a cell array.
        orig_I{file} = IMG;
        
        % Reshape the IMG so that it is square by padding with zeros.
        [xdim ydim colors] = size(IMG);
        
        maxdim = max(xdim,ydim);
        PADIMG = zeros(maxdim,maxdim,colors);
        for plane=1:size(IMG,3)
            tempimg = padarray(IMG(:,:,plane),[floor((maxdim-xdim)/2) floor((maxdim-ydim)/2)],'pre');
            PADIMG(:,:,plane) = padarray(tempimg,[ceil((maxdim-xdim)/2) ceil((maxdim-ydim)/2)],'post');
        end
        % Store the padded images into a matrix (as they are all the same
        % dimension).
        I(:,:,:,file) = PADIMG;
        
        % Increment the number of images found so far.
        image=image+1;
    end
    
end

%% Subtract the mean and divide by STD?
if ZERO_MEAN
    fprintf('Computing Means\n')
    
    % Take the mean/std of each plane of the images independently.
    mn = squeeze(mean(mean(I)));
    sd = squeeze(std(std(I)));
    for image=1:size(I,4)
        for i=1:size(I,3)
            orig_I{image}(:,:,i) = orig_I{image}(:,:,i) - mn(i,image);
            % orig_I{image(:,:,i) = orig_I{image}(:,:,i)/sd(i,image);
        end
    end
else
    %     mn = zeros(size(I,3),size(I,4));
    mn = [];
    %     sd = zeros(size(I,3),size(I,4));
    sd = [];
end



%% Contrast normalize the image?
if CONTRAST_NORMALIZE
  
    
        CN_I = cell(size(I,4));
    res_I = cell(size(I,4));
    
%         % Run a laplacian over images to get edge features.
%     h = fspecial('laplacian',0.2);
%                 
%             shifti = floor(size(h,1)/2)+1;
%             shiftj = floor(size(h,2)/2)+1;
% % Loop through the number of images
% for image=1:size(I,4)
%         fprintf('Contrast Normalizing Image with Laplacian: %d\n',image);
%     for j=1:size(I,3)  % Each color plane needs to be passed with laplacin
%         CN_I{image}(:,:,j) = ipp_conv2(single(orig_I{image}(:,:,j)),single(h),'valid');
%         res_I{image}(:,:,j) = double(orig_I{image}(shifti:shifti+size(CN_I{image},1)-1,shiftj:shiftj+size(CN_I{image},2)-1,j))-double(CN_I{image}(:,:,j));  % Compute the residual image.
%     end
% end
    
    

  
    %% Local Constrast Normalization.
    k = fspecial('gaussian',[5 5],1.591);
    for image=1:size(I,4)
        fprintf('Contrast Normalizing Image with Local CN: %d\n',image);
        for j=1:size(I,3)
            dim = double(orig_I{image}(:,:,j));
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
            res_I{image}(:,:,j) = double(orig_I{image}(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1,j))-double(CN_I{image}(:,:,j));  % Compute the residual image.
            %             IMG = conI;
        end
    end
else
    % Then make the result the original image.
    CN_I = orig_I;
    res_I = orig_I;
    clear orig_I;
end





clear I PADIMG;  % Size of I may have changed by this point due to CN.
if(SQUARE_IMAGES)
    % Now pad them again to ensure they are square.
    % This has to be done after contrast normalizing to avoid strong edges on padded
    % regions.
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
else
    
    %fprintf('XXXXXXXXXX Squaring is commencted out right now XXXXX')
    for image=1:size(CN_I,2)
        I(:,:,:,image) = CN_I{image};
    end
end

% Get the sizes.
[xdim ydim planes cases] = size(I);




