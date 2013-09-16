%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% % This concatenates the two descriptors formed for each layer (after they
% are formed just like GenerateConvDesciptors but before making the dictionaries).
%
% Based on code by Svetlana Lazebnik for Spatial Pyramid Matching.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @spm_file @copybrief GenerateConcatenatedDescriptors.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief GenerateConcatenatedDescriptors.m
%
% @param imageFileList cell of file paths
% @param imageFileList2 cell of file paths for layer 2.
% @param trainimageBaseDir the base directory for the training image files
% @param testimageBaseDir the base directory for the testing image files
% @param trainimageBaseDir2 the base directory for the training image files for layer 2.
% @param testimageBaseDir2 the base directory for the testing image files for layer 2.
% @param train_test_split the number of training images.
% @param train_test_split2 the number of training images for layer 2.
% @param dataBaseDir the base directory for the data files that are generated by the algorithm. If this dir is the same as imageBaseDir the files will be generated in the same location as the image files
% @param maxImageSize the max plane size. (not implemented here).
% @param gridSpacing the spacing for the grid to be used when generating the sift descriptors
% @param patchSize the patch size used for generating the sift descriptor
% @param poolSize the amount to pool the image patches down before collecting them on the grid of pooled pixels.
% @param gridSize the size of the regions on the pooled maps to collect x the number of pooled maps gives the size of the descriptor.
% @param cropSize the size of the crop region to take out of the middle of each feature map.
% @param canSkip if true the calculation will be skipped if the appropriate data
% file is found in dataBaseDir. This is very useful if you just want to update
% some of the data or if you've added new images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = GenerateConcatenatedDescriptors( imageFileList, imageFileList2, trainimageBaseDir, testimageBaseDir, trainimageBaseDir2,testimageBaseDir2,train_test_split, train_test_split2,dataBaseDir, maxImageSize, gridSpacing, patchSize, poolSize, poolType, gridSize, cropSize, canSkip )


fprintf('Building Combined Convolutional Descriptors\n\n');

%% parameters

if(nargin<6)
    maxImageSize = 1000
end

if(nargin<7)
    gridSpacing = 8
end

if(nargin<8)
    patchSize = 16
end

if(nargin<9)
    poolSize = [4 4]
end

if(nargin<10)
    poolType = 'Avg'
end

if(nargin<11)
    gridSize = [4 4]
end

if(nargin<12)
    canSkip = 0
end




for f = 1:size(imageFileList,1)
    
    %% load image
    imageFName = imageFileList{f};
    imageFName2 = imageFileList2{f};
    
    [dirN base] = fileparts(imageFName);
    baseFName = [dirN filesep base];
    outFName = fullfile(dataBaseDir, sprintf('%s_sift.mat', baseFName));
    if(f<=train_test_split)
        imageFName = fullfile(trainimageBaseDir, imageFName);
        imageFName2 = fullfile(trainimageBaseDir2, imageFName2);
    else
        imageFName = fullfile(testimageBaseDir, imageFName);
        imageFName2 = fullfile(testimageBaseDir2, imageFName2);
    end
    %     imageFName = fullfile(imageBaseDir, imageFName);
    
    if(size(dir(outFName),1)~=0 && canSkip)
        fprintf('GenerateConvDescriptors Skipping %s\n', imageFName);
        continue;
    end
    
    for i=1:2
        if(i==1)
    load(imageFName); 
        else
                load(imageFName2);
        end
        I = z;



    
    %% make grid (coordinates of upper left patch corners just for staying consistent with her format)
    [hgt wid blah] = size(I);
    
    % Crop to the feature maps to the specified size.
    xdim = cropSize(1);
    ydim = cropSize(2);
    otheroffX = floor(abs(xdim-hgt)/2);
    otheroffY = floor(abs(ydim-wid)/2);
    I = I(otheroffX+1:end-otheroffX,otheroffY+1:end-otheroffY,:,:);
    [hgt wid blah blah] = size(I);
    
    
    remX = mod(wid-patchSize,gridSpacing);
    offsetX = floor(remX/2)+1;
    remY = mod(hgt-patchSize,gridSpacing);
    offsetY = floor(remY/2)+1;
    [gridX,gridY,blah] = meshgrid(offsetX:gridSpacing:wid-patchSize+1, offsetY:gridSpacing:hgt-patchSize+1,1);
    
    if(i==1)
    fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
        imageFName, wid, hgt, size(gridX,2), size(gridX,1), numel(gridX));
    else
    fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
        imageFName2, wid, hgt, size(gridX,2), size(gridX,1), numel(gridX));        
    end
    
    
    
    % Get the overlapping patches and spread them out into a large image.
    patchedI = dessimate_spread(I,patchSize,gridSpacing);
    
    % Pool the
    switch poolType
        case 'Max'
            pooled_features = max_pool(patchedI,poolSize);
        case 'Avg'
            pooled_features = avg_pool(patchedI,poolSize);
        case 'Abs_Avg'
            pooled_features = abs_avg_pool(patchedI,poolSize);
        case 'Max_Abs'
            pooled_features = abs(max_pool(patchedI,poolSize));
        case 'None'
            pooled_features = abs(patchedI);
    end
    
    if(i==1)
    % Collect the descriptors for the current image.
    % Can use the same normalization of the descriptors as Svetlana.
    features.data = make_descriptors(pooled_features,gridSize);
    %             figure(100)
    %     hist(features.data(:),100)
    %     title('Mine before rescale_row_0_1')
    
    features.data = rescale_row_0_1(features.data);
    %             figure(101)
    %     hist(features.data(:),100)
    %     title('Mine before normalizing')
    features.data = sp_normalize_sift(features.data);
    %
    %         figure(102)
    %     hist(features.data(:),100)
    %     title('Mine after normalizing')
    %     keyboard
    features.x = gridX(:) + patchSize/2 - 0.5;
    features.y = gridY(:) + patchSize/2 - 0.5;
    features.wid = wid;
    features.hgt = hgt;
    else
        features2 = make_descriptors(pooled_features,gridSize);
        features2 = rescale_row_0_1(features2);
        features2 = sp_normalize_sift(features2);        
        features.data = cat(2,features.data,features2);
    end
    
    if(size(features.x,1) ~= size(features.data,1))
        error('In GenerateCombinedDescriptors the dimensions dont match')
    end
    
    end
    
    sp_make_dir(outFName);
    save(outFName, 'features');
    
end % for

end % function