%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Finds patches of patchSize spaced patchSpacing pixels over from one
% another in both x and y directions and then returns a larger plane(s) with
% all these patches spread apart and stitched together.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @spm_file @copybrief dessimate_spread.m
% @other_comp_file @copybrief dessimate_spread.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief dessimate_spread.m
%
% @param input the input image(s). xdim x ydim x colors x num_images
% @param patchSize the size of the SIFT-equivalent patches shifted over the image.
% @param patchSpacing number of pixels in original image planes to shift over when making each pooling region.
% @param gridSize the size of the region to group together on each plane to form the descriptors.
% @param poolSize the pooling region of each feature map.
%
% @retval out the spread out feature maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = dessimate_spread(input,patchSize,patchSpacing,gridSize,poolSize)

[xdim ydim numplanes numcases] = size(input);

%% make grid (coordinates of upper left patch corners)
% Thise is taken from Svetlana's GenerateSiftDescriptors code.
hgt = size(input,1);
wid = size(input,2);
remX = mod(wid-patchSize,patchSpacing*(gridSize(1)/(patchSize/poolSize(1))));
offsetX = floor(remX/2)+1;
remY = mod(hgt-patchSize,patchSpacing*(gridSize(1)/(patchSize/poolSize(1))));
offsetY = floor(remY/2)+1;

[grid_cols,grid_rows] = meshgrid(offsetX:patchSpacing:wid-patchSize+1, offsetY:patchSpacing:hgt-patchSize+1);



% fprintf('width %d, heigt %d, grid size: %d x %d, %d patches\n', ...
%     xdim, ydim, size(grid_cols,2), size(grid_cols,1), numel(grid_cols));
% grid_cols
% grid_rows


% These are the patches over the image (slid by patchSpacing pixels).
rows = 0:patchSize-1;
cols = 0:patchSize-1;
out = zeros(size(grid_cols,1)*patchSize,size(grid_cols,2)*patchSize,size(input,3),size(input,4));

% Get each block of the image and put into the larger image.
for i=1:size(grid_cols,1)
    for j=1:size(grid_cols,2)
        % Get the start indices.
        start_col = grid_cols(i,j);
        start_row = grid_rows(i,j);
        
        % Get current block for each plane and case.
        out((i-1)*patchSize+rows+1,(j-1)*patchSize+cols+1,:,:) = ...
            input(min(start_row+rows,xdim),min(start_col+cols,ydim),:,:);
    end
end



end