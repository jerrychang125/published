%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This concatenates each pool_size region of each plane over all planes for
% a given image.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @spm_file @copybrief make_descriptors.m
% @other_comp_file @copybrief make_descriptors.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief make_descriptors.m
%
% @param input the input image(s). xdim x ydim x colors x num_images
% @param poolSize the pooling region of each feature map.
%
% @retval output the grouped descriptors as num_desc x
% num_feature_maps*pool_size^2 x num_cases (usually 1).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output] = make_descriptors(input,pool_size)

[xdim ydim numplanes numcases] = size(input);

rows = 1:pool_size(1);
cols = 1:pool_size(2);
rblocks = ceil(xdim/pool_size(1));
cblocks = ceil(ydim/pool_size(2));
blockel = pool_size(1)*pool_size(2); %number of elements in block
output = zeros(rblocks*cblocks,blockel*numplanes,numcases,'single'); %this is made double to work with randbinom

% input = reshape(input,size(input,1),size(input,2),size(input,3)*size(input,4));
% The index to the descriptor (linear).
desc_ind = 1;
% Get blocks of the image in column order (just like their SIFT code).
for ii=0:rblocks-1
    for jj=0:cblocks-1
        % Get current block for each plane and case.
%         if(any(ii*pool_size(1)+rows > xdim) ||...
%                 any(jj*pool_size(2)+cols > ydim))
%             continue
%         else
            output(desc_ind,:,:) = reshape(input(min(ii*pool_size(1)+rows,xdim),min(jj*pool_size(2)+cols,ydim),:,:), ...
                [1 blockel*numplanes numcases]);
            desc_ind = desc_ind + 1;
%         end
    end
end

end