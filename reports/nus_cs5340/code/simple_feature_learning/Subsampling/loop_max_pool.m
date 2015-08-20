%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This selects only the max element of each pooling region and makes the other
% elements within the pooling region zero. This simulates doing max_pool and
% then reverse_max_pool but is faster.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooling_file @copybrief loop_max_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief loop_max_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @retval output the planes with only the maxes selected.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output] = loop_max_pool(input,pool_size)

if(ndims(input)==3)
    [xdim ydim numplanes] = size(input);
    numcases = 1;
else
    [xdim ydim numplanes numcases] = size(input);
end


% The pooled input planes (not dimensions 3 and 4 are reversed).
output = zeros(xdim,ydim,numplanes*numcases,'single');

% Switch the number of cases with number of maps.
% input = permute(input,[1 2 4 3]);

rows = 1:pool_size(1);
cols = 1:pool_size(2);
rblocks = ceil(xdim/pool_size(1));
cblocks = ceil(ydim/pool_size(2));
blockel = pool_size(1)*pool_size(2); %number of elements in block
x = zeros(blockel,numplanes*numcases); %this is made double to work with randbinom

% Loop over each plane (parallel over number of cases).
% for plane=1:numplanes
% Get blocks of the image.
for ii=0:rblocks-1
    for jj=0:cblocks-1
        % Get the current block for each plane and case.
        x(1:blockel,:) = reshape(input(min(ii*pool_size(1)+rows,xdim),min(jj*pool_size(2)+cols,ydim),:,:), ...
            blockel,numplanes*numcases);
        
        % Get the absolute maxes in each block.
        %             [maxes,inds] = max(abs(x));
        %             % Make the negative maxes negative again.
        % %             maxes = maxes.*sign(x(sub2ind(size(x),inds,1:size(x,2))));
        %             maxes = maxes.*sign(x(([1:size(x,2)]-1)*blockel+inds));
        
        % Get most positive and most negative numbers (and their indices).
        [maxA,maxind] = max(x);
        [minA,inds] = min(x);
        maxes = minA; % Iitialize to the mins (and their indices).
        % If abs(minA) smaller than maxA elements then replace them.
        gtind = maxA>=abs(minA);
        maxes(gtind) = maxA(gtind);
        inds(gtind) = maxind(gtind);
        
        
        % Compute offsets into the output planes.
        xoffset = rem(inds-1,pool_size(1))+1;
        yoffset = (inds-xoffset)/pool_size(1)+1+jj*pool_size(2);
        xoffset = xoffset+ii*pool_size(1);
        
        % Set the indices for all cases.
        output(([1:numcases*numplanes]-1)*xdim*ydim + ...
            (yoffset-1)*xdim + xoffset) = maxes;
    end
end
output = output(1:xdim,1:ydim,:);
output = reshape(output,xdim,ydim,numplanes,numcases);
end