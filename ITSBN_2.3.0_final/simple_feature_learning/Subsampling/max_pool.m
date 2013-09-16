%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Max pools the input maps within pool_size region.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooling_file @copybrief max_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief max_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @retval pooled the pooled output planes
% @retval indices the indice within each pool region that was selected as max.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pooled,indices] = max_pool(input,pool_size)

[xdim ydim numplanes numcases] = size(input);

% The pooled input planes (not dimensions 3 and 4 are reversed).
pooled = zeros(ceil(xdim/pool_size(1)),ceil(ydim/pool_size(2)),numplanes*numcases,'single');
% Store the indices for each plane.
% indices = zeros(size(im2col(input(:,:,1),pool_size,'distinct'),2),numplanes);
indices = zeros(ceil(xdim/pool_size(1)),ceil(ydim/pool_size(2)),numplanes*numcases);


% Switch the number of cases with number of maps.
% input = permute(input,[1 2 4 3]);

% Reshape all the planes into one large image.
% input = reshape(input,xdim,ydim*numplanes,numcases);

rows = 1:pool_size(1);
cols = 1:pool_size(2);
rblocks = ceil(xdim/pool_size(1));
cblocks = ceil(ydim/pool_size(2));
blockel = pool_size(1)*pool_size(2); %number of elements in block
x = zeros(blockel,numplanes*numcases,'single'); %this is made double to work with randbinom

input = reshape(input,size(input,1),size(input,2),size(input,3)*size(input,4));


% Get blocks of the image.
for ii=0:rblocks-1
    for jj=0:cblocks-1
        % Get current block for each plane and case.
        x(1:blockel,:) = reshape(input(min(ii*pool_size(1)+rows,xdim),min(jj*pool_size(2)+cols,ydim),:), ...
            blockel,numplanes*numcases);
        
        % Get most positive and most negative numbers (and their indices).
        [maxA,maxind] = max(x);
        [minA,inds] = min(x);
        maxes = minA; % Iitialize to the mins (and their indices).
        % If abs(minA) smaller than maxA elements then replace them.
        gtind = maxA>=abs(minA);
        maxes(gtind) = maxA(gtind);
        inds(gtind) = maxind(gtind);
        
        % Set the indices for all cases.
        indices(ii+1,jj+1,:) = inds;
        pooled(ii+1,jj+1,:) = maxes;
    end
end
indices = reshape(indices,size(indices,1),size(indices,2),numplanes,numcases);
pooled = reshape(pooled,size(pooled,1),size(pooled,2),numplanes,numcases);

end