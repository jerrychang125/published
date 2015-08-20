%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This uses a set of training examples to compute a probability of which index
% within each pool region that the max comes from. Then this probability is
% sampled and used to get an index for each pooling region to place the max
% value back into.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooling_file @copybrief reverse_prob_max_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief reverse_prob_max_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param indices the indices where the max came from during max_pool
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param unpooled_size this is used to specify the correct size of the unpooled
% region. If this is not passed in then xdim*pool_size(1) x ydim*pool_size(2)
% will be used.
% @retval unpooled the unpooled output planes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unpooled] = reverse_prob_max_pool(input,indices,pool_size,unpooled_size)

if(ndims(input)==3)
    [xdim ydim numplanes] = size(input);
    numcases = 1;
else
    [xdim ydim numplanes numcases] = size(input);
end

if(ndims(indices)==3)
    [indxdim indydim indplanes] = size(indices);
    indcases = 1;
else
    [indxdim indydim indplanes indcases] = size(indices);
end


% The unpooled input planes.
% if(nargin<4)
unpooled = zeros(ceil(xdim*pool_size(1)),ceil(ydim*pool_size(2)),numplanes*numcases,'single');
% else % Will need this to be computed externally.
%     % Feature map sizes (unpooled sizes) are xdim+filter_size-1...
%     unpooled = zeros(unpooled_size(1),unpooled_size(2),numplanes*numcases,'single');
% end


% Switch the number of cases with number of maps.
% input = permute(input,[1 2 4 3]);


rblocks = xdim;
cblocks = ydim;
rows = 1:pool_size(1);
cols = 1:pool_size(2);
blockel = pool_size(1)*pool_size(2); %number of elements in block
x = zeros(blockel,numplanes*numcases,'single'); %this is made double to work with randbinom

%% Get probabilities from the indices.
% There is a probabilities for each unpooled block location.
probs = zeros(xdim*pool_size(1),ydim*pool_size(2),indplanes*indcases,'single');

% Reshape indices for speed below.
indices = reshape(indices,size(indices,1),size(indices,2),indplanes*indcases);


%     for plane=1:numplanes
% Get blocks of the image.
for ii=0:rblocks-1
    for jj=0:cblocks-1
        % Make the indices into the rows.
        inds = squeeze(indices(ii+1,jj+1,:));
        
        % Get offsets into the output image.
        xoffset = rem(inds-1,pool_size(1))+1;
        yoffset = (inds-xoffset)/pool_size(1)+1+jj*pool_size(2);
        xoffset = xoffset+ii*pool_size(1);
        
        % Set the indices for all cases (into the output dimension)
        probs(([1:indcases*indplanes]'-1)*size(unpooled,1)*size(unpooled,2) + ...
            (yoffset-1)*size(unpooled,1) + xoffset) = 1;
    end
end

% Make planar again.
probs = reshape(probs,size(probs,1),size(probs,2),indplanes,indcases);
% Take mean over numcases to get the probabiliites.
probs = mean(probs,4);

% Make one of these for each input case (not indice case).
% Note probs is now the size of the unpooled image.
probs = repmat(probs,[1 1 1 numcases]);

% Indic for the sampled probabilities.
probs = reshape(probs,size(probs,1),size(probs,2),numplanes*numcases);

input = reshape(input,xdim,ydim,numplanes*numcases);


% Get blocks of the probabilities (loops over pooled image size).
for ii=0:xdim-1
    for jj=0:ydim-1
        % Get current block for each plane and case.
        x(1:blockel,:) = reshape(probs(ii*pool_size(1)+rows,jj*pool_size(2)+cols,:), ...
            blockel,numplanes*numcases);
        % Sample from the probabilities (in each column) to give indices.
        inds = mat_sample_discrete(x);
        
        % Get offsets into the output image.
        xoffset = rem(inds-1,pool_size(1))+1;
        yoffset = (inds-xoffset)/pool_size(1)+1+jj*pool_size(2);
        xoffset = xoffset+ii*pool_size(1);
        %         [xind yind] = ind2sub(pool_size,inds);
        %         xoffset = xind+ii*pool_size(1);
        %         yoffset = yind+jj*pool_size(2);
        
        % Set the indices for all cases.
        unpooled(([1:numplanes*numcases]-1)*size(unpooled,1)*size(unpooled,2) + ...
            (yoffset-1)*size(unpooled,1) + xoffset) = reshape(input(ii+1,jj+1,:),numplanes*numcases,1);
    end
end

unpooled = reshape(unpooled,size(unpooled,1),size(unpooled,2),numplanes,numcases);

% The unpooled input planes.
if(nargin==4)
    % Feature map sizes (unpooled sizes) are xdim+filter_size-1...
    unpooled = unpooled(1:unpooled_size(1),1:unpooled_size(2),:,:);
end


end
