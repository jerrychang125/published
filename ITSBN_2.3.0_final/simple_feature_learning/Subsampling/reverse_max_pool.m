%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Undoes the max pooling by placing the max back into it's indexed location.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooling_file @copybrief reverse_max_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief reverse_max_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param indices the indices where the max came from during max_pool
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @param unpooled_size this is used to specify the correct size of the unpooled
% region. If this is not passed in then xdim*pool_size(1) x ydim*pool_size(2)
% will be used.
% @retval unpooled the unpooled output planes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [unpooled] = reverse_max_pool(input,indices,pool_size,unpooled_size)

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
% x = zeros(blockel,numcases,'single'); %this is made double to work with randbinom


if(size(indices,4)~=numcases)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Take the most seen index out of all the samples (for each block)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % There is a probabilities for each unpooled block location.
    probs = zeros(xdim*pool_size(1),ydim*pool_size(2),indplanes*indcases,'single');
    % denominator is the number if indexes times number of samples.
    %     probdenom = size(indices,4);
    
    % Make the indices into the rows.
    indices = reshape(indices,size(indices,1),size(indices,2),indplanes*indcases);            % Make the negative maxes negative again.
    
    
    %     for plane=1:numplanes
    % Get blocks of the image.
    for ii=0:rblocks-1
        for jj=0:cblocks-1
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
    % Get the indices of the maxes.
    [blah,probs_indices] = max_pool(probs,pool_size);
    % Make a copy of this for each number of input cases (not indice cases).
    probs_indices = repmat(probs_indices,[1 1 1 numcases]);
    % Reshape for speed below.
    probs_indices = reshape(probs_indices,size(probs_indices,1),size(probs_indices,2),numplanes*numcases);
    input = reshape(input,size(input,1),size(input,2),size(input,3)*size(input,4));
    
    for ii=0:rblocks-1
        for jj=0:cblocks-1
            % Make the indices into the rows.
            inds = squeeze(probs_indices(ii+1,jj+1,:));            % Make the negative maxes negative again.
            
            % Get offsets into the output image.
            xoffset = rem(inds-1,pool_size(1))+1;
            yoffset = (inds-xoffset)/pool_size(1)+1+jj*pool_size(2);
            xoffset = xoffset+ii*pool_size(1);
            
            % Set the indices for all cases.
            unpooled(([1:numplanes*numcases]'-1)*size(unpooled,1)*size(unpooled,2) + ...
                (yoffset-1)*size(unpooled,1) + xoffset) = squeeze(input(ii+1,jj+1,:));
        end
    end
    unpooled = reshape(unpooled,size(unpooled,1),size(unpooled,2),numplanes,numcases);
    
    % The unpooled input planes.
    if(nargin==4)
        % Feature map sizes (unpooled sizes) are xdim+filter_size-1...
        unpooled = unpooled(1:unpooled_size(1),1:unpooled_size(2),:,:);
    end
else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% For each sample there is an index so just use that to place max.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Make the indices into the rows.
    indices = reshape(indices,size(indices,1),size(indices,2),indplanes*indcases);            % Make the negative maxes negative again.
    input = reshape(input,size(input,1),size(input,2),size(input,3)*size(input,4));
    
    % Get blocks of the image.
    for ii=0:rblocks-1
        for jj=0:cblocks-1
            % Make the indices into the rows.
            inds = squeeze(indices(ii+1,jj+1,:));            % Make the negative maxes negative again.
            
            % Get offsets into the output image.
            xoffset = rem(inds-1,pool_size(1))+1;
            yoffset = (inds-xoffset)/pool_size(1)+1+jj*pool_size(2);
            xoffset = xoffset+ii*pool_size(1);
            
            % Set the indices for all cases.
            unpooled(([1:numcases*numplanes]'-1)*size(unpooled,1)*size(unpooled,2) + ...
                (yoffset-1)*size(unpooled,1) + xoffset) = squeeze(input(ii+1,jj+1,:));
        end
    end
    unpooled = reshape(unpooled,size(unpooled,1),size(unpooled,2),numplanes,numcases);
    
    % The unpooled input planes.
    if(nargin==4)
        % Feature map sizes (unpooled sizes) are xdim+filter_size-1...
        unpooled = unpooled(1:unpooled_size(1),1:unpooled_size(2),:,:);
    end
end

end