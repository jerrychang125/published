%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This puts the average value of each pooling region into each element
% within the pooling region. This simulates doing avg_pool and
% then reverse_avg_pool but is faster.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooling_file @copybrief loop_avg_pool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief loop_avg_pool.m
%
% @param input the planes to be normalized (xdim x ydim x n3 [x n4])
% @param pool_size a vector specifying the pooling sizein x and y dimensions.
% @retval output the planes with only the maxes selected.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output] = loop_avg_pool(input,pool_size)


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

input = reshape(input,size(input,1),size(input,2),size(input,3)*size(input,4));


% Loop over each plane (parallel over number of cases).
% for plane=1:numplanes
% Get blocks of the image.
for ii=0:rblocks-1
    for jj=0:cblocks-1
        % Get the current block for each plane and case.
        x(1:blockel,:) = reshape(input(min(ii*pool_size(1)+rows,xdim),min(jj*pool_size(2)+cols,ydim),:), ...
            blockel,numplanes*numcases);
        
        output(ii*pool_size(1)+rows,jj*pool_size(2)+cols,:) = ...
            reshape(repmat(mean(x,1),[pool_size(1)*pool_size(2) 1]),...
            pool_size(1),pool_size(2),numplanes*numcases);
    end
end
output = output(1:xdim,1:ydim,:);
output = reshape(output,xdim,ydim,numplanes,numcases);
end