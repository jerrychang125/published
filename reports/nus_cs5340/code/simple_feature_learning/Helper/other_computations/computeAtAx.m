%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Coputes AtAx of the system. This is used by condition_num.m to compute the
% condition number.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @other_comp_file @copybrief computeAtAx.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief computeAtAx.m
%
% @param z feature maps (with the x*y collapsed).
% @param y input maps (with the x*y collapsed).
% @param F filters (with the x*y collapsed).
% @param xdim x-dimension of input maps.
% @param ydim y-dimension of input maps.
% @param lambda coefficient on the reconstruction term.
% @param alpha the sparsity norm.
% @retval AtAx the computed AtAx matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AtAx] = computeAtAx(z,y,F,xdim,ydim,lambda,alpha)

% Author: Matthew Zeiler
% Returns: Return the loss function evaluation and gradient w.r.t. z.
% Purpose: for use with minimize (conjugate gradient descent).
% Passes output as a vector.
% Input is not a vector.

sizez = size(z);

% Get the number of filters.
num_filters = size(F,2);
% Get the size of the filters.
filter_size = sqrt(size(F,1));

% The zks passed in here is the vectorized version of imagesize x num_filters
% Thus we should convert to the num_filters feature maps vertion by
% reshaping.
% It can simply use the size of w to reshape the z.
z = reshape(z,sizez);
% z = rand(size(z));

% beta=0;
%%%%%%
%%Compute the right hand side (A'b) term
%%%%%%
% Do the f'y convolutions.
for filter = 1:num_filters
    temp = conv2(reshape(y,xdim,ydim),flipud(fliplr(reshape(F(:,filter),filter_size,filter_size))),'full');
    % Place in correct location so when conctemp(:) is used below it will be
    % the correct vectorized form for dfz.
    conctemp(:,filter) = temp(:);
end
% This is the RHS. Only comput this once.
Atb = lambda*conctemp(:);

%%%%%%
%%Compute the left hand side (A'Ax) term
%%%%%%
% Initialize a variable to keep the running some of the other convolutions
% between f*z.
convsum = zeros(xdim,ydim);

% Loop over all the other filters and compute the sume of their
% convolutions (f*z). This is the Ax term, ie sum (f conv z)
for filter = 1:num_filters
    % Convolve F filter with z feature map and comput runnign sum.
    convsum = convsum + conv2(reshape(z(:,filter),xdim+filter_size-1,ydim+filter_size-1),...
        reshape(F(:,filter),filter_size,filter_size),'valid');
end
% This is the A'Ax term.
for filter = 1:num_filters
    temp = conv2(convsum,flipud(fliplr(reshape(F(:,filter),filter_size,filter_size))),'full');
    % Place in correct location so when conctemp(:) is used below it will be
    % the correct vectorized form for dfz.
    conctemp(:,filter) = temp(:);
end
% This is the left hand side.
AtAx = lambda*conctemp(:) + alpha*sign(z(:)).*abs(z(:)).^(alpha-1);

size(AtAx)

end