%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates a connectivity matrix where the features maps connect to every
% possible pair of input maps.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief conmat_alldoub.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief conmat_alldoub.m
%
% @param xdim This is first dimension of the connectivity matrix. This
% represents the number of input maps.
% @param ydim This is the second dimension fo the connectivity matrix. This
% represents the number of features maps.
% @retval C The specified connectivity matrix.
% @retval recommended The number of features maps that should be used given the
% requsted size and type of connectivity matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C,recommended] = conmat_alldoub(xdim,ydim)

% Only need singles along the diagonal
C = zeros(xdim,ydim);

indices = nchoosek(1:xdim,2);
% Put a one for each possible pair
for j=1:size(indices,1)
    C(indices(j,1),j)=1;
    C(indices(j,2),j)=1;
end

% Return the recommended size of the feature maps.
recommended = size(indices,1);


end