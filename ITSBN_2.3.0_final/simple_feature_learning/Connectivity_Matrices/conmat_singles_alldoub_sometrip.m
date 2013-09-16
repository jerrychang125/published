%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates a connectivity matrix where the features maps connect to every
% single input map, all possible pairs of input maps, and the triples of input
% maps that lie on the diagonal of the connectivity matrix.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief conmat_singles_alldoub_sometrip.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief conmat_singles_alldoub_sometrip.m
%
% @param xdim This is first dimension of the connectivity matrix. This
% represents the number of input maps.
% @param ydim This is the second dimension fo the connectivity matrix. This
% represents the number of features maps.
% @retval C The specified connectivity matrix.
% @retval recommended The number of features maps that should be used given the
% requsted size and type of connectivity matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C,recommended] = conmat_singles_alldoub_sometrip(xdim,ydim)

% Only need singles along the diagonal
C = zeros(xdim,ydim);

% Put a single diagonal of ones
for j=1:xdim
    % Keep placing ones along diagonals.
    i = mod(j-1,xdim)+1;
    C(i,j)=1;
end


indices = nchoosek(1:xdim,2);
% Put a one for each possible pair
for j=1:size(indices,1)
    C(indices(j,1),j+xdim)=1;
    C(indices(j,2),j+xdim)=1;
end


start = j+xdim;
% Put a single diagonal of pairs
for j=1:xdim
    % Keep placing ones along diagonals.
    i = mod(j-1,xdim)+1;
    C(i,j+start)=1;
    C(mod(i,xdim)+1,j+start)=1;
        C(mod(i+1,xdim)+1,j+start)=1;
end

% Return the recommended size of the feature maps.
recommended = xdim+start;


end