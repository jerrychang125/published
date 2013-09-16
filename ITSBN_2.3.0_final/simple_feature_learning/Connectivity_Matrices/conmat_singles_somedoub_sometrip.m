%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates a connectivity matrix where the features maps connect to every single
% input map, the diagonal pairs of input maps, and the diagonal triples of input
% maps.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief conmat_singles_somedoub_sometrip.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief conmat_singles_somedoub_sometrip.m
%
% @param xdim This is first dimension of the connectivity matrix. This
% represents the number of input maps.
% @param ydim This is the second dimension fo the connectivity matrix. This
% represents the number of features maps.
% @retval C The specified connectivity matrix.
% @retval recommended The number of features maps that should be used given the
% requsted size and type of connectivity matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C,recommended] = conmat_singles_somedoub_sometrip(xdim,ydim)

% Only need singles along the diagonal
C = zeros(xdim,ydim);

% Put a single diagonal of ones
for j=1:xdim
    % Keep placing ones along diagonals.
    i = mod(j-1,xdim)+1;
    C(i,j)=1;
end


% Put a single diagonal of pairs
for j=1:xdim
    % Keep placing ones along diagonals.
    i = mod(j-1,xdim)+1;
    C(i,j+xdim)=1;
    C(mod(i,xdim)+1,j+xdim)=1;
end


% Put a single diagonal of pairs
for j=1:xdim
    % Keep placing ones along diagonals.
    i = mod(j-1,xdim)+1;
    C(i,j+2*xdim)=1;
    C(mod(i,xdim)+1,j+2*xdim)=1;
        C(mod(i+1,xdim)+1,j+2*xdim)=1;
end

% Return the recommended size of the feature maps.
recommended = xdim*3;


end