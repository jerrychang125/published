%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates a connectivity matrix where the features maps connect to the pairs
% of input maps that lie along the diagonal of the connectivity matrix.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief conmat_somedoub.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief conmat_somedoub.m
%
% @param xdim This is first dimension of the connectivity matrix. This
% represents the number of input maps.
% @param ydim This is the second dimension fo the connectivity matrix. This
% represents the number of features maps.
% @retval C The specified connectivity matrix.
% @retval recommended The number of features maps that should be used given the
% requsted size and type of connectivity matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C,recommended] = conmat_somedoub(xdim,ydim)

% Only need singles along the diagonal
C = zeros(xdim,ydim);


% Put a single diagonal of pairs
for j=1:xdim
    % Keep placing ones along diagonals.
    i = mod(j-1,xdim)+1;
    C(i,j)=1;
    C(mod(i,xdim)+1,j)=1;
end

% Return the recommended size of the feature maps.
recommended = xdim;


end