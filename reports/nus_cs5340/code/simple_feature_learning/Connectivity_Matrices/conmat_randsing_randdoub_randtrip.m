%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates a connectivity matrix where the features maps connect to rnadomly 
% selected single connections to input maps, pairs of inputs maps, and triples
% of input maps. An even proportion of each is used.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief conmat_randsing_randdoub_randtrip.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief conmat_randsing_randdoub_randtrip.m
%
% @param xdim This is first dimension of the connectivity matrix. This
% represents the number of input maps.
% @param ydim This is the second dimension fo the connectivity matrix. This
% represents the number of features maps.
% @retval C The specified connectivity matrix.
% @retval recommended The number of features maps that should be used given the
% requsted size and type of connectivity matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C,recommended] = conmat_randsing_randdoub_randtrip(xdim,ydim)
% An even number of random singles, doubles, and triples.

% Only need singles along the diagonal
C = zeros(xdim,ydim);

% Put a single diagonal of ones
for j=1:floor(ydim/3)
    % Keep placing ones along diagonals.
    indices = randperm(xdim);
    C(indices(1),j)=1;
end

start = j;
for j=1:floor(ydim/3)
    indices = randperm(xdim);
    C(indices(1),j+start)=1;
    C(indices(2),j+start)=1;
end


start = j+start;
% Put a one for each possible pair
for j=1:ceil(ydim/3)
    indices = randperm(xdim);
    C(indices(1),j+start)=1;
    C(indices(2),j+start)=1;
    C(indices(3),j+start)=1;
end

% Make sure the last column always has something.
if(~any(C(:,ydim)))
       indices = randperm(xdim);
    C(indices(1),ydim)=1;
    C(indices(2),ydim)=1;
    C(indices(3),ydim)=1; 
end


% Return the recommended size of the feature maps.
recommended = ydim;


end