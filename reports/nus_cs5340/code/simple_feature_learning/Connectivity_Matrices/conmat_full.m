%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates a connectivity matrix where the features maps connect to every
% input maps.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief conmat_full.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief conmat_full.m
%
% @param xdim This is first dimension of the connectivity matrix. This
% represents the number of input maps.
% @param ydim This is the second dimension fo the connectivity matrix. This
% represents the number of features maps.
% @retval C The specified connectivity matrix.
% @retval recommended The number of features maps that should be used given the
% requsted size and type of connectivity matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C,recommended] = conmat_full(xdim,ydim)

% Fully connected connectivity matrix is just ones.
C = ones(xdim,ydim);

% Return the ydim for teh recommended size.
recommended = ydim;

end