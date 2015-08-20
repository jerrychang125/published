%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Scales each row of input data to [0,1] by shifting by the minimum up and 
% then dividing by the max value.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @normalization_file @copybrief rescale_row_0_1.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief rescale_row_0_1.m
%
% @param z This is the data you want to scale.
% @retval minz the mininum z before scaling
% @retval maxz the maximum z after subtracting the minimum.
% @retval z This is the scaled output.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z,minz,maxz] = rescale_row_0_1(z)
% Scales the descriptors (each row).
% They are assumed to be positive already.

% Rescales the data to [0,1]
z = single(z);
minz = min(z,[],2);
z = z-repmat(minz,1,size(z,2));  % shifts the bottom of the array to 0.
maxz = max(z,[],2); % need to use imstack so that each display is set consistently over the whole figure.
z = z./repmat(maxz,1,size(z,2));  % Scale the max to be 1.
% z = z*0.5;
end
