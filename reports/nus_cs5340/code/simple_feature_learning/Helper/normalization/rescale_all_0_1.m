%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Scales all input data (as a lump) to [0,1] by shifting by the minimum up and 
% then dividing by the max value.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @normalization_file @copybrief rescale_all_0_1.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief rescale_all_0_1.m
%
% @param z This is the data you want to scale.
% @retval minz the mininum z before scaling
% @retval maxz the maximum z after subtracting the minimum.
% @retval z This is the scaled output.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z,minz,maxz] = rescale_all_0_1(z)

% Rescales the data to [0,1]
minz = min(z(:));
z = z-minz;  % shifts the bottom of the array to 0.
maxz = max(z(:)-min(z(:))); % need to use imstack so that each display is set consistently over the whole figure.
%     z(fiborder_indices==1))=0;   % Set the borders back to zero.

z = z./maxz;  % Scale the max to be 1.

%     z = z-mean(z(:)); % zero mean.

%     z = double(z);  % convert to unsigned ints between (0,255)
end

