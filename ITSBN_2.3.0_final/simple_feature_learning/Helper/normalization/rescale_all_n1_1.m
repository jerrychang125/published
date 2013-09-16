%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Scales all input data (as a lump) to roughly [-1,1] by shifting by the minimum up and
% then dividing by the max value/2 and then making it zero mean. Since it makes
% it zero mean there is no guarantee that the valeus will be [-1,1] after the scaling.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @normalization_file @copybrief rescale_all_n1_1.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief rescale_all_n1_1.m
%
% @param z This is the data you want to scale.
% @retval z This is the scaled output.
% @retval minz the mininum z before scaling
% @retval maxz the maximum z after subtracting the minimum.
% @retval meanz the mean of the resulting z that is subtracted off.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z,minz,maxz,meanz] = rescale_all_n1_1(z)
% Outputs the scaled z, dynamic range,
% Rescales the data to [0,1]
z = single(z);
minz = min(z(:));
z = z-minz;  % shifts the bottom of the array to 0.
maxz = max(z(:)); % need to use imstack so that each display is set consistently over the whole figure.
%     z(fiborder_indices==1))=0;   % Set the borders back to zero.
z = z./maxz*2;  % Scale the max to be 1.

% Find the bin with the most counts and get it's value.
%     bins = 0:0.0001:1;
%     counts = histc(z(:),bins);
%
%     [maxvalue,maxind] = max(counts);
%     maxmean = bins(maxind+1)
% Subtract by that max bin's value.
%     z = z-maxmean;

meanz = mean(z(:));
z = z-meanz; % zero mean.
end
