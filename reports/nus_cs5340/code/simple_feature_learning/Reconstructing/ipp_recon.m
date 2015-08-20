%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Reconstructs the input maps from the feature maps convolved with the filters 
% (and possibly z0 maps as well) (image) using
% the IPP libraries (and thus is fast).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @recon_file @copybrief ipp_recon.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief ipp_recon.m
%
% @param z0 the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps).
% @param z0_filter_size the size of the z0 filters (if used).
% @param z the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps).
% @param F the filters (Fxdim x Fydim x num_input_maps x num_feature_maps).
% @param C the connectivity matrix for the layer.
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
% @param COMP_THREADS the number of threads to split computation over.
%
% @retval I the reconstructed input maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I] = ipp_recon(z0,z0_filter_size,z,F,C,TRAIN_Z0,COMP_THREADS)
% Note this assumes that not thresholding of the variance has been done.

num_input_maps = size(F,3);
filter_size = size(F,1);
xdim = size(z,1)-filter_size+1;
ydim = size(z,2)-filter_size+1;
% C = logical(C);
C = single(C);

if(TRAIN_Z0)  % The z0 * F0 convoltuions (returns one per input map).
   I = ipp_conv2(z0,ones(z0_filter_size,z0_filter_size,'single')/single(z0_filter_size),'valid',COMP_THREADS);
else
    I = zeros(xdim,ydim,num_input_maps);
end
    
% % For each input map
% for j=1:num_input_maps
%     % The sum over k of zk * Fjk convolutions.
%     I(:,:,j) = sum(ipp_conv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3);
% end
% Does the sum over feature maps of zk * Fjk plust the z0 contribution.
I = sum(valid_eachK_loopJ(z,F,C,COMP_THREADS),4) + I;

end
