%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the feature maps for a single training sample (image) using
% the IPP libraries (and thus is fast). This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inference_file @copybrief ipp_infer.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief ipp_infer.m
%
% @param max_it number of conjugate gradient iterations
% @param z the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps).
% @param w the auxilary variable (same size as z).
% @param y the input maps for the layer (xdim x ydim x num_input_maps).
% @param F the filters (Fxdim x Fydim x num_input_maps x num_feature_maps).
% @param z0 the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps).
% @param z0_filter_size the size of the z0 filters (if used).
% @param lambda the coefficient on the reconstruction error term.
% @param beta the continuation variable on the ||z-x|| term.
% @param C the connectivity matrix for the layer.
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
% @param COMP_THREADS the number of threads to split computation over.
%
% @retval z the updated feature maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z] = ipp_infer(max_it,z,w,y,F,z0,z0_filter_size,lambda,beta,C,TRAIN_Z0,COMP_THREADS)

% Get the number of ks.
% num_feature_maps = size(F,4);
% num_input_maps = size(F,3);
% xdim = size(y,1);
% ydim = size(y,2);

% Initialize the running sum for each feature map.
z0_filter = ones(z0_filter_size,z0_filter_size,'single')/single(z0_filter_size);
% C = logical(C);
C = single(C);

% Flipped version of F.
Fflip = flipdim(flipdim(F,1),2);

%%%%%%%%%%
%%Compute the right hand side (A'b) term
% Atb = zeros(size(z),'single');
% Do the f'y convolutions.
if(TRAIN_Z0) % Also convolve flipped Fjk with z0 maps convolution
    % Convolve z0 map for each j with it's filter.
    z0conv = ipp_conv2(z0,z0_filter,'valid');
    %     for j=1:num_input_maps
    %         % For all feature maps, convolve with input map j and z0's
    %         % convolution.
    %         Atb(:,:,C(j,:)) = Atb(:,:,C(j,:)) +...
    %             ipp_conv2(y(:,:,j),squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS) -...
    %             ipp_conv2(z0conv(:,:,j),squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
    %     end
    Atb = sum(full_eachJ_loopK(y,Fflip,C,COMP_THREADS),3) - ...
        sum(full_eachJ_loopK(z0conv,Fflip,C,COMP_THREADS),3);
    
else
    %     for j=1:num_input_maps
    %         % For all feature maps, convolve with inptu map j.
    %         Atb(:,:,C(j,:)) = Atb(:,:,C(j,:)) +...
    %             ipp_conv2(y(:,:,j),squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
    %     end
    Atb = sum(full_eachJ_loopK(y,Fflip,C,COMP_THREADS),3);
    
end
% This is the RHS. Only comput this once.
Atb = lambda*Atb(:) + beta*w(:);
%%%%%%%%%%


% %%%%%%%%%%
% %%Compute the left hand side (A'Ax) term
% % Initialize the running sum for each feature map.
% AtAx = zeros(size(z),'single');
% for j=1:num_input_maps
%     % This is the sum over k feature maps of the zk*Fjk convolutions as the first term of teh second convolution.
%     % The second convolution is the convolution of the above with F* summed over the input maps.
%     %temp1 = sum(ipp_conv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3);
%     %AtAX(:,:,C(j,:)) = AtAx(:,:,C(j,:)) + ipp_conv2(temp1,squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
%     AtAx(:,:,C(j,:)) = AtAx(:,:,C(j,:)) + ipp_conv2(...
%         sum(ipp_conv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3),...
%         squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
% end
AtAx = sum(full_eachJ_loopK(sum(valid_eachK_loopJ(z,F,C,COMP_THREADS),4),Fflip,C,COMP_THREADS),3);
% % This is the left hand side.
AtAx = lambda*AtAx(:)+beta*z(:);
% %%%%%%%%%%


% Compute the residual.
r = Atb - AtAx;

for iter = 1:max_it
    %         rho = dot(r(:),r(:));
    
    %     rho = sum(r(:).^2); % This saves memory.
    %         rho = (r(:)'*r(:));
    rho =norm(r(:))^2; % This is the fastest
    
    
    if ( iter > 1 ),                       % direction vector
        their_beta = rho / rho_1;
        %         their_beta = double(abs(rho_1) > 1e-9).*rho / rho_1;  % Added from dilips.m
        p(:) = r(:) + their_beta*p(:);
    else
        p = r;
        p = reshape(p,size(z));
    end
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term
    % Initialize the running sum for each feature map.
    %     q = zeros(size(z),'single');
    %     for j=1:num_input_maps
    %         % This is the sum over k feature maps of the zk*Fjk convolutions as the first term of teh second convolution.
    %         % The second convolution is the convolution of the above with F* summed over the input maps.
    %         q(:,:,C(j,:)) = q(:,:,C(j,:)) + ipp_conv2(...
    %             sum(ipp_conv2(p(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3),...
    %             squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
    %     end
    q = sum(full_eachJ_loopK(sum(valid_eachK_loopJ(p,F,C,COMP_THREADS),4),Fflip,C,COMP_THREADS),3);
    % This is the left hand side.
    q = lambda*q(:)+beta*p(:);
    %%%%%%%%%%
    
    %     their_alpha = rho / (p(:)'*q(:) );
    temp = sum(p(:).*q(:));
    %     temp = p(:)'*q(:);
    %     their_alpha = double(abs(temp) > 1e-9).*rho / temp;
    their_alpha = rho / temp;
    
    z(:) = z(:) + their_alpha * p(:);           % update approximation vector
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
%         fprintf('\nIteration %d |residual| %.3g', iter, norm(r));
    if(norm(r) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end