%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the z0 feature maps for a single training sample (image) using
% the IPP libraries (and thus is fast). This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inference_file @copybrief ipp_z0.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief ipp_z0.m
%
% @param max_it number of conjugate gradient iterations
% @param z the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps).
% @param y the input maps for the layer (xdim x ydim x num_input_maps).
% @param F the filters (Fxdim x Fydim x num_input_maps x num_feature_maps).
% @param z0 the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps).
% @param z0_filter_size the size of the z0 filters (if used).
% @param lambda the coefficient on the reconstruction error term.
% @param C the connectivity matrix for the layer.
% @param psi the coefficient on the z0 laplacian regularization term.
% @param COMP_THREADS the number of threads to split computation over.
%
% @retval z0 the updated z0 feature maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z0] = ipp_z0(max_it,z,y,F,z0,z0_filter_size,lambda,C,psi,COMP_THREADS)

% Get the number of ks.
% num_feature_maps = size(F,4);
% num_input_maps = size(F,3);
% xdim = size(y,1);
% ydim = size(y,2);
% C = logical(C);
C = single(C);

% The z0 filters used to convolve with each of the z0 maps.
z0_filter = ones(z0_filter_size,z0_filter_size,'single')/single(z0_filter_size);

%%%%%%%%%%
%%Compute the right hand side (A'b) term
% Do the f'y convolutions.
% Atb = zeros(size(z0));
% for j=1:num_input_maps
%     % The first term is z0*F0 The second term is sum over k of zk*Fjk then convolvd with z0.
%     Atb(:,:,j) = lambda*(ipp_conv2(y(:,:,j),z0_filter,'full',COMP_THREADS) - ...
%         ipp_conv2(sum(ipp_conv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3),...
%         z0_filter,'full',COMP_THREADS));
% end
Atb = lambda*(ipp_conv2(y,z0_filter,'full',COMP_THREADS) - ...
    ipp_conv2(sum(valid_eachK_loopJ(z,F,C,COMP_THREADS),4),z0_filter,'full',COMP_THREADS));
% This is the RHS. Only comput this once.
% This is f0' * y
Atb = Atb(:);
%%%%%%%%%%



%%%%%%%%%%
%%Compute the left hand side (A'Ax) term
% The remaining convolution term on the LHS (just involves the given j
% with no summation.
AtAx = lambda*ipp_conv2(ipp_conv2(z0,z0_filter,'valid',COMP_THREADS),z0_filter,'full',COMP_THREADS) + ...
    psi*ipp_conv2(ipp_conv2(z0,single([1 -1]),'valid',COMP_THREADS),flipud(fliplr(single([1 -1]))),'full',COMP_THREADS) + ...
    psi*ipp_conv2(ipp_conv2(z0,single([1 -1]'),'valid',COMP_THREADS),flipud(fliplr(single([1 -1]'))),'full',COMP_THREADS);
% This is the left hand side.
AtAx = AtAx(:);
%%%%%%%%%%


% Compute the residual.
r = Atb - AtAx;

for iter = 1:max_it
    %     rho = (r(:)'*r(:));
    rho = norm(r(:))^2;
    
    if ( iter > 1 ),                       % direction vector
        their_beta = rho / rho_1;
        %         their_beta = double(abs(rho_1) > 1e-9).*rho / rho_1;  % Added from dilips.m
        p(:) = r(:) + their_beta*p(:);
    else
        p = r;
        p = reshape(p,size(z0));
%         q = zeros(size(z0));
    end
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term
    % Initialize the running sum for each feature map.
    % Initialize the running sum for each feature map.
    % The remaining convolution term on the LHS (just involves the given j
    % with no summation.
    q = lambda*ipp_conv2(ipp_conv2(p,z0_filter,'valid',COMP_THREADS),z0_filter,'full',COMP_THREADS) + ...
        psi*ipp_conv2(ipp_conv2(p,single([1 -1]),'valid',COMP_THREADS),flipud(fliplr(single([1 -1]))),'full',COMP_THREADS) + ...
        psi*ipp_conv2(ipp_conv2(p,single([1 -1]'),'valid',COMP_THREADS),flipud(fliplr(single([1 -1]'))),'full',COMP_THREADS);
    
    % This is the left hand side.
    q = q(:);
    %%%%%%%%%%
    
    %     %     their_alpha = rho / (p(:)'*q(:) );
    %     temp = p(:)'*q(:);
    temp = sum(p(:).*q(:));
    %     their_alpha = double(abs(temp) > 1e-9).*rho / temp;
    their_alpha = rho / temp;
    
    z0(:) = z0(:) + their_alpha * p(:);           % update approximation vector
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
    %         fprintf('Iteration %d |residual| %.3g\n', iter, norm(r));
    if(norm(r) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end