%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the feature maps for a single training sample (image) using
% the IPP libraries (and thus is fast) jointly for both layers. This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @filters_file @copybrief ipp_learn_filters_yann.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief ipp_learn_filters_yann.m
%
% @param max_it number of conjugate gradient iterations
% @param z1 the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps) for layer 1.
% @param z2 the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps) for layer 2.
% @param w1 the auxilary variable (same size as z1) for layer 1.
% @param w2 the auxilary variable (same size as z2) for layer 2.
% @param y the input maps for the layer (xdim x ydim x num_input_maps).
% @param F1 the filters (Fxdim x Fydim x num_input_maps x num_feature_maps) for layer 1.
% @param F2 the filters (Fxdim x Fydim x num_input_maps x num_feature_maps) for layer 2.
% @param z0 the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps).
% @param z0_filter_size the size of the z0 filters (if used).
% @param lambda the coefficient on the reconstruction error term.
% @param beta the continuation variable on the ||z-x|| term (same for both layers).
% @param C1 the connectivity matrix for layer 1.
% @param C2 the connectivity matrix for layer 2.
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
% @param COMP_THREADS the number of threads to split computation over.
%
% @retval z1 the updated feature maps for layer 1.
% @retval z2 the updated feature maps for layer 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z1,z2] = ipp_infer_yann(max_it,z1,z2,w1,w2,y,F1,F2,z0,z0_filter_size,lambda,beta,C1,C2,TRAIN_Z0,COMP_THREADS)


% The following are implemented below, just not in the rest of the code
% yet.
lambda1 = lambda(1);
lambda2 = lambda(2);
beta1 = beta;
beta2 = beta;

% Get the number of ks.
% num_feature_maps = size(F,4);
% num_input_maps = size(F,3);
% xdim = size(y,1);
% ydim = size(y,2);

% Initialize the running sum for each feature map.
z0_filter = ones(z0_filter_size,z0_filter_size,'single')/single(z0_filter_size);
C1 = single(C1);
C2 = single(C2);

% Flipped version of F.
Fflip1 = flipdim(flipdim(F1,1),2);
Fflip2 = flipdim(flipdim(F2,1),2);
%%%%%%%%%%
%%Compute the right hand side (A'b) term for Layer 1
% Do the f'y convolutions.
if(TRAIN_Z0) % Also convolve flipped Fjk with z0 maps convolution
    % Convolve z0 map for each j with it's filter.
    z0conv = ipp_conv2(z0,z0_filter,'valid');
    Atb1 = sum(full_eachJ_loopK(y,Fflip1,C1,COMP_THREADS),3) - ...
        sum(full_eachJ_loopK(z0conv,Fflip1,C1,COMP_THREADS),3);
    Atb1new = sum(valid_eachK_loopJ(z2,F2,C2,COMP_THREADS),4);
else
    Atb1 = sum(full_eachJ_loopK(y,Fflip1,C1,COMP_THREADS),3);
    Atb1new = sum(valid_eachK_loopJ(z2,F2,C2,COMP_THREADS),4);
end
% This is the RHS. Only comput this once.
Atb1 = lambda1*Atb1(:) + beta1*w1(:) + lambda2*Atb1new(:);
%%%%%%%%%%

%%%%%%%%%%
%%Compute the right hand side (A'b) term for Layer 2
% Do the f'y convolutions.
Atb2 = sum(full_eachJ_loopK(z1,Fflip2,C2,COMP_THREADS),3);
% This is the RHS. Only comput this once.
Atb2 = lambda2*Atb2(:) + beta2*w2(:);
%%%%%%%%%%

% %%%%%%%%%%
% %%Compute the left hand side (A'Ax) term for Layer 1
AtAx1 = sum(full_eachJ_loopK(sum(valid_eachK_loopJ(z1,F1,C1,COMP_THREADS),4),Fflip1,C1,COMP_THREADS),3);
% % This is the left hand side.
AtAx1 = lambda1*AtAx1(:)+(beta1+lambda2)*z1(:);
% %%%%%%%%%%
% %%%%%%%%%%
% %%Compute the left hand side (A'Ax) term for Layer 2
AtAx2 = sum(full_eachJ_loopK(sum(valid_eachK_loopJ(z2,F2,C2,COMP_THREADS),4),Fflip2,C2,COMP_THREADS),3);
% % This is the left hand side.
AtAx2 = lambda2*AtAx2(:)+beta2*z2(:);
% %%%%%%%%%%

% Compute the residual.
r = [Atb1;Atb2] - [AtAx1;AtAx2];

size1 = length(Atb1);

for iter = 1:max_it
    rho =norm(r(:))^2; % This is the fastest
    
    if ( iter > 1 ),                       % direction vector
        their_beta = rho / rho_1;
        p = r(:) + their_beta*p;
    else
        p = r(:);
%         p = reshape(p(1:size1),size(z));
    end
    p1 = reshape(p(1:size1),size(z1));
    p2 = reshape(p(size1+1:end),size(z2));
    
    % %%%%%%%%%%
    % %%Compute the left hand side (A'Ax) term for Layer 1
    q1 = sum(full_eachJ_loopK(sum(valid_eachK_loopJ(p1,F1,C1,COMP_THREADS),4),Fflip1,C1,COMP_THREADS),3);
    % % This is the left hand side.
    q1 = lambda1*q1(:)+(beta1+lambda2)*p1(:);
    % %%%%%%%%%%
    % %%%%%%%%%%
    % %%Compute the left hand side (A'Ax) term for Layer 2
    q2 = sum(full_eachJ_loopK(sum(valid_eachK_loopJ(p2,F2,C2,COMP_THREADS),4),Fflip2,C2,COMP_THREADS),3);
    % % This is the left hand side.
    q2 = lambda2*q2(:)+beta2*p2(:);
    % %%%%%%%%%%
    q = [q1;q2];
        
    temp = sum(p.*q(:));
    their_alpha = rho / temp;
    
    z1(:) = z1(:) + their_alpha * p(1:size1);           % update approximation vector
    z2(:) = z2(:) + their_alpha * p(size1+1:end);           % update approximation vector
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
%         fprintf('\nInfer_yann Iteration %d |residual| %.3g', iter, norm(r));
    if(norm(r) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end