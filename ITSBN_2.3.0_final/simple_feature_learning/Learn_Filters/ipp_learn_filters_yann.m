%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the filters based on a single training sample (image) using
% the IPP libraries (and thus is fast) for both layers jointly. This is done via
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
% @param y the input maps for the layer (xdim x ydim x num_input_maps).
% @param F1 the filters (Fxdim x Fydim x num_input_maps x num_feature_maps) for layer 1.
% @param F2 the filters (Fxdim x Fydim x num_input_maps x num_feature_maps) for layer 2.
% @param z0 the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps).
% @param z0_filter_size the size of the z0 filters (if used).
% @param lambda the coefficient on the reconstruction error term.
% @param C1 the connectivity matrix for layer 1.
% @param C2 the connectivity matrix for layer 2.
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
% @param COMP_THREADS the number of threads to split computation over.
%
% @retval F1 the updated filters for layer 1.
% @retval F2 the updated filters for layer 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [F1,F2] = ipp_learn_filters_yann(max_it,z1,z2,y,F1,F2,z0,z0_filter_size,lambda,C1,C2,TRAIN_Z0,COMP_THREADS)

% Author: Matthew Zeiler
% Returns: Update ks, F for a single traiing sample (image).
% Purpose: Save as fast_learn_filters.m but with IPP convolutions.
% Conducts conjugate gradient to update the ks.
% This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).

% The following are implemented below, just not in the rest of the code
% yet.
lambda1 = lambda(1);
lambda2 = lambda(2);
% beta1 = beta;
% beta2 = beta;

% sizeF = size(F);

% Get the number of ks.
% num_feature_maps = size(F,4);
% num_input_maps = size(F,3);
% xdim = size(y,1);
% ydim = size(y,2);

% Initialize variable for the results.
z0_filter = ones(z0_filter_size,z0_filter_size,'single')/single(z0_filter_size);
zflip1 = flipdim(flipdim(z1,1),2);
zflip2 = flipdim(flipdim(z2,1),2);
C1 = single(C1);
C2 = single(C2);

%%%%%%%%%%
%%Compute the right hand side (A'b) term for layer 1
if(TRAIN_Z0) % If using z0 maps, must convolve with the z0conv
    Atb1 = valid_loopK_loopJ(zflip1,y,C1,COMP_THREADS) - ...
        valid_loopK_loopJ(zflip1,ipp_conv2(z0,z0_filter,'valid'),C1,COMP_THREADS);
else
    Atb1 = valid_loopK_loopJ(zflip1,y,C1,COMP_THREADS);
end
% This is the RHS. Only comput this once.
% Atb1 = lambda1*Atb1(:) + (kappa/2)*alphaF*((sign(F1(:))).*abs(F1(:)).^(alphaF-1));
Atb1 = lambda1*Atb1(:);
%%%%%%%%%%
%%%%%%%%%%
%%Compute the right hand side (A'b) term for layer 2
Atb2 = valid_loopK_loopJ(zflip2,z1,C2,COMP_THREADS);
% This is the RHS. Only comput this once.
% Atb2 = lambda2*Atb2(:) + (kappa/2)*alphaF*((sign(F2(:))).*abs(F2(:)).^(alphaF-1));
Atb2 = lambda2*Atb2(:);
%%%%%%%%%%


%%%%%%%%%%
%%Compute the left hand side (A'Ax) term for layer 1
AtAx1 = valid_loopK_loopJ(zflip1,sum(valid_eachK_loopJ(z1,F1,C1,COMP_THREADS),4),C1,COMP_THREADS);
% This is the left hand side.
AtAx1 = lambda1*AtAx1(:);
%%%%%%%%%%
%%%%%%%%%%
%%Compute the left hand side (A'Ax) term for layer 2
AtAx2 = valid_loopK_loopJ(zflip2,sum(valid_eachK_loopJ(z2,F2,C2,COMP_THREADS),4),C2,COMP_THREADS);
% This is the left hand side.
AtAx2 = lambda2*AtAx2(:);
%%%%%%%%%%

% Compute the residual.
r = [Atb1;Atb2] - [AtAx1;AtAx2];

size1 = length(Atb1);

for iter = 1:max_it
    rho = norm(r(:))^2;
    
    if ( iter > 1 ),                       % direction vector
        their_beta = rho / rho_1;
        p = r(:) + their_beta*p;
    else
        p = r(:);
        %         p = reshape(p,sizeF);
    end
    p1 = reshape(p(1:size1),size(F1));
    p2 = reshape(p(size1+1:end),size(F2));
    
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term for layer 1
    q1 = valid_loopK_loopJ(zflip1,sum(valid_eachK_loopJ(z1,p1,C1,COMP_THREADS),4),C1,COMP_THREADS);
    % This is the left hand side.
    q1 = lambda1*q1(:);
    %%%%%%%%%%
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term for layer 2
    q2 = valid_loopK_loopJ(zflip2,sum(valid_eachK_loopJ(z2,p2,C2,COMP_THREADS),4),C2,COMP_THREADS);
    % This is the left hand side.
    q2 = lambda2*q2(:);
    %%%%%%%%%%
    q = [q1;q2];
    
    %      p = p(:);
    temp = sum(p.*q(:));
    %     their_alpha = double(abs(temp) > 1e-9).*rho / temp;
    their_alpha = rho / temp;
    F1(:) = F1(:) + their_alpha * p(1:size1);                    % update approximation vector
    F2(:) = F2(:) + their_alpha * p(size1+1:end);                    % update approximation vector
    
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
%                          fprintf('\nLearn Filters Iteration %d |residual| %.3g', iter, norm(r));
    if(norm(r) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end