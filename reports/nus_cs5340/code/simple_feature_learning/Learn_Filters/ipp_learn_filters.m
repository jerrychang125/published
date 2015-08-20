%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the filters based on a single training sample (image) using
% the IPP libraries (and thus is fast). This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @filters_file @copybrief ipp_learn_filters.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief ipp_learn_filters.m
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
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
% @param COMP_THREADS the number of threads to split computation over.
%
% @retval F the updated filters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [F] = ipp_learn_filters(max_it,z,y,F,z0,z0_filter_size,lambda,C,TRAIN_Z0,COMP_THREADS)

sizeF = size(F);

% Get the number of ks.
% num_feature_maps = size(F,4);
% num_input_maps = size(F,3);
% xdim = size(y,1);
% ydim = size(y,2);

% Initialize variable for the results.
z0_filter = ones(z0_filter_size,z0_filter_size,'single')/single(z0_filter_size);
zflip = flipdim(flipdim(z,1),2);
% C = logical(C); % Make the connectivity matrix for indexing.
C = single(C);

%%%%%%%%%%
%%Compute the right hand side (A'b) term
% Do the f'y convolutions.
% Atb = zeros(size(F),'single');
if(TRAIN_Z0) % If using z0 maps, must convolve with the z0conv
    % Convolve z0 map for j with it's filter.
    %     z0conv = ipp_conv2(z0,z0_filter,'valid');
    %     for j=1:num_input_maps
    %         Atb(:,:,j,C(j,:)) = ipp_conv2(zflip(:,:,C(j,:)),y(:,:,j),'valid',COMP_THREADS) -...
    %             ipp_conv2(zflip(:,:,C(j,:)),z0conv(:,:,j),'valid',COMP_THREADS);
    %     end
    Atb = valid_loopK_loopJ(zflip,y,C,COMP_THREADS) - ...
        valid_loopK_loopJ(zflip,ipp_conv2(z0,z0_filter,'valid'),C,COMP_THREADS);
else
    %     for j=1:num_input_maps
    %         % Place in correct location so when conctemp(:) is used below it will be
    %         % the correct vectorized form for dfz.
    %         Atb(:,:,j,C(j,:)) = ipp_conv2(zflip(:,:,C(j,:)),y(:,:,j),'valid',COMP_THREADS);
    %     end
    Atb = valid_loopK_loopJ(zflip,y,C,COMP_THREADS);
end
% This is the RHS. Only comput this once.
% Atb = lambda*Atb(:) + (kappa/2)*alphaF*((sign(F(:))).*abs(F(:)).^(alphaF-1));
Atb = lambda*Atb(:);
%%%%%%%%%%

%%%%%%%%%%
%%Compute the left hand side (A'Ax) term
% Loop over each input plane.
% AtAx = zeros(size(F),'single');
% for j=1:num_input_maps
%     % The outer convolution is the flipped z* with inner convoltuion.
%     % The inner convolution is the sume over k of zk * Fjk.
%     AtAx(:,:,j,C(j,:)) = ipp_conv2(zflip(:,:,C(j,:)),...
%         sum(ipp_conv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3),'valid',COMP_THREADS);
% end
AtAx = valid_loopK_loopJ(zflip,sum(valid_eachK_loopJ(z,F,C,COMP_THREADS),4),C,COMP_THREADS);
% This is the left hand side.
AtAx = lambda*AtAx(:);
%%%%%%%%%%


% Compute the residual.
r = Atb - AtAx;

for iter = 1:max_it
    %     rho = (r(:)'*r(:));
    rho = norm(r(:))^2;
    
    if ( iter > 1 ),                       % direction vector
        their_beta = rho / rho_1;
        p(:) = r(:) + their_beta*p(:);
    else
        p = r;
        p = reshape(p,sizeF);
    end
    
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term
    % Loop over each input map
    %     q = zeros(size(F),'single');
    %     for j=1:num_input_maps
    %         % The outer convolution is the flipped z* with inner convoltuion.
    %         % The inner convolution is the sume over k of zk * Fjk.
    %         q(:,:,j,C(j,:)) = ipp_conv2(zflip(:,:,C(j,:)),...
    %             sum(ipp_conv2(z(:,:,C(j,:)),squeeze(p(:,:,j,C(j,:))),'valid',COMP_THREADS),3),'valid',COMP_THREADS);
    %     end
    q = valid_loopK_loopJ(zflip,sum(valid_eachK_loopJ(z,p,C,COMP_THREADS),4),C,COMP_THREADS);
    % This is the left hand side.
    q = lambda*q(:);
    
    %      p = p(:);
    temp = sum(p(:).*q(:));
    %     their_alpha = double(abs(temp) > 1e-9).*rho / temp;
    their_alpha = rho / temp;
    F(:) = F(:) + their_alpha * p(:);                    % update approximation vector
    
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
%                          fprintf('\nIteration %d |residual| %.3g', iter, norm(r));
    if(norm(r) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end