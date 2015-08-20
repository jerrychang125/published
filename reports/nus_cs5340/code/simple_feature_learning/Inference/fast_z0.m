%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the z0 feature maps for a single training sample (image) without using
% the IPP libraries (and thus is SLOW). This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inference_file @copybrief fast_z0.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief fast_z0.m
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
%
% @retval z0 the updated z0 feature maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z0] = fast_z0(max_it,z,y,F,z0,z0_filter_size,lambda,C,psi)

% Get the number of ks.
num_feature_maps = size(F,4);
num_input_maps = size(F,3);
xdim = size(y,1);
ydim = size(y,2);

% Initialize the running sum for each feature map.
Atb = zeros(size(z0));
% The z0 filters used to convolve with each of the z0 maps.
z0_filter = ones(z0_filter_size,z0_filter_size)/z0_filter_size;

%%%%%%%%%%
%%Compute the right hand side (A'b) term
% Do the f'y convolutions.
for j=1:num_input_maps
    % Running sum fot he zk * fkj convoltuions.
    convsum = zeros(xdim,ydim);
    % Move the sum_k (z_k \conv f_k^j) to the RHS (with a minus infront)
    for k=1:num_feature_maps
        if(C(j,k)==1)
            % Compute the sum of k of zk * Fjk
            convsum = convsum + conv2(z(:,:,k),F(:,:,j,k),'valid');
        end
    end
    % Compute the RHS term.
    Atb(:,:,j) = lambda*(conv2(y(:,:,j),z0_filter,'full') - ...
        conv2(convsum,z0_filter,'full'));
end
% This is the RHS. Only comput this once.
% This is f0' * y
Atb = Atb(:);
%%%%%%%%%%



%%%%%%%%%%
%%Compute the left hand side (A'Ax) term
AtAx = zeros(size(z0));

for j=1:num_input_maps
    % The remaining convolution term on the LHS (just involves the given j
    % with no summation.
    AtAx(:,:,j) = lambda*conv2(conv2(z0(:,:,j),z0_filter,'valid'),z0_filter,'full') + ...
        psi*conv2(conv2(z0(:,:,j),[1 -1],'valid'),flipud(fliplr([1 -1])),'full') + ...
        psi*conv2(conv2(z0(:,:,j),[1 -1]','valid'),flipud(fliplr([1 -1]')),'full');
end
% This is the left hand side.
AtAx = AtAx(:);
%%%%%%%%%%


% Compute the residual.
r = Atb - AtAx;

for iter = 1:max_it
    rho = (r(:)'*r(:));
    
    if ( iter > 1 ),                       % direction vector
        %         their_beta = rho / rho_1;
        their_beta = double(abs(rho_1) > 1e-9).*rho / rho_1;  % Added from dilips.m
        p(:) = r(:) + their_beta*p(:);
    else
        p = r;
        p = reshape(p,size(z0));
        q = zeros(size(z0));
    end
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term
    % Initialize the running sum for each feature map.
    % Initialize the running sum for each feature map.
    q = reshape(q,size(z0));
    
    for j=1:num_input_maps
        % The remaining convolution term on the LHS (just involves the given j
        % with no summation.
        q(:,:,j) = lambda*conv2(conv2(p(:,:,j),z0_filter,'valid'),z0_filter,'full') + ...
            psi*conv2(conv2(p(:,:,j),[1 -1],'valid'),flipud(fliplr([1 -1])),'full') + ...
            psi*conv2(conv2(p(:,:,j),[1 -1]','valid'),flipud(fliplr([1 -1]')),'full');
    end
    % This is the left hand side.
    q = q(:);
    %%%%%%%%%%
    
    %     their_alpha = rho / (p(:)'*q(:) );
    temp = p(:)'*q(:);
    their_alpha = double(abs(temp) > 1e-9).*rho / temp;
    z0(:) = z0(:) + their_alpha * p(:);           % update approximation vector
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
    %     fprintf('\nIteration %d |residual| %.3g\n', iter, norm(r));
    if(norm(r) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end