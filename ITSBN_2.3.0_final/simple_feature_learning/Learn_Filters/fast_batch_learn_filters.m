%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the filters based on a batch of training samples (images) without using
% the IPP libraries (and thus is SLOW). This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
% Note: this can take advantage of convnfft.m from MATLABCentral if you have
% downloaded that and it is on your path (this is MUCH faster).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inference_file @copybrief fast_batch_learn_filters.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief fast_batch_learn_filters.m
%
% @param max_it number of conjugate gradient iterations
% @param zbatch the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps x batch_size).
% @param ybatch the input maps for the layer (xdim x ydim x num_input_maps x batch_size).
% @param F the filters (Fxdim x Fydim x num_input_maps x num_feature_maps).
% @param z0batch the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps x batch_size).
% @param z0_filter_size the size of the z0 filters (if used).
% @param lambda the coefficient on the reconstruction error term.
% @param C the connectivity matrix for the layer.
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
%
% @retval F the updated filters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [F] = fast_batch_learn_filters(max_it,zbatch,ybatch,F,z0batch,z0_filter_size,lambda,C,TRAIN_Z0)

sizeF = size(F);

% Get the number of ks.
num_feature_maps = size(F,4);
num_input_maps = size(F,3);
xdim = size(ybatch,1);
ydim = size(ybatch,2);

% Keeps the running total of the gradient updates.
summed_F = zeros(size(F),'single');

% If you have the MatlabCentral convnfft.m file, then this is used.
if(exist('convnfft','file'))
    USE_FFT = 1;
else
    USE_FFT = 0;
end



% Just simply loop over all the images in the batch.
for image=1:size(zbatch,4)
    z = zbatch(:,:,:,image);
    y = ybatch(:,:,:,image);
    z0 = z0batch(:,:,:,image);
    
    % Initialize variable for the results.
    conctemp = zeros(size(F));
    
    %%%%%%%%%%
    %%Compute the right hand side (A'b) term
    % Do the f'y convolutions.
    for j=1:num_input_maps
        if(TRAIN_Z0) % Convolve z0 map for j with it's filter.
            z0conv = conv2(z0(:,:,j),ones(z0_filter_size,z0_filter_size)/z0_filter_size,'valid');
        end
        for k = 1:num_feature_maps
            if(C(j,k)==1)
                if(TRAIN_Z0) % If using z0 maps, must convolve with the z0conv
                    if(USE_FFT)
                        conctemp(:,:,j,k) = convnfft(flipud(fliplr(z(:,:,k))),y(:,:,j),'valid') -...
                            conv2(flipud(fliplr(z(:,:,k))),z0conv,'valid');
                    else
                        conctemp(:,:,j,k) = conv2(flipud(fliplr(z(:,:,k))),y(:,:,j),'valid') -...
                            conv2(flipud(fliplr(z(:,:,k))),z0conv,'valid');
                    end
                else
                    % Place in correct location so when conctemp(:) is used below it will be
                    % the correct vectorized form for dfz.
                    if(USE_FFT)
                        conctemp(:,:,j,k) = convnfft(flipud(fliplr(z(:,:,k))),y(:,:,j),'valid');
                    else
                        conctemp(:,:,j,k) = conv2(flipud(fliplr(z(:,:,k))),y(:,:,j),'valid');
                    end
                end
            end
        end
    end
    % This is the RHS. Only comput this once.
    % Atb = lambda*conctemp(:) + (kappa/2)*alphaF*((sign(F(:))).*abs(F(:)).^(alphaF-1));
    Atb = lambda*conctemp(:);
    %%%%%%%%%%
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term
    % Loop over each input plane.
    conctemp = zeros(size(F));
    
    for j=1:num_input_maps
        % Initialize a variable to keep the running some of the other convolutions
        % between f*z.
        convsum = zeros(xdim,ydim);
        % Loop over all the other ks and compute the sume of their
        % convolutions (f*z). This is the Ax term.
        for k = 1:num_feature_maps
            if(C(j,k)==1)
                % Convolve F k with z feature map and comput runnign sum.
                convsum = convsum + conv2(z(:,:,k),F(:,:,j,k),'valid');
            end
        end
        
        % This is the A'Ax term.
        for k = 1:num_feature_maps
            if(C(j,k)==1)
                % Place in correct location so when conctemp(:) is used below it will be
                % the correct vectorized form for dfz.
                if(USE_FFT)
                    conctemp(:,:,j,k) = convnfft(flipud(fliplr(z(:,:,k))),convsum,'valid');
                else
                    conctemp(:,:,j,k) = conv2(flipud(fliplr(z(:,:,k))),convsum,'valid');
                end
            end
        end
    end
    
    % This is the left hand side.
    AtAx = lambda*conctemp(:);
    %%%%%%%%%%
    
    
    % Compute the residual.
    r = Atb - AtAx;
    
    for iter = 1:max_it
        rho = (r(:)'*r(:));
        
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
        conctemp = zeros(size(F));
        for j=1:num_input_maps
            % Initialize a variable to keep the running some of the other convolutions
            % between f*z.
            convsum = zeros(xdim,ydim);
            % Loop over all the other ks and compute the sume of their
            % convolutions (f*z). This is the Ax term.
            for k = 1:num_feature_maps
                if(C(j,k)==1)
                    % Convolve F k with z feature map and comput runnign sum.
                    convsum = convsum + conv2(z(:,:,k),p(:,:,j,k),'valid');
                end
            end
            
            % This is the A'Ax term.
            for k = 1:num_feature_maps
                if(C(j,k)==1)
                    % Place in correct location so when conctemp(:) is used below it will be
                    % the correct vectorized form for dfz.
                    if(USE_FFT)
                        conctemp(:,:,j,k) = convnfft(flipud(fliplr(z(:,:,k))),convsum,'valid');
                    else
                        conctemp(:,:,j,k) = conv2(flipud(fliplr(z(:,:,k))),convsum,'valid');
                    end
                end
            end
        end
        % This is the left hand side.
        q = lambda*conctemp(:);
        
        temp = sum(p(:).*q(:));
        their_alpha = rho / temp;
        summed_F(:) = summed_F(:) + their_alpha * p(:);                    % update approximation vector
        
        r = r - their_alpha*q;                      % compute residual
        
        rho_1 = rho;
        %                  fprintf('\nIteration %d |residual| %.3g', iter, norm(r));
        if(norm(r) < 1e-6)
            %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
            break;
        end;
    end
end
% Update by the average update.
F(:) = F(:) + summed_F(:)/(size(zbatch,4)*max_it);

end