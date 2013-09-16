%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the feature maps for a single training sample (image) using
% the GPUmat libraries (and thus is FAST). This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inference_file @copybrief gpu_infer.m
% @gpu_file @copybrief gpu_infer.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief gpu_infer.m
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
function [z] = gpu_infer(max_it,z,w,y,F,z0,z0_filter_size,lambda,beta,C,TRAIN_Z0,COMP_THREADS)

% Get the number of ks.
% num_feature_maps = size(F,4);
num_input_maps = size(F,3);
% filter_size = size(F,1);
% xdim = size(y,1);
% ydim = size(y,2);

% siny = single(y);
% sinz = single(z);
% sinF = single(F);
% sinFflip = flipdim(flipdim(sinF,1),2);
% sinC = logical(single(C));

% Flipped version of F.
Fflip = flipdim(flipdim(F,1),2);
% Same as flipping like above.
% Fflip = slice(slice(F,[END -1 1],':',':',':'),':',[END -1 1],':',':');
% newFflip = reshape(Fflip,size(Fflip,1)*size(Fflip,2),size(Fflip,3)*size(Fflip,4));


%%%%%%%%%%%%%%%

% Initialize the running sum for each feature map.
z0_filter = ones(z0_filter_size,z0_filter_size,GPUsingle)/single(z0_filter_size);
% C = GPUsingle(C);

% numOutputsX = xdim - filter_size + 1;
% numOutputs = numOutputsX*numOutputsX;
% targets = zeros(num_feature_maps*numOutputs,1, GPUsingle); %note this is

%%%%%%%%%%
%%Compute the right hand side (A'b) term
Atb = zeros(size(z),GPUsingle);
% sinAtb = single(Atb);
% Do the f'y convolutions.
if(TRAIN_Z0) % Also convolve flipped Fjk with z0 maps convolution
    % Convolve z0 map for each j with it's filter.
    z0conv = mycuConv2(z0,z0_filter,'valid');
    for j=1:num_input_maps
        % For all feature maps, convolve with input map j and z0's
        % convolution.
        Atb(:,:,C(j,:)) = Atb(:,:,C(j,:)) +...
            mycuConv2(y(:,:,j),squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS) -...
            mycuConv2(z0conv(:,:,j),squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
    end
else
%     'CPU edition'
%     tic
%     for j=1:num_input_maps
%         % For all feature maps, convolve with inptu map j.
%         sinAtb(:,:,sinC(j,:)) = sinAtb(:,:,sinC(j,:)) +...
%             ipp_conv2(siny(:,:,j),squeeze(sinFflip(:,:,j,sinC(j,:))),'full',COMP_THREADS);
%     end
%     t=toc
%     'GPUedition'
%     tic
    for j=1:num_input_maps
        idx = GPUsingle(find(single(C(j,:))));
        % For all feature maps, convolve with inptu map j.
        Atb(:,:,idx) = Atb(:,:,idx) + gpu_conv2(y(:,:,j),Fflip(:,:,j,idx),'full');
    end
%     t=toc
    
    %         idx = GPUsingle(find(single(C)));
    %     size(Atb)
    %
    %     Atb = gpu_conv2(y,Fflip(:,:,C),'full');
    %     sizeAtb = size(Atb);
    %     selector = GPUsingle(repmat(GPUsingle(eye(num_input_maps,num_input_maps)),num_feature_maps,1));
    %     Atb = Atb(:,:,selector);
    %     Atb = reshape(Atb,sizeAtb);
    %     'blah'
    %     size(Atb)
    %     Atb = sum(Atb,4);
    %     'blah2'
    %     size(Atb)
    %     Atb = reshape(Atb,size(Atb,1),size(Atb,2),num_feature_maps,num_input_maps);
    %     Atb = sum(Atb,4);
    % %     Atb = reshape(Atb,size(Atb,1),size(Atb,2),num_feature_maps,num_input_maps,size(Atb,4));
    % %     Atb = Atb(:,:,GPUsingle(eye(num_input_maps,num_input_maps)),:);
    % % %     Atb = sum(Atb,5);
    % %     Atb = sum(Atb,4);
    % %     Atb = Atb(:,:,C);
    %     size(C)
    % %     Atb(1:2,1:2,:,:);
    %     size(Atb)
    %         t=toc
    
%     fprintf('Error: %f\n\n\n',max(sinAtb(:)-single(Atb(:))))
    
    
end
% This is the RHS. Only comput this once.
Atb = lambda*Atb(:) + beta*w(:);
%%%%%%%%%%



%%%%%%%%%%
%%Compute the left hand side (A'Ax) term
% Initialize the running sum for each feature map.
AtAx = zeros(size(z),GPUsingle);
% sinAtAx = single(AtAx);
% 'CPU edition'
% tic
% for j=1:num_input_maps
%     % This is the sum over k feature maps of the zk*Fjk convolutions as the first term of teh second convolution.
%     % The second convolution is the convolution of the above with F* summed over the input maps.
%     %temp1 = sum(mycuConv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3);
%     %AtAX(:,:,C(j,:)) = AtAx(:,:,C(j,:)) + mycuConv2(temp1,squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
%     
%     sinAtAx(:,:,sinC(j,:)) = sinAtAx(:,:,sinC(j,:)) + ipp_conv2(...
%         sum(ipp_conv2(sinz(:,:,sinC(j,:)),squeeze(sinF(:,:,j,sinC(j,:))),'valid',COMP_THREADS),3),...
%         squeeze(sinFflip(:,:,j,sinC(j,:))),'full',COMP_THREADS);
%     
%     
%     
% end
% t=toc
% 'GPU edition'
% tic

for j=1:num_input_maps
    % This is the sum over k feature maps of the zk*Fjk convolutions as the first term of teh second convolution.
    % The second convolution is the convolution of the above with F* summed over the input maps.
    %temp1 = sum(mycuConv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3);
    %AtAX(:,:,C(j,:)) = AtAx(:,:,C(j,:)) + mycuConv2(temp1,squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
    idx = GPUsingle(find(single(C(j,:))));
    AtAx(:,:,idx) = AtAx(:,:,idx) + gpu_conv2(...
        sum(gpu_conv2(z(:,:,idx),F(:,:,j,idx),'valid'),3),...
        Fflip(:,:,j,idx),'full');
end
% t=toc
% 
% fprintf('Error: %f\n\n\n',max(sinAtAx(:)-single(AtAx(:))))

% This is the left hand side.
AtAx = lambda*AtAx(:)+beta*z(:);
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
        p = reshape(p,size(z));
    end
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term
    % Initialize the running sum for each feature map.
    q = zeros(size(z),GPUsingle);
    %     for j=1:num_input_maps
    %         % This is the sum over k feature maps of the zk*Fjk convolutions as the first term of teh second convolution.
    %         % The second convolution is the convolution of the above with F* summed over the input maps.
    %         q(:,:,C(j,:)) = q(:,:,C(j,:)) + mycuConv2(...
    %             sum(mycuConv2(p(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3),...
    %             squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
    %     end
    
    for j=1:num_input_maps
        % This is the sum over k feature maps of the zk*Fjk convolutions as the first term of teh second convolution.
        % The second convolution is the convolution of the above with F* summed over the input maps.
        %temp1 = sum(mycuConv2(z(:,:,C(j,:)),squeeze(F(:,:,j,C(j,:))),'valid',COMP_THREADS),3);
        %AtAX(:,:,C(j,:)) = AtAx(:,:,C(j,:)) + mycuConv2(temp1,squeeze(Fflip(:,:,j,C(j,:))),'full',COMP_THREADS);
        idx = GPUsingle(find(single(C(j,:))));
        q(:,:,idx) = q(:,:,idx) + gpu_conv2(...
            sum(gpu_conv2(p(:,:,idx),F(:,:,j,idx),'valid'),3),...
            Fflip(:,:,j,idx),'full');
    end
    
    % This is the left hand side.
    q = lambda*q(:)+beta*p(:);
    %%%%%%%%%%
    
    %     their_alpha = rho / (p(:)'*q(:) );
    temp = p(:)'*q(:);
    their_alpha = double(abs(temp) > 1e-9).*rho / temp;
    z(:) = z(:) + their_alpha * p(:);           % update approximation vector
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
    %     fprintf('\nIteration %d |residual| %.3g', iter, norm(r));
    if(sqrt(sum(abs(r(:)).^2)) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end