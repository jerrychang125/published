%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Function to solve per-pixel cubic equations of the form:
%  
%%% Full Cubic equation:
% x^3   - 2v * x^2   + v^2 * x  +  m  = 0;
% A        B             C       D 
%
% @file
% @author Dilip Krishnan
% @date Mar 11, 2010
%
% @optimization_file @copybrief cubic_solve_image.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief cubic_solve_image.m
% @param v target values v the size of the feautre maps.
% @param beta the constant beta clamping to the feature maps.
% @retval w the best computed root per-pixel.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w] = cubic_solve_image(v,beta)
 
 RANGE = 0.6;
 STEP  = 0.0001;
 
 %RANGE = 700;
 %STEP = 0.05;
 USE_LOOKUP = 1;
 
 if USE_LOOKUP
%    fprintf('LUTs are ON ');
 else
%    fprintf('LUTs are OFF ');
 end;
 
 if USE_LOOKUP
   persistent lookup_v known_beta xx
   %keyboard 
   ind = find(known_beta==beta);
   if isempty(known_beta)
     xx = [-RANGE:STEP:RANGE];
   end
   
   if any(ind)
     %%% already computed 
     %   w = interp1q(xx',lookup_v(ind,:)',v(:));
     %   w = qinterp1(xx',lookup_v(ind,:)',v(:),1);
     %   w = reshape(w,size(v,1),size(v,2));
     if (maxNumCompThreads > 1) 
       w = pointOp_mt(double(v),lookup_v(ind,:),-RANGE,STEP,0);
     else
       w = pointOp(double(v),lookup_v(ind,:),-RANGE,STEP,0);
     end;
   else
     %%% now go and recompute xx for new value of beta.
     tmp = compute_w(xx,beta);
     lookup_v =  [ lookup_v ; tmp(:)' ];
     known_beta = [ known_beta , beta ];
     
     %%% and lookup current v's in the new lookup table row.
     %   w = interp1q(xx',lookup_v(end,:)',v(:));
     %   w = qinterp1(xx',lookup_v(end,:)',v(:),1);
     %   w = reshape(w,size(v,1),size(v,2));
     if (maxNumCompThreads > 1)
       w = pointOp_mt(double(v),lookup_v(end,:),-RANGE,STEP,0);
     else
       w = pointOp(double(v),lookup_v(end,:),-RANGE,STEP,0); 
     end;
     
     fprintf('Recomputing lookup table for new value of beta\n');
   end
   
 else
   
   w = compute_w(v,beta);
   
 end
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% for a general alpha, use Newton-Raphson; more accurate root-finders may
% be substituted here; we are finding the roots of the equation:
% \alpha*|w|^{\alpha - 1} + \beta*(v - w) = 0
%
% @param v target values v the size of the feautre maps.
% @param beta the constant beta clamping to the feature maps.
% @param alpha the sparsity norm.
% @param kappa the coefficient on the sparsity term (usually just set to 1).
% @retval w the best computed root per-pixel.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w=compute_w(v,beta)
  
% UNFINISHED
  % Function to solve per-pixel quartic equations of the form:
  %  
  %%% Full Cubic equation:
  % x^3   - 2v * x^2   + v^2 * x  +  m  = 0;
  % A        B             C       D 
  %
  % Input: Target values v the size of the image, and constant beta
  % Output: Best root per pixel.
   
  DEBUG = 0;  
  EPSILON = 1e-6; %% tolerance on imag part of real root  
    
  k = -0.25/beta^2;
  m = ones(size(v))*k.*sign(v);

  if DEBUG
  
    % Do exact solve
    c = [1 -2*v v^2 k];
    exact_roots = roots(c)
  
    %% Check math derivation
    xx = [-5:0.01:2]; beta2 = 1; v = -1;
    f = abs(xx).^0.5 + (beta2/2)*(xx-v).^2;
    df1 = 0.5*sign(xx).*abs(xx).^(-0.5)+beta2*(xx-v);
    df2 = xx.^3 - 2*v*xx.^2 + v^2*xx + (1/(4*beta2^2)); % last term goes + for -ve v
    figure; subplot(2,1,1); plot(xx,f,'b'); axis([-5 2 0 2.5]); grid on;
    subplot(2,1,2); plot(xx,df1,'r'); axis([-5 2 -6 6]); grid on; hold on;
    plot(xx,df2,'g');
    
  end
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%% Compute the roots (all 3)
  
  %%%%% Direct solve from Mathematica
  
  % Command: 
  % a=1; b=-2v; c=v^2; d=m;
  % Solve[a*x^3 + b*x^2 + c*x + d == 0, x]

  t1 = (2/3)*v; 
  
  v2 = v .* v;
  v3 = v2 .* v;
  
  %%% slow (50% of time), not clear how to speed up...
  t2 = exp(log(-27*m - 2*v3 + (3*sqrt(3))*sqrt(27*m.^2 + 4*m.*v3))/3);
  
  t3 = v2./t2;
  
  %%% find all 3 roots
  root = zeros(size(v,1),size(v,2),3);
  root(:,:,1) = t1 + (2^(1/3))/3*t3 + (t2/(3*2^(1/3)));
  root(:,:,2) = t1 - ((1+i*sqrt(3))/(3*2^(2/3)))*t3 - ((1-i*sqrt(3))/(6*2^(1/3)))*t2;
  root(:,:,3) = t1 - ((1-i*sqrt(3))/(3*2^(2/3)))*t3 - ((1+i*sqrt(3))/(6*2^(1/3)))*t2;
  
  root(find(isnan(root) | isinf(root))) = 0; %%% catch 0/0 case
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%% Pick the right root

  %%% Clever fast approach that avoids lookups
  v2 = repmat(v,[1 1 3]); 
  sv2 = sign(v2);
  rsv2 = real(root).*sv2;
  root_flag3 = sort(((abs(imag(root))<EPSILON) & ((rsv2)>(2*abs(v2)/3)) &  ((rsv2)<(abs(v2)))).*rsv2,3,'descend').*sv2;
  %%% take best
  w=root_flag3(:,:,1);
  
  %%%%%%%%%%%%% Slow but safe way
  %%% make into nPixels by 3 and add in zero solution, just in case.
  %root = [ reshape(real(root),[prod(size(v)) 3]) , zeros(prod(size(v)),1) ]';
  
  %%% put solutions back into equaion: |root|^0.5 + beta/2 ||root - v||^2
  %%% evaluate all roots and zero solution to find min. cost
  %cost = sqrt(abs(root)) + 0.5*beta*(root-(ones(4,1)*v(:)')).^2;
  %%% find min cost
  %[tmp,ind] = min(cost,[],1);
  
  %%% select corresponding values of w
  %ind = ind + ([1:4:prod(size(root))]-1);
  %w = reshape(root(ind),[size(v,1) size(v,2)]);

  %%% possible way to speed up: look for real roots btw. v and 0,
  %including 0 option.
