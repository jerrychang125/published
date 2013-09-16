%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Function to solve per-pixel quartic equations of the form:
%  
%%% Full Quartic equation:
%  x^4   - 3v * x^3   + 3v^2 * x^2   - v^3 * x  +  m  = 0;
% A        B             C              D          E 
%
% @file
% @author Dilip Krishnan
% @date Mar 11, 2010
%
% @optimization_file @copybrief quartic_solve_image.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief quartic_solve_image.m
% @param v target values v the size of the feautre maps.
% @param beta the constant beta clamping to the feature maps.
% @retval w the best computed root per-pixel.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w] = quartic_solve_image(v,beta)

 % use this code if input data is normalized between 0 and 1
 RANGE = 0.6;
 STEP  = 0.0001;
 
 % use this code for v in full dynamic range
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
%      if (maxNumCompThreads > 1)
       w = pointOp_mt(double(v),lookup_v(ind,:),-RANGE,STEP,0);
%      else
%        w = pointOp(double(v),lookup_v(ind,:),-RANGE,STEP,0);
%      end;
   else
     %%% now go and recompute xx for new value of beta.
     tmp = compute_w(xx,beta);
     lookup_v =  [ lookup_v ; tmp(:)' ];
     known_beta = [ known_beta , beta ];
     
     %%% and lookup current v's in the new lookup table row.
     %   w = interp1q(xx',lookup_v(end,:)',v(:));
     %   w = qinterp1(xx',lookup_v(end,:)',v(:),1);
     %   w = reshape(w,size(v,1),size(v,2));
%      if (maxNumCompThreads > 1)
       w = pointOp_mt(double(v),lookup_v(end,:),-RANGE,STEP,0);
%      else
%        w = pointOp(double(v),lookup_v(end,:),-RANGE,STEP,0);
%      end;
     
     fprintf('Recomputing lookup table for new value of beta\n');
   end
   
 else
   
   w = compute_w(v,beta);
   
 end
 
  
function w = compute_w(v,beta)  
%
  % Function to solve per-pixel quartic equations of the form:
  %  
  %%% Full Quartic equation:
  %  x^4   - 3v * x^3   + 3v^2 * x^2   - v^3 * x  +  m  = 0;
  % A        B             C              D          E 
  %
  % Input: Coefficient matrices v and m, each the size of the image
  % Output: 4 Roots per pixel of equation, same size as v and m

  DEBUG = 0;  
  EPSILON = 1e-6; %% tolerance on imag part of real root  
       
  k = 8/(27*beta^3);
  m = ones(size(v))*k;
  
  if DEBUG
    % Do exact solve
    c = [1 -3*v 3*v^2 -v^3 k];
    exact_roots = roots(c);

    %% Check math derivation
    xx = [-16:0.01:16]; beta2 = 1; v = 0.1;
    f = abs(xx).^(2/3) + (beta2/2)*(xx-v).^2;
    df1 = ((2/3)*sign(xx).*(abs(xx).^(-1/3)))+beta2*(xx-v);
    df2 = xx.^4 - 3*v*xx.^3 + 3*v^2*xx.^2 - v^3*xx + (8/(27*beta2^3)); % last term goes - for + ve v
    figure; subplot(2,1,1); plot(xx,f,'b'); axis([-15 15 -16 16]); grid on;
    subplot(2,1,2); plot(xx,df1,'r'); axis([-15 15 -16 16]); grid on; hold on;
    plot(xx,df2,'g');
  
  end
  
  
  % Now use formula from
  % http://en.wikipedia.org/wiki/Quartic_equation (Ferrari's method)
  % running our coefficients through Mathmetica (quartic_solution.nb)
  
  % optimized by Rob to use as few operations as possible...
        
  %%% precompute certain terms
  v2 = v .* v;
  v3 = v2 .* v;
  v4 = v3 .* v;
  m2 = m .* m;
  m3 = m2 .* m;
  
  %% Compute alpha & beta
  alpha = -1.125*v2;
  beta2 = 0.25*v3;
  
  %%% Compute p,q,r and u directly.
  q = -0.125*(m.*v2);
  r1 = -q/2 + sqrt(-m3/27 + (m2.*v4)/256);

  %u = r1.^(1/3);   %% supposedly either root with do, but not sure why
  u = exp(log(r1)/3);   %% slightly faster, not sure why
  
  %% proper way to do it....
  %ind = find(u==0);
  %ind2 = find(u~=0);
  %y = zeros(size(v));
  %y(ind) = 2*(-5/18*alpha(ind) + u(ind) - q(ind).^(1/3));
  %y(ind2) =  2*(-5/18*alpha(ind2) + u(ind2) + (m(ind2)./(3*u(ind2)))); 
  
  %% if you assume that u is never 0:
  y = 2*(-5/18*alpha + u + (m./(3*u))); 
    
  W = sqrt(alpha./3 + y);
  
  %%% now form all 4 roots
  root = zeros(size(v,1),size(v,2),4);
  root(:,:,1) = 0.75.*v  +  0.5.*(W + sqrt(-(alpha + y + beta2./W )));
  root(:,:,2) = 0.75.*v  +  0.5.*(W - sqrt(-(alpha + y + beta2./W )));
  root(:,:,3) = 0.75.*v  +  0.5.*(-W + sqrt(-(alpha + y - beta2./W )));
  root(:,:,4) = 0.75.*v  +  0.5.*(-W - sqrt(-(alpha + y - beta2./W )));
  
    
  %%%%%% Now pick the correct root, including zero option.
  
    %%% Clever fast approach that avoids lookups
    v2 = repmat(v,[1 1 4]); 
    sv2 = sign(v2);
    rsv2 = real(root).*sv2;
    
    % take out imagingary roots, take out roots outside range 0 to v....
    %root_flag = ((abs(imag(root))<EPSILON) & ((root.*sign(v2))>0) &  ((root.* sign(v2))<(v2.*sign(v2)))).*root;
    % pick one closest to v
    %root_flag2 = sort(root_flag.*sign(v2),3,'descend').*sign(v2);
    % apply zero check
    %w = root_flag2(:,:,1);
    %w = ((root_flag2(:,:,1)./v)>(0.5)).*root_flag2(:,:,1);
    
    %%% condensed fast version
    %%%             take out imaginary                roots above v/2            but below v
    root_flag3 = sort(((abs(imag(root))<EPSILON) & ((rsv2)>(abs(v2)/2)) &  ((rsv2)<(abs(v2)))).*rsv2,3,'descend').*sv2;
    %%% take best
    w=root_flag3(:,:,1);
    
    
    
    
    %%%%% Slow but careful way of finding roots
    %%% make into nPixels by 3 and add in zero solution, just in case.
    %root = [ reshape(real(root),[prod(size(v)) 4]) , zeros(prod(size(v)),1) ]';
  
    %%% put solutions back into equaion: |root|^0.5 + beta/2 ||root - v||^2
    %%% evaluate all roots and zero solution to find min. cost
    %cost = (abs(root)).^(2/3) + 0.5*beta*(root-(ones(5,1)*v(:)')).^2;
    %%% find min cost
    %[tmp,ind] = min(cost,[],1);
    
    %%% select corresponding values of w
    %ind = ind + ([1:5:prod(size(root))]-1);
    %w2 = reshape(root(ind),[size(v,1) size(v,2)]);
    
