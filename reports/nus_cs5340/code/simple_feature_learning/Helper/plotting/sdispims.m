%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Displays a stack of images in the specified color space separated by a border
% and scaled to have the same contrast.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @plotting_file @copybrief sdispims.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief sdispims.m
%
% @param imstack the filters in xdim x ydim x color_planes x num_images
% @param COLOR_TYPE [optional] 'gray','rgb','ycbcr','hsv' specifying the
% colorspace of the input images. Default is 'rgb' (doesn't do anything to the
% resulting image before displaying it).
% @param n2 [optional] the number of number of rows in the resulting plot.
% Defaults to ceil(sqrt(num_feature_maps)).
% @param titles [optional] Defaults to 'none': \li 'none' for no titles li 'auto' for numbered titles
% automatically generates \li {cell} passed in for different specified titles
% per subplot. 
% @param scalar [optional] a multiplicative constant to change the constrast of
% the plot. Defaults to 1.
% @param border [optional] the size of the border between plots. Defaults to 2.
% @param fud [optional] flips the images up-down. Defaults to 0.
% @retval imdisp the entire image displayed together is returned.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [imdisp] = sdispims(imstack,COLOR_TYPES,n2,titles,scalar,border,fud)

imstack = double(imstack);

if(ndims(imstack)==4)
    [drows,dcols,numcolors,N]=size(imstack);
elseif(ndims(imstack)==3) % If only one sample, it may appear as a 3D array.
    [drows,dcols,N]=size(imstack);
    imstack=reshape(imstack,[drows,dcols,3,1]);
    numcolors=3;
    N=1;
elseif(ndims(imstack)==2)
    [drows,dcols] = size(imstack);
    N=1;
    numcolors=1;
    imstack = reshape(imstack,[drows,dcols,1,1]);
    %     error('dispims3: imstack must be a 3 or 4 dimensional array');
end

if(nargin<7) fud=0; end
if(nargin<6) border=2; end
if(nargin<5) scalar=1; end
if(nargin<4) titles = 'none'; end
if(nargin<3)
    n2=ceil(sqrt(N));   % n2 is number of rows of images.
end
if(nargin<2)
    COLOR_TYPES = 'rgb';   % rgb is deault and doesn't do anything.
end
% If not titles are passed in and it's not auto titles, then no titles
% should be used.
if(iscell(titles)==0 && strcmp(titles,'auto')==0)
    titles = 'none';
end


% Size of each square.
drb=drows+border;
dcb=dcols+border;

% Initialize the image size to -1 so that borders are black still.
imdisp=zeros(n2*drb-border,ceil(N/n2)*dcb-border,numcolors);
border_indices = ones(n2*drb-border,ceil(N/n2)*dcb-border,numcolors);

for nn=1:N
    
    ii=rem(nn,n2); if(ii==0) ii=n2; end
    jj=ceil(nn/n2);
    
    imdisp(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border),:) ...
        = imstack(:,:,:,nn);
    border_indices(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border),:) = 0;
end

if(fud)
    imdisp=flipud(imdisp);
end

%%%%%%%%%%
%% Scale the input values to be between zero and one.
% m = mean(imstack(:))
A = scalar*imdisp;
A = (imdisp-min(imstack(:)));  % shifts the bottom of the array to 0.
maxA = max(imstack(:)-min(imstack(:))); % need to use imstack so that each display is set consistently over the whole figure.
A(find(border_indices==1))=0;   % Set the borders back to zero.
A = A/maxA;
imdisp = uint8(A*255); % Convert to unsigned in and scale if needed.
%%%%%%%%%%
% Make the border pixels black after scalin (otherwise scaling would be
% altered).
switch COLOR_TYPES
    case 'gray'
        imdisp = rgb2gray(imdisp);
    case 'rgb'
        
    case 'ycbcr'
        imdisp = ycbcr2rgb(imdisp);
    case 'hsv'
        imdisp = hsv2rgb(imdisp);
end
imdisp(find(border_indices==1)) = 0;

imshow(imdisp,'Border','tight'); axis equal; axis off;

if(numcolors==1)
    colormap gray;
end

% Process the title if there is one.
if(iscell(titles))
    title(titles{1});
elseif(strcmp(titles,'auto')) % autogenerate titles.
    title('Image');
end


drawnow;

