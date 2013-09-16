%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Displays the filters from the model as a multiple images in the same
% ultimateSubplot and one subplot for each input map. All the images displayed
% are scaled to have the contrast (within each subplot only).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @plotting_file @copybrief sdispfilts.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief sdispfilts.m
%
% @param imstack the filters in xdim x ydim x input_maps x feature_maps
% @param cm the connectivity matrix for this layer (must be passed in).
% @param n2 [optional] the number of number of rows in the resulting plot.
% Defaults to ceil(sqrt(num_feature_maps)).
% @param titles [optional] Defaults to 'none': \li 'none' for no titles li 'auto' for numbered titles
% automatically generates \li {cell} passed in for different specified titles
% per subplot. 
% @param scalar [optional] a multiplicative constant to change the constrast of
% the plot. Defaults to 1.
% @param border [optional] the size of the border between plots. Defaults to 2.
% @param fud [optional] flips the images up-down. Defaults to 0.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = sdispfilts(imstack,cm,n2,titles,scalar,border,fud)

if(ndims(imstack)==4)
    [drows,dcols,num_input_maps,num_feature_maps]=size(imstack);
    numcolors = 1; % Always displaed in gray.
elseif(ndims(imstack==3)) % Assumes the last dimension is 1
    [drows,dcols,num_input_maps]=size(imstack);
    num_feature_maps = 1;
    numcolors = 1; % Always displaed in gray.
else
    error('scale_dispfilts: imstack must be a 4 dimensional array');
end

if(nargin<7) fud=0; end
if(nargin<6) border=2; end
if(nargin<5) scalar=1; end
if(nargin<4) titles = 'none'; end
if(nargin<3) n2=ceil(sqrt(num_feature_maps)); end % n2 is number of rows of images.
if(nargin<2) error('scale_dispfilts: Need to input (F,connectivity_matrix)'); end
% If not titles are passed in and it's not auto titles, then no titles
% should be used.
if(iscell(titles)==0 && strcmp(titles,'auto')==0)
    titles = 'none';
end

if(size(cm,1) ~= num_input_maps || size(cm,2) ~= num_feature_maps)
    error('scale_dispfilts: connectivity matrix is not the correct size')
end


% Size of each square.
drb=drows+border;
dcb=dcols+border;

% Setup the size for ultimateSubplot
sub2 = floor(sqrt(num_input_maps)); % number of rows of subplots
sub1 = ceil(num_input_maps/sub2);  % number of cols of subplots

% Get the cofeature_mapectivity matrix.
% cm = model.conmats{layer};

% Make an ultimate subplot for each of the input maps.
for input_map=1:num_input_maps
    % Initialize the image size to -1 so that borders are black still.
    imdisp=zeros(n2*drb-border,ceil(num_feature_maps/n2)*dcb-border,numcolors);
    border_indices = ones(n2*drb-border,ceil(num_feature_maps/n2)*dcb-border,numcolors);
    
    for feature_map=1:num_feature_maps
        
        ii=rem(feature_map,n2); 
        if(ii==0) 
            ii=n2; 
        end
        jj=ceil(feature_map/n2);
        
        % If there is a cofeature_mapection between input_map and feature map.
        if(cm(input_map,feature_map))
            imdisp(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border),:) ...
                = imstack(:,:,input_map,feature_map);
        else
            imdisp(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border),:) = 0;
        end
        border_indices(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border),:) = 0;
    end
    
    if(fud)
        imdisp=flipud(imdisp);
    end
    
    %%%%%%%%%%
    %% Scale the input values to be between zero and one.
    %     m = mean(imstack(:));
    %     imdisp = uint8(((imdisp-m)*255)+128);
    %% Scale the input values to be between zero and one.
    A = imdisp-min(imstack(:));  % shifts the bottom of the array to 0.
    maxA = max(imstack(:)-min(imstack(:))); % need to use imstack so that each display is set consistently over the whole figure.
    A(find(border_indices==1))=0;   % Set the borders back to zero.
    A = A/maxA;  % Scale the max to be 1.
    imdisp = uint8(A*255*scalar);  % convert to unsigned ints between (0,255)
    
    % Set the non connected compnents to black again.
    for feature_map=1:num_feature_maps
        ii=rem(feature_map,n2); if(ii==0) ii=n2; end
        jj=ceil(feature_map/n2);
        
        % If there is a cofeature_mapection between input_map and feature map.
        if(cm(input_map,feature_map)==0)
            imdisp(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border),:) = 0;
        end
    end
    %%%%%%%%%%
    
    % Make the border pixels black after scalin (otherwise scaling would be
    % altered).
    imdisp(find(border_indices==1)) = 0;
    
    % Just like location within the subplot, get location of subplot.
    % Except these are indices so +1 is added.
    subii = rem(input_map,sub2);
    if(subii==0)
        subii=sub2;
    end
    subjj=ceil(input_map/sub2);
    
    % Ultimate subplot is backwards for some reason.
    ultimateSubplot(sub1,sub2,subjj,subii,0.1);
    imshow(imdisp,'Border','tight'); axis equal; axis off;
    
    % Process the titles if any.
    if(iscell(titles))
        if(input_map <= length(titles)) % ensure not exceeding titles size
            title(titles{input_map});
        end
    elseif(strcmp(titles,'auto')) % autogenerate titles.
        title(strcat('F',num2str(input_map)));
    end
    
    
    drawnow;
    
end

