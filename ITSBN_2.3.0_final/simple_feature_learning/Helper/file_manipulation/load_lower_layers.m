%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Loads all the layers below the top layer you want to use (which is specified in
% model.fullmodelpath of the gui model that is input).
% The model passed in is the gui model which has the path to the top layer
% model.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inparam \li \e fullmodepath a path to the top model (loaded in here)
%
% @fileman_file @copybrief load_lower_layers.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Remove the '.mat' from the top model's path.
topmodelpath = remove_dot_mat(fullmodelpath);

% Checks how many epochs are in the fullmodelpath (after .mat is removed)
% The startpath is used as prefixes for the epochs that are read from the
% top_model.
if(strcmp(topmodelpath(end-8:end-8),'h')) %Single digit epochs
    startpath = topmodelpath(1:end-8);
elseif(strcmp(topmodelpath(end-9:end-9),'h')) % Double digit epochs
    startpath = topmodelpath(1:end-9);
else % Triple digit epochs
    startpath = topmodelpath(1:end-10);
end

% Load the top layer of the model.
% load(fullmodelpath,'model','F','z0');  % Only need these varaibles.
load(fullmodelpath);  % Only need these varaibles.

% Get where the pooling files are stored.
pooldir = parentdir(fullmodelpath);

% Save the top model and layer.
top_model = backwards_compatible(model);
top_layer = model.layer;
% Save the size of the top layer (for one training sample)
top_size = size(z(:,:,:,1));


% Save the epochs of the layers below.
maxepochs = model.maxepochs;



% Save the variables of the top layer.
eval(strcat('model',num2str(model.layer),'=model;'));
eval(strcat('F',num2str(model.layer),'=single(F);'));
% eval(strcat('z',num2str(model.layer),'=z;'));
eval(strcat('z0',num2str(model.layer),'=single(z0);'));


% If there was pooling done previously then load the pooling file.
% This loads both the indices and maps (only maps for this top layer).
if(exist(strcat(pooldir,'/layer',num2str(top_layer),'_pooling.mat'),'file'))
    load(strcat(pooldir,'/layer',num2str(top_layer),'_pooling.mat'))
    eval(strcat('pooled_indices',num2str(top_layer),' = pooled_indices;'));
    eval(strcat('pooled_maps',num2str(top_layer),' = pooled_maps;'));
    top_size = size(pooled_maps(:,:,:,1)); % Overwrite the top_size to be the pooled size.
    clear pooled_maps
else % Make them empty if there was no pooling done.
    eval(strcat('pooled_indices',num2str(top_layer),' = [];'));
end

% If there pooling of the input image then load that too
if(exist(strcat(pooldir,'/layer0_pooling.mat'),'file'))
    load(strcat(pooldir,'/layer0_pooling.mat'))
    pooled_indices0 = pooled_indices;
else % Make them empty if there was no pooling done.
    pooled_indices0 = [];
end


% Save the variables for each of the other layers.
for layer=model.layer-1:-1:1
    
    % Load the model's below at the epochs for the top layer model.
    epoch = maxepochs(layer);
    try % Try if this file exists.
        load(strcat(startpath,num2str(epoch),'_layer',num2str(layer)));
    catch % Load the highest one you can find.
        load(strcat(startpath,num2str(get_highest_epoch(parentdir(startpath),layer)),'_layer',num2str(layer)));
    end
    % Make sure loaded layers are compatible.
    model = backwards_compatible(model);
    
    % Make new variable names based on the layer you are currently on.
    eval(strcat('model',num2str(model.layer),'=model;'));
    eval(strcat('F',num2str(model.layer),'=single(F);'));
    %         eval(strcat('z',num2str(model.layer),'=z;'));
    eval(strcat('z0',num2str(model.layer),'=single(z0);'));
    
    % If there was pooling done previously then load the pooling file.
    if(exist(strcat(pooldir,'/layer',num2str(layer),'_pooling.mat'),'file'))
        load(strcat(pooldir,'/layer',num2str(layer),'_pooling.mat'))
        eval(strcat('pooled_indices',num2str(layer),' = pooled_indices;'));
    else % Make them empty if there was no pooling done.
        eval(strcat('pooled_indices',num2str(layer),' = [];'));
    end
    
end
clear z z0 F model maxepochs startpath epoch layer pooled_indices pooled_maps


% returns the model#,F#,z0# triples for each layer