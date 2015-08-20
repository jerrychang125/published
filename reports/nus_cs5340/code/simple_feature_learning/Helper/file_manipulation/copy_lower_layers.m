%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Copies all the layers below the top layer you want to train above (which is specified in
% model.fullmodelpath of the gui model that is input).
% The model passed in is the gui model which has the path to the top layer
% model.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief copy_lower_layers.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief copy_lower_layers.m
%
% @param model the standard struct of the model parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = copy_lower_layers(model)

% Save the input model.
guimodel = model;

% If there is .mat at the end of the fulldatapath for the top layer, remove
% it so that processing below can load layer by layer.
% Make it a matrix.
topmodelpath = remove_dot_mat(guimodel.fullmodelpath);

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


% Only do this if the gui's directories are different.
% Have to compare the directory of the model path since it is a file.
if(strcmp(guimodel.fullsavepath,dir_of_file(guimodel.fullmodelpath)) == 0 && model.SAVE_RESULTS>0)
    
    guimodel.fullmodelpath = add_dot_mat(guimodel.fullmodelpath);
    
    % Load the top layer of the model.
    load(guimodel.fullmodelpath);
    % Make sure loaded layers are compatible.
    model = backwards_compatible(model);    
    
    % Get where the pooling files are stored.
    pooldir = parentdir(guimodel.fullmodelpath);
    
    % Save the top model and layer to derive the epochs of layers below
    % from.
    top_model = model;
    top_layer = model.layer;
    
    % Copy the top layer model to the gui's destination save path.
    copyfile(guimodel.fullmodelpath,guimodel.fullsavepath);
    % Copy the top pooling layer.
    if(exist(strcat(pooldir,'/layer',num2str(top_layer),'_pooling.mat'),'file'))
        copyfile(strcat(pooldir,'/layer',num2str(top_layer),'_pooling.mat'),guimodel.fullsavepath)
    end
    % Copy the input pooling of the image.
    if(exist(strcat(pooldir,'/layer0_pooling.mat'),'file'))
        copyfile(strcat(pooldir,'/layer0_pooling.mat'),guimodel.fullsavepath)
    end
    
    % Save the variables for each of the other layers.
    for layer=top_layer-1:-1:1
        
        % Copy the model's below at the epochs for the top layer model.
        epoch = top_model.maxepochs(layer);
        copyfile(add_dot_mat(strcat(startpath,num2str(epoch),'_layer',num2str(layer))),...
            guimodel.fullsavepath);
        
        % Copy the other pooling layers.
        if(exist(strcat(pooldir,'/layer',num2str(layer),'_pooling.mat'),'file'))
            copyfile(strcat(pooldir,'/layer',num2str(layer),'_pooling.mat'),guimodel.fullsavepath)
        end
    end
end
end