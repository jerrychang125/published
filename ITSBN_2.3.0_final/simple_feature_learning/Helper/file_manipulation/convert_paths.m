%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Checks to see if the fullsavepath and fulldatapath exist on the given
% machine.
% If they do not, then change the first part of the directory to match.
% This assumes if the directories doesn't exist then must be different
% machine.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief convert_paths.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief convert_paths.m
%
% @param model the standard struct of the model parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model] = convert_paths(model)


% model.datadirectory is the ending potion of fulldatapath
if(exist(model.fullmodelpath,'dir') == 0)
    % If running locally, then use the pwd which contains Results
    if(strcmp(model.machine,' laptop'))
        model.fullmodelpath = strcat('/Users/mattzeiler/work/Thesis/A',model.modeldirectory);
    else % Otherwise, use the raid array.
        model.fullmodelpath = strcat('/misc/FergusGroup/zeiler',model.modeldirectory);
    end 
end

% model.datadirectory is the ending potion of fulldatapath
if(exist(model.fulldatapath,'dir') == 0)
    % If running locally, then use the pwd which contains Results
    if(strcmp(model.machine,' laptop'))
        model.fulldatapath = strcat('/Users/mattzeiler/work/Thesis/A',model.datadirectory);
    else % Otherwise, use the raid array.
        model.fulldatapath = strcat('/misc/FergusGroup/zeiler',model.datadirectory);
    end 
end

% model.savedirectory si the ending portion of fullsavepath

if(exist(model.fullsavepath,'dir') == 0)
    model.fullsavepath = strcat(pwd,model.savedirectory);
    % If running locally, then use the pwd which contains Results
    if(strcmp(model.machine,' laptop'))
        model.fullsavepath = strcat('/Users/mattzeiler/work/Thesis/A',model.savedirectory);
    else % Otherwise, use the raid array.
        model.fullsavepath = strcat('/misc/FergusGroup/zeiler',model.savedirectory);
    end
end

% model.pwd is where the experiment was originally run from.
model.pwd = pwd;

end