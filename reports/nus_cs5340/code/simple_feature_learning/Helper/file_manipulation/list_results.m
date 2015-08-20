%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% List the given parameters (from model structure) for each Run it finds in
% the given input directory.
% The path should have no '/' characters at the start or end.
% The first agruement should always be the path to the directory of choice.
% The following arguements are any other variables you want to display.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief getsubdirs.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief getsubdirs.m
%
% @param varagin \li \e path the first arguement is the input path. \li \e
% arguemnts any other varaibles you want to display.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = list_results(varargin)

format loose

% Get the input path.
path = varargin{1};

%% Preprocess the input path.
% Remove the '/' at the end of the path.
if(strcmp(path(end),'/'))
    path = path(1:end-1);
end
% if(strcmp(path(1),'/'))
%     path = path(2:end);
% end


if(isdir(path))
    
    % Get all files and directories.
    subdirs = dir(path);
    
    % Remove the . and .. directories.
    % subdirs = subdirs(find(~cellfun(@cmp_dots,{subdirs(:).name})));
    subdirs = remove_dots(subdirs);
    
    [subfolders,subfiles] = split_folders_files(subdirs);
    
    %     path
    % Extract the name of the folder.
    [a,pathfolder,c,d] = fileparts(path);
    
    %     [blah,pathfolder] = parentdir(subfolders)
    
    %% If it is a run directory then extract the highest file.
    if(strcmp(pathfolder(1:4),'Run_'))
        
        %% Find the last generated file.
        maxepoch = 0;
        maxlayer = 0;
        
        % Go over the files in the subdirectory and find the highest one
        % In terms of epoch and layer.
        % This is the file you want to list from.
        for file=1:length(subfiles)
            % Find only the run files.
            if(strcmp(subfiles(file).name(1:5),'epoch'))
                [epoch,layer] = extract_epoch_layer(subfiles(file).name);
                if(layer>maxlayer)
                    maxlayer = layer;
                end
            end
        end
        for file=1:length(subfiles)
            blahname = remove_dot_mat(subfiles(file).name);
            % Find only the run files with maxlayer
            if(strcmp(subfiles(file).name(1:5),'epoch') &&...
                    strcmp(blahname(end),num2str(maxlayer)))
                [epoch,layer] = extract_epoch_layer(subfiles(file).name);
                if(epoch>maxepoch)
                    maxepoch = epoch;
                end
            end
        end
        
        % Only recurse to that file if one was found.
        if(maxepoch> 0 && maxlayer > 0)
            %% Now recurse to the maxepoch,maxlayer file.
            newvarargin = varargin;
            newvarargin{1} = strcat(path,'/epoch',num2str(maxepoch),'_layer',num2str(maxlayer));
            % Put the cell into a string to pass in.
            newstr = '';
            for i=1:length(newvarargin)
                if(i==1)
                    newstr = strcat(newstr,char(39),newvarargin{i},char(39));
                else
                    newstr = strcat(newstr,',',char(39),newvarargin{i},char(39));
                end
            end
            
            eval(strcat('list_results(',newstr,');'))
        end
    else
        %         'recursing over folders'
        %% Have to recurse to each of the other folders.
        for folder=1:length(subfolders)
            
            
            %% Now recurse to the maxepoch,maxlayer file.
            newvarargin = varargin;
            newvarargin{1} = strcat(path,'/',subfolders(folder).name);
            % Put the cell into a string to pass in.
            newstr = '';
            for i=1:length(newvarargin)
                if(i==1)
                    newstr = strcat(newstr,char(39),newvarargin{i},char(39));
                else
                    newstr = strcat(newstr,',',char(39),newvarargin{i},char(39));
                end
            end
            %             strcat('list_results(',newstr,');')
            eval(strcat('list_results(',newstr,');'))
            
            
        end
        
    end
    
    
    
    
    
    
    
else
    %% If it is a file, read directly from it.
    
    [a,structname,c,d] = fileparts(path);
    
    % Load the model parameters.
    load(path,'model')
    newpath = path;
    i=1;
    while(i<=length(newpath))
        if(strcmp(newpath(i),'/'))
            newpath = strcat(newpath(1:i-1),'__',newpath(i+1:end));
            i = i+2;
        else
            i = i +1;
        end
    end
    
    % Truncate from the start if name is longer than the matlab maximum.
    if(length(newpath)>63)
        newpath = newpath(length(newpath)-62:end);
    else % Otherwise pad it to be 63 characters.
        %         for i=1:(63-length(newpath))
        %            newpath = strcat('X',newpath);
        %         end
    end
    
    % Now make sure the first letter isn't _'
    if(strcmp(newpath(1),'_'))
        newpath(1) = 'X';
    end
    
    % Default to displayin the models.
    if(length(varargin)==1)
        eval(strcat(newpath,' = model;'))
        CONMATS = 0;
    else
        
        % Process each of the variables.
        for i=2:length(varargin)
            
            if(strcmp(varargin{i},'conmats'))
                
                CONMATS = 1;
            elseif(strcmp(varargin{i},'model')) % Show the whole model.
                CONMATS = 0;
                eval(strcat(newpath,' = model;'))
            else
                CONMATS = 0;
                try
                    eval(strcat(newpath,'.',varargin{i},' = model.',varargin{i},';'))
                catch
                    fprintf('XXXXX Failed to assign variable "%s" to new model structure.XXXXX\n',varargin{i})
                end
            end
        end
    end
    
    % Display the results
    eval(newpath)
    % Display connectivity matrices after the rest.
    if(CONMATS)
        
        eval(strcat('conlength = length(model.conmats);'))
        for j=1:conlength
            eval(strcat('disp(model.conmats{j});'))
        end
        
    end
    
    
end
end