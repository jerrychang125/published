%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% View the specified figures for each Run it finds in a directory or recursively
% from that directory down.
% The path should have no '/' characters at the start or end.
% The first agruement should always be the path to the directory of choice.
% The following arguements are any other variables you want to display.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @plotting_file @copybrief view_results.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief view_results.m
%
% @param varargin{1} this is the path which show have no '/' character at the
% start or the end.
% @param varargin{2} this string indicates how you want to view the results.
% \li 'gui' opens a gui to find a specific figure. \li 'list' lists all the .fig files within the
% directory. \li 'list recurse' lists all the .fig files within and below the
% directory. \li 'view' plots all the .fig files within the directory. 
% \li 'view recurse' plots all the .fig files within and below the
% directory.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = view_results(varargin)



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


%% Open a gui to load a specific file.
if(length(varargin)>1 && strcmp(varargin{2},'gui'))
    if(isdir(path))
        savedpwd = pwd;
        cd(path)
        uiopen('FIGURE')
        cd(savedpwd)
    else
        uiopen('FIGURE')
    end
    
elseif(length(varargin)>1 && strcmp(varargin{2},'list'))
    %% Just list all files with .fig extensions (not recursively).
    % Get all files and directories.
    subdirs = dir(path);
    
    % Remove the . and .. directories.
    % subdirs = subdirs(find(~cellfun(@cmp_dots,{subdirs(:).name})));
    subdirs = remove_dots(subdirs);
    
    [subfolders,subfiles] = split_folders_files(subdirs);
    
    %% List all files in this current folder then recurse.
    % This is the file you want to list from.
    for file=1:length(subfiles)
        filestr = subfiles(file).name;
        [filepath,filename,fileext,d] = fileparts(filestr);
        % If it is a figure plot it.
        if(strcmp(fileext,'.fig'))
            fprintf('Directory: %s , File: %s\n',path,strcat(filename,fileext))
        end
    end
    
elseif(length(varargin)>1 && strcmp(varargin{2},'list recurse'))
    %% Just list all files with .fig extensions (recursively).
    % Get all files and directories.
    subdirs = dir(path);

%     subdirs = dir(path(2:end);
    % Remove the . and .. directories.
    % subdirs = subdirs(find(~cellfun(@cmp_dots,{subdirs(:).name})));
    subdirs = remove_dots(subdirs);
    
    [subfolders,subfiles] = split_folders_files(subdirs);
    
    %% List all files in this current folder then recurse.
    % This is the file you want to list from.
    for file=1:length(subfiles)
        filestr = subfiles(file).name;
        [filepath,filename,fileext,d] = fileparts(filestr);
        % If it is a figure plot it.
        if(strcmp(fileext,'.fig'))
            fprintf('Directory: %s , File: %s\n',path,strcat(filename,fileext))
        end
    end
    
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
        eval(strcat('view_results(',newstr,');'))
    end
    
elseif(length(varargin)>1 && strcmp(varargin{2},'view recurse'))

    %% Plot each figure in this directory (and recurse to next directories).
    % Only process if it is a directory.
    if(isdir(path))
        % Get all files and directories.
        subdirs = dir(path);
        
        % Remove the . and .. directories.
        % subdirs = subdirs(find(~cellfun(@cmp_dots,{subdirs(:).name})));
        subdirs = remove_dots(subdirs);
        
        [subfolders,subfiles] = split_folders_files(subdirs);
        
        ANY_FIGS = 0;
        % This is the file you want to list from.
        for file=1:length(subfiles)
            filestr = subfiles(file).name;
            [filepath,filename,fileext,d] = fileparts(filestr);
            % If it is a figure plot it.
            if(strcmp(fileext,'.fig'))
                f = hgload(strcat(path,'/',filestr));
                get(f,'Name')
                get(f,'Position')
                set(f,'Name',get(f,'Name')); % Set the name back.
                set(f,'Position',get(f,'Position'));  % Set the position back.
                ANY_FIGS = 1; % SHOWS THERE WAS A FIGURE IN THIS DIRECTORY
            end
        end
        if(~ANY_FIGS)
            fprintf('Note: There are no figure files in the current directory. Recursing now...\n')
        end
        
        %% Have to recurse to each of the other folders.
        for folder=1:length(subfolders)           
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
            eval(strcat('view_results(',newstr,');'))
            
            
        end
        
    else % It must be path directl to a figure file.
        [filepath,filename,fileext,d] = fileparts(path);
        if(strcmp(fileext,'.fig'))
            f = hgload(path);
            
                set(f,'Name',get(f,'Name')); % Set the name back.
                set(f,'Position',get(f,'Position'));  % Set the position back.
        else
            fprintf('Note: First arguement must be a path to a directory of figures to plot or directly to a .fig file.\n')
        end
        
        
    end
    
    
    
elseif(length(varargin)>1 && strcmp(varargin{2},'view'))
 %% Plot each figure in this directory (and do not! recurse to next directories).
    % Only process if it is a directory.
    if(isdir(path))
        % Get all files and directories.
        subdirs = dir(path);
        
        % Remove the . and .. directories.
        % subdirs = subdirs(find(~cellfun(@cmp_dots,{subdirs(:).name})));
        subdirs = remove_dots(subdirs);
        
        [subfolders,subfiles] = split_folders_files(subdirs);
        
        ANY_FIGS = 0;
        % This is the file you want to list from.
        for file=1:length(subfiles)
            filestr = subfiles(file).name;
            [filepath,filename,fileext,d] = fileparts(filestr);
            % If it is a figure plot it.
            if(strcmp(fileext,'.fig'))
                f = hgload(strcat(path,'/',filestr));
                set(f,'Name',get(f,'Name')); % Set the name back.
                set(f,'Position',get(f,'Position'));  % Set the position back.
                ANY_FIGS = 1; % SHOWS THERE WAS A FIGURE IN THIS DIRECTORY
            end
        end
        if(~ANY_FIGS)
            fprintf('Note: There are no figure files in the current directory. Recursing now...\n')
        end        
    else % It must be path directl to a figure file.
        [filepath,filename,fileext,d] = fileparts(path);
        if(strcmp(fileext,'.fig'))
            f = hgload(path);
                set(f,'Name',get(f,'Name')); % Set the name back.
                set(f,'Position',get(f,'Position'));  % Set the position back.            
        else
            fprintf('Note: First arguement must be a path to a directory of figures to plot or directly to a .fig file.\n')
        end
        
        
    end
    
else
   error('Not a valid string input. Valid strings are: gui, list, list recurse, plot, plot recurse') 
end


end