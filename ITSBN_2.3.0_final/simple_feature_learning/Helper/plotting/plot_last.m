%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% It reads the fullsavepath from guimodel (which it expects to be in the 
% workspace) and appends the expfile name if needed.
% It then loads up all the saved figures it finds in that directory (and lists
% them first as well).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @plotting_file @copybrief plot_last.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(~exist('guimodel','var'))
    fprintf('\nguimodel does not exist in the workspace and so the previous experiment is unknown\n');
else
    
    expfile = guimodel.expfile;
    if(strcmp(expfile(1),' '))
        expfile = expfile(2:end);
    end
    fullsavepath = guimodel.fullsavepath
    
    fullsavepath(end-length(expfile)-1:end-1)
    % Already pointing to an experiment that just saves figures.
    if(~strcmp(fullsavepath(end-length(expfile):end-1),expfile))
        % Saving the figures was a last resort.
        fullsavepath = strcat(fullsavepath,expfile);
    end
    
    % Now find the subdiretory named epoch## with the highest epoch number.
    subdir = dir(fullsavepath);
    [subfolders,subfiles] = split_folders_files(subdir);
    subfolders = remove_dots(subfolders);
    last_epoch = 0;
    for i=1:length(subfolders)
        subfolders(i).name(1:5)
        if(strcmp(subfolders(i).name(1:5),'epoch'))
           if(last_epoch < str2num(subfolders(i).name(6:end)))
               last_epoch = str2num(subfolders(i).name(6:end));
           end
        end
    end
    % Path to the last epoch.
    lastpath = strcat(fullsavepath,'/epoch',num2str(last_epoch));
    
    % List them
    fprintf('Here are the figures I found:\n\n')
    view_results(fullsavepath,'list');
    view_results(lastpath,'list recurse');
   
    
    user_entry = input('Do you want to plot all these figures? (y/n):\n','s');
    if(strcmp(user_entry,'y') || strcmp(user_entry,'yes'))
        % Show them
        view_results(fullsavepath,'view');
        view_results(lastpath,'view recurse');
    end
    
    
end


