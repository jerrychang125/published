%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Searches a directory for the .mat named as epoch##_layer# that has the highest
% epoch for the given input layer.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief get_highest_epoch.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief get_highest_epoch.m
%
% @param path a directory of epoch##_layer#.mat files.
% @param layer only search for files that match this layer.
% @retval maxepoch the maximum epoch for the given layer found in the path.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [maxepoch] = get_highest_epoch(path,layer)

% searches the given path for the highest epoch of the a filename of the
% form epoch##_layer#.mat.

dirCell = dir(path);

[subfolder,subfiles] = split_folders_files(dirCell);

maxepoch = -1;

for file=1:length(subfiles)
    % Get the name of the file.
    filename = remove_dot_mat(subfiles(file).name);
    
    
     % Make sure it is for the given layer.
    if(strcmp(filename(end:end),num2str(layer)))
        
    if(strcmp(filename(end-8:end-8),'h')) %Single digit epochs
        epoch = str2num(filename(end-7:end-7));
    elseif(strcmp(filename(end-9:end-9),'h')) % Double digit epochs
        epoch = str2num(filename(end-8:end-7));
    else % Triple digit epochs
        epoch = str2num(filename(end-9:end-7));
    end
    
    
    if(epoch > maxepoch)
        maxepoch = epoch;
    end
    
    

    end
end
        
        
  if(maxepoch == -1)
      fprintf('get_highest_epoch: No saved models for this layer (%d) present.',layer);
  end
    
    
    
end