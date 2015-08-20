%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Recurses from the starting path to all subdirectories and removes unwanted
% files and folders. The unwanted ones are those created by experiments that
% were stopped before saving any files into the folders. Any empty Run_...
% directories will be renamed to Empty_Run. Any folder containing only Empty_Run
% directories will be removed. All Empty_Run's after a Run will be removed.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief clean_results_dir.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief clean_results_dir.m
%
% @param inpath this can either be a single path (string) or the result of 
% getsubdirs() in which case this function will process all such directories.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = clean_results_dir(inpath)

numpaths = 0; % Initialize to something different so initially the while loop is satisfied.

while(length(getsubdirs(inpath))~=numpaths)
    
    % Get all the subdirectories
    pathCell = getsubdirs(inpath);
    numpaths = length(pathCell); % Save the previous number of paths.
    
    % For each path
    for i=2:length(pathCell)
        path = pathCell{i};
        
        letter = length(path);
        while(strcmp(path(letter),'/')==0)
            letter = letter-1;
        end
        
        
        % folder has to be a Run_## folder.
        if(strcmp(path(letter+1:letter+4),'Run_'))
            % See if there are any files in this folder.
            subdir = dir(path);
            file_exists = 0;
            for sub=1:length(subdir)
                if(subdir(sub).isdir==0)
                    file_exists = 1;
                end
            end
            
            if(~file_exists) % This was an empty Run directory.
                % Rename the directory to Empty_Run_...
                movefile(path,strcat(path(1:letter),'Empty_Run_',path(letter+5:end)));
                fprintf(strcat('Renamed:                            ',path,' to:\n        ',strcat(path(1:letter),'Empty_Run_',path(letter+5:end)),'\n'));
                
                numpaths = numpaths+1; % This being empty could mean it's the only folder left.
            end
        elseif(length(path) >= letter+10 && strcmp(path(letter+1:letter+10),'Empty_Run_')) % Don't remove these unless they are the only onces in their parent's directory.
            % If they are not the only one's in their parent's directory
            % then they are placeholders so can't be deleted.
            if(length(dir(path(1:letter)))==3) % If this was the only run in this parent directory.
                % This is compared to 3 since . and .. are always there.
                fprintf(strcat('Removed Parent of single Empty_Run: ',path(1:letter),'\n'));
                rmdir(path(1:letter),'s'); % Remove that parent directory and all it's paths.
                numpaths = numpaths + 1; % alter this.
            else  % Want to check if this Empty_Run is followed by any real Run_'s. If it is not then it is safe to delete it.
                subdirCell = dir(path(1:letter)); % Get all files in this directory.
                
                % We know all subdirectories here are Run_## or Empty_Run_##
                % Therefore the last one in subdirCell is the larger Run_#
                if(strcmp(subdirCell(end).name(1:4),'Run_'))
                    maxrun = str2num(subdirCell(end).name(5:end));
                elseif(length(subdirCell(end).name)>= 10 && strcmp(subdirCell(end).name(1:10),'Empty_Run_')) % Only empty runs.
                    maxrun = -1;
                else
                    maxrun = inf;
                end
                % If the current empty run is higher in number than maxrun then delete it.
                if(str2num(path(letter+11:end))>maxrun)
                    fprintf(strcat('Removed Empty_Run_:',path,'\n higher than the max Run_: ',path(1:letter),subdirCell(end).name,'\n'));
                    rmdir(path,'s');
                end
            end
        else % Otherwise remove empty directories.
            subdir = dir(path);
            if(length(subdir)==2) % Only the . and .. directories
                rmdir(path)
                fprintf(strcat('Removed empty directory:            ',path,'\n'));
            end
        end
    end
end



end