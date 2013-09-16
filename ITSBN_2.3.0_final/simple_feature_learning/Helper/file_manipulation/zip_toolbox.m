%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Packages a toolbox with the desired files/folders and geneates the
% documentation from the code using doxygen.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief zip_toolbox.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief zip_toolbox.m
%
% @param zip_file_name the name of the zip file you want to make (not path to it
% though). 
% @param documentation_title the title used by doxygen to make the
% documentation.
% @param list_of_includes a cell array of paths to files or folders (which will
% be recursively added) to be added to the zip file.
% @param start_path_of_includes [optional] this is used by zip() to ensure the
% list_of_includes are put into the zip with their entire path (removes this
% portion of the path). The list_of_includse must start from this path though.
% Defaults to '.'.
% @param output_dir [optional] where to put the zip file. Defaults to '.'.
% @param doxyfile_path [optional] string to a desired Doxyfile to use with
% doxygen. Defaults to '~/work/docs/Doxyfile' if that exists or uses the default
% doxygen one.
% @retval path removes the \c .mat extension if one exists.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = zip_toolbox(zip_file_name,documentation_title,list_of_includes,start_path_of_includes,output_dir,doxyfile_path)

% Put the .zip in the current directory if none is specified.
if(nargin<6)
    output_dir = '.';
end
if(nargin<7)
    doxyfile_path = '~/work/docs/Doxyfile';
    if(~exist(doxyfile_path,'file'))
        doxyfile_path = '';
    end
end

cd(output_dir);

% Make the zip file of all code.
zip(zip_file_name,...
    list_of_includes,start_path_of_includes)

% Keep track of where you were.
saved_pwd = pwd;
[a,zipname,c,d] = fileparts(zip_file_name);
new_directory = strcat('./temp/',zipname,'/');
mkdir2(new_directory)

% Unzip the code into that directory.
cd(new_directory)
unzip(strcat('../../',zip_file_name))

% Remove all .svn and .DS_Store things in temp directory.
A = getsubdirs('.');
for i=size(A,1):-1:1
    if(~isempty(strfind(A{i},'.svn')))
        fprintf('Removed a .svn directory at: %s\n',A{i});
        rmdir(A{i},'s')
    end
    
    [subfolders,subfiles] = split_folders_files(dir(A{i}));
    % Remove the .DS_Store files.
    for j=1:size(subfiles,1)
        
        if(strcmp(subfiles(j).name,'.DS_Store'))
            blah = strcat(A{i},'/',subfiles(j).name);
            fprintf('Removed a .Ds_Store file at: %s\n',blah);
            delete(blah)
        end
        
        % Delete all ~ ending files.
        if(strcmp(subfiles(j).name(end),'~'))
            blah = strcat(A{i},'/',subfiles(j).name);
            fprintf('Removed a ~ ending file at: %s\n',blah);
            delete(blah)
        end 
        
        % Delete all .mex* ending files.
        if(~isempty(strfind(subfiles(j).name,'.mex')))
            blah = strcat(A{i},'/',subfiles(j).name);
            fprintf('Removed a .mex* ending file at: %s\n',blah);
            delete(blah)
        end  
    end
end
%  rmdir('.svn','s')
B = strcat({'! (cat '},doxyfile_path,' ; echo "OUTPUT_DIRECTORY=',...
    pwd,'"; echo "INPUT=',pwd,'"; echo "GENERATE_MAN=NO";',...
    'echo "PROJECT_NAME=',documentation_title,'";',...
    'echo "HTML_OUTPUT=Documentation";) | /home/zeiler/packages/doxygen-1.6.3/bin/doxygen -');
B{1}
% Move back up and geneate the documentation.
eval(B{1});



% Go back to start, rezip, and remove temp.
cd(saved_pwd)
% Zip the contents of the temp directory.
zip(zip_file_name,...
    '.','./temp')


rmdir('temp','s')

