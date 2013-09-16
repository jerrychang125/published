%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Loads each file in directory, removes the saved z feature map
% variable within the file, and resaves it as epoch##_layer##_noz.mat.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief remove_zs.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief remove_zs.m
%
% @param path a directory containing .mat files with z features maps in them.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = remove_zs(path)

% Get list of files in directory.
dirCell = dir(path);


for file=1:length(dirCell)
    if(dirCell(file).isdir==0) % Only files
       pathtofile = strcat(path,'/',dirCell(file).name)
       
       
       load(pathtofile);
       
       pathtofile = remove_dot_mat(pathtofile);
       
       save(strcat(pathtofile,'_noz.mat'),'F','z0','y','model');
        
       
        
    end


end