%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Extracts the epoch number and layer number from the path string to a file.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief extract_epoch_layer.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief extract_epoch_layer.m
%
% @param filename a path to a file name from which you want to extract the epoch
% and layer information
% @retval epoch the extracted epoch (numeric)
% @retval layer the extracted layer (numeric)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [epoch,layer] = extract_epoch_layer(filename)
% Filename is a path to the file that you want to extract the epoch and layer
% from the name.

% Remove the .mat extension.
filename = remove_dot_mat(filename);

% Remove the 'epoch' part.
filename = filename(6:end);

if(strcmp(filename(2),'_'))
    epoch = str2num(filename(1));
    filename = filename(3:end);
elseif(strcmp(filename(3),'_'))
    epoch = str2num(filename(1:2));
    filename = filename(4:end);
elseif(strcmp(filename(4),'_'))
    epoch = str2num(filename(1:3));
    filename = filename(5:end);
else
    epoch = 0;
end

filename = filename(6:end);
layer = str2num(filename);
% get_folder(filename)




end