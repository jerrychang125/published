%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Clears (delets) the last results directory (Run_##).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief clear_last.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief clear_last.m
%
% @param instring determines which struct to look into to find the path to the
% last save directory (the path you want to remove).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outpath] = clear_last(instring)
% function [model.fullsavepath] = clear_last()
%
% Clears the last results directory (Run_#/).
% This is loaded from

if(nargin<1)
    instring = 'guimodel';
end

switch instring
    case 'guimodel'
        fprintf('Attempting to delete: \n %s \n',guimodel.fullsavepath);
        outpath = guimodel.fullsavepath;
        if(exist(outpath,'dir'))
            rmdir(guimodel.fullsavepath,'s')
            fprintf('Success!\n')
        else
            outpath = 'Directory does not exist in clear_last';
            fprintf('%s \n',outpath);
        end
    case 'set_parameters'
        load('set_parameters.mat')
        fprintf('Attempting to delete: \n %s \n',model.fullsavepath);
        outpath = model.fullsavepath;
        if(exist(outpath,'dir'))
            rmdir(model.fullsavepath,'s')
            fprintf('Success!\n')            
        else
            outpath = 'Directory does not exist in clear_last';
            fprintf('%s \n',outpath);
            
        end
end
end