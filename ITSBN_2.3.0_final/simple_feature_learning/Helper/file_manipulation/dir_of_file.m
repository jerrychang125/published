%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Returns the directory in which the input filepath resides.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief dir_of_file.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief dir_of_file.m
%
% @param path path to a file.
% @retval path the path to the directoy the file is in.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [path] = dir_of_file(path)
matpath = str2mat(path);
% Make sure it is not just a direcotry.
if(strcmp(matpath(end:end),'/')==0)
    
    i=length(matpath);
    while(strcmp('/',matpath(i))==0)
        i=i-1;
    end
    % Now i is the index of the last '/'
    path = matpath(1:i);
    
end

end