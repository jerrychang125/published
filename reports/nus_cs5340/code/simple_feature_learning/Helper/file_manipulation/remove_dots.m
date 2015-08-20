%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Removes the \c dot and \c dotdot directories from a list of directories.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief remove_dots.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief remove_dots.m
%
% @param input a list of directies (returne by dir() for example).
% @retval output the same list without the . and .. directories.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output] = remove_dots(input)
% Removes the . and .. directories from the input structure array.

output = input(find(~cellfun(@cmp_dots,{input(:).name})));
    
    function [result] = cmp_dots(input)
        if(strcmp(input,'.') || strcmp(input,'..'))
            result = 1;
        else
            result = 0;
        end
    end



end