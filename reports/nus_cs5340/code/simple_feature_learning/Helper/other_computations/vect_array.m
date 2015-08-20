%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Vectorizes the input array as in(:).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @other_comp_file @copybrief vect_array.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief vect_arry.m
%
% @param in any sized array.
% @retval in the input(:) vectorized.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [in] = vect_array(in)

% Useful when you want to save memory and need to vectorize 4-D matrices.
in = in(:);

end