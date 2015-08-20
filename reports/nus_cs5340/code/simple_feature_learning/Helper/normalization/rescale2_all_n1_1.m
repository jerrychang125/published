%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Very similar to \c svm_rescale2.m but I'm not exactly sure what range this
% guarantees (if any). This was provided by Graham Taylor.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @normalization_file @copybrief rescale2_all_n1_1.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief rescale2_all_n1_1.m
%
% @param data This is the data you want to scale.
% @retval data This is the scaled output.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data] = rescale2_all_n1_1(data)

data = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

end