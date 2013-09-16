function zigzag_index = fn_zigzagPatternGenerator(Nrow, Ncol)
% % For example
% Nrow = 4;
% Ncol = 3; 
% The matrix will be
%      1     5     9
%      2     6    10
%      3     7    11
%      4     8    12
% 
% zigzag_index =  
% 
% 1     5     2     3     6     9    10     7     4     8    11    12


S= [Ncol Nrow]; %[ps ps]; %(col row)
i=fliplr(spdiags(fliplr(reshape(1:prod(S),S)')));
i(:,1:2:end)=flipud(i(:,1:2:end));
tmp_index = i(i~=0)';

% map the index back to our orientation
tmp1 = 1:Ncol*Nrow;
tmp_A = reshape(tmp1,Ncol,[])';
tmp_B = reshape(tmp1,Nrow,[]);
tmp2 = [tmp_A(:) tmp_B(:)];
f_mapping = sortrows(tmp2,1);
zigzag_index = f_mapping(tmp_index,2)';