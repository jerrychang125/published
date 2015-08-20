function new_segm = fn_regularizeIndices(segm)

% ########################################################################
% Since the indices produced by Mori's super pixel are occasionally out of
% order, for instance, 1, 2, 3, 5, 6, 8, as opposed to 1, 2, 3, 4, 5, 6. 
% This confusion is the cause for the algorithm not being robust. 
% So, this code reorders the indices in ascending order. 
% ========================================================================

new_segm = segm;
label_list = unique(segm(:),'rows');
cnt_idx = 1;
for j = label_list'
    new_segm(segm==j) = cnt_idx;
    cnt_idx = cnt_idx + 1;
end