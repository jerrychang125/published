% #########################################################################
% This code gets important information of the output from ITSBN, for
% instance, marginal posterior of each node in ITSBN, maximum posterior
% marginal (MPM).
% posterior_marg is a Lx4 cell array. 
% column#1: the level number (l)
% column#2: containing the posterior marginal p(xj|evid) of node xj in the
% level Hl.
% column#3: maximum posterior marginal class 
% column#4: value of maximum posterior marginal 
% #########################################################################

    % ========== calculate the posterior marginal p(xj|evid) ================
    posterior_marg = cell(L,4);
    for l = (L-1):-1:1
        pxj_Hl = zeros(C{l+1,2},length(H{l+1,2})); % marg of nodes in level Hl
        tmp_cnt = 1;
        for j = H{l+1,2}
            % calculate the posterior marginal p(xj|evid)
            pxj_g_evid = marginal_nodes(engine_bnet2,j);
            pxj_Hl(:,tmp_cnt) = pxj_g_evid.T;
            tmp_cnt = tmp_cnt + 1;
        end
        posterior_marg{l+1,1} = l; % level Hl
        posterior_marg{l+1,2} = pxj_Hl; % marg of nodes in level Hl
        [mpm_value, mpm_class] = max(pxj_Hl,[],1);
        posterior_marg{l+1,3} = mpm_class;
        posterior_marg{l+1,4} = mpm_value;
    end
    
    % ============== SEGMENTATION IMAGE IN EACH SCALE =====================
    % segs is a cell array L-2 x 1
    % Each row of segs contain the segmentation result with the same size
    % as the original image. The segmentation is valid from level L-2 down
    % to leve 1. Note that we don't store the root level here as it is
    % trivial.
    % =====================================================================
    
    segs = cell(L-2,1);
    for l = (L-2):-1:1
        [seg_result] = fn_displaySegmentationResult(posterior_marg{l+1,3}, H{l+1,2}, node_label_hier{l,3});
        segs{l,1} = seg_result;
        figure; imagesc(seg_result);
    end
    





