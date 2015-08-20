%% ======================= script_plottingResult ===================
% This script is used in fn_ITSBNImageSegm.m
% This script plots CPT, marginal distribution at each node in the network,
% clustering result in feature space, segmentation output of an image.
%=====================================================================   




    % ============== plot CPT =================================
    figure(fig_CPT_each_level);
    tmp_cnt = 1;
    for l = (L-1):-1:1
        % left panel (old CPT)
        subplot(L-1, 2, 2*tmp_cnt-1); imagesc(CPT_old{l+1,2} );
        colormap('gray'); caxis([0 1]); colorbar;
        set(gca, 'XTick', [1:C{l+1,2}],'XTickLabel',1:C{l+1,2});
        ylabel(['H^',num2str(l)]);
        daspect([1 1 1]);
        % right panel (new CPT)
        subplot(L-1, 2, 2*tmp_cnt); imagesc(CPT_new{l+1,2} );
        colormap('gray'); caxis([0 1]); colorbar;
        set(gca, 'XTick', [1:C{l+1,2}],'XTickLabel',1:C{l+1,2});
        %         ylabel(['H^',num2str(l)]);
        daspect([1 1 1]);
        tmp_cnt = tmp_cnt + 1;
    end
%     print('-depsc','-r200',['CPT_',num2str(exp_number),'.eps']);
    print('-djpeg','-r100',['CPT_',num2str(exp_number),'_itt',num2str(iteration),'.jpg']);
    % h = gcf; saveas(h,['CPT_',num2str(exp_number),'.fig'])
    movefile(['./CPT_',num2str(exp_number),'_itt',num2str(iteration),'.jpg'], ['./',imagename]);
    
    
    
    
    % ========== plot the posterior marginal p(xj|evid) ================
    figure(fig_posterior_each_level); hold on; title('posterior distribution at each node');
    tmp_cnt2 = 1;
    for l = (L-1):-1:1
        pxj_Hl = zeros(C{l+1,2},length(H{l+1,2}));
        tmp_cnt = 1;
        for j = H{l+1,2}
            % calculate the posterior marginal p(xj|evid)
            pxj_g_evid = marginal_nodes(engine_bnet2,j);
            pxj_Hl(:,tmp_cnt) = pxj_g_evid.T;
            tmp_cnt = tmp_cnt + 1;
        end
        % plot the posterior marginal
        figure(fig_posterior_each_level); hold on; subplot(L+1, 1, tmp_cnt2);
        imagesc(pxj_Hl); colormap('gray'); caxis([0 1]); colorbar;
        if length(H{l+1,2}) <= 32 % when there are too many node, ticks are too many
            set(gca, 'XTick', [1:length(H{l+1,2})],'XTickLabel',H{l+1,2});
        end
        ylabel(['H^',num2str(l)]);
        tmp_cnt2 = tmp_cnt2 + 1;
    end
    subplot(L+1, 1, tmp_cnt2); imagesc(gmm_posterior'); colormap('gray'); caxis([0 1]); colorbar;
%     print('-depsc','-r200',['marginal_posterior_',num2str(exp_number),'.eps']);
    print('-djpeg','-r100',['marginal_posterior_',num2str(exp_number),'_itt',num2str(iteration),'.jpg']);
    % h = gcf; saveas(h,['marginal_posterior_',num2str(exp_number),'.fig'])
    movefile(['./marginal_posterior_',num2str(exp_number),'_itt',num2str(iteration),'.jpg'], ['./',imagename]);
    
    
    
    
    
    % =============== plot the clustering result ======================
    pxj_H1 = zeros(C{1+1,2},length(H{1+1,2}));
    tmp_cnt = 1;
    for j = H{1+1,2}
        % calculate the posterior marginal p(xj|evid)
        pxj_g_evid = marginal_nodes(engine_bnet2,j);
        pxj_H1(:,tmp_cnt) = pxj_g_evid.T;
        tmp_cnt = tmp_cnt + 1;
    end
    
    % -- classification ---
    [max_postr, class_result] = max(pxj_H1,[],1);
    
    
    % plot the classification
    figure(fig_GMM_feature_space); hold on; daspect([1 1 1]); title('GMM clustering in feature space');
    for c = 1:C{1+1,2}
        class_index = class_result == c;
        d_plot = Y(class_index,:);
        color_index = mod(c,length(color_cls))+1;
        plot(d_plot(:,1),d_plot(:,2),'o','Color',color_cls(color_index));
    end
%     print('-depsc','-r200',['classification result_',num2str(exp_number),'.eps']);
    print('-djpeg','-r100',['classification result_',num2str(exp_number),'_itt',num2str(iteration),'.jpg']);
    % h = gcf; saveas(h,['classification result_',num2str(exp_number),'.fig'])
    movefile(['./classification result_',num2str(exp_number),'_itt',num2str(iteration),'.jpg'], ['./',imagename]);
    
    
    
    
    % ====== plot the clustering on the image ==========
    figure(fig_TSBN_segmentation); daspect([1 1 1]); 
        [I_overlay] = fn_displaySegmentationResult(class_result, Y_index', node_label_hier{1,3}, img_RGB);
        imagesc(I_overlay); daspect([1 1 1]); title('ITSBN segmentation result');
    %     print('-depsc','-r200',['TSBN_clustering_',num2str(exp_number),'.eps']);
    print('-djpeg','-r100',['ITSBN_clustering_',num2str(exp_number),'_itt',num2str(iteration),'.jpg']);
%     h = gcf; saveas(h,['TSBN_clustering_',num2str(exp_number),'.fig'])
    movefile(['./ITSBN_clustering_',num2str(exp_number),'_itt',num2str(iteration),'.jpg'], ['./',imagename]);
    iteration = iteration + 1;