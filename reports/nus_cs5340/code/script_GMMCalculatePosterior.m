% ------- script_GMMCalculatePosterior.m -------
% This code is stem from the function code fn_ITSBNImageSegm.m.
% Basically, this code is just a script (macro) cropped from the main
% function. The reason to separate this code from the main functin is
% to improve the readability of the main code. In all cases, you can
% copy and past this code back to the main function without any risk.


% #########################################################################
% ############ Tree Structure Image Segmentation ########################
% ========================================================================
% Mohammad Akbari, NGS,
% Shahab Ensafi, Soc,
% Fu Jie, NGS,
% National University of Singapore
% {Akbari, shahab.ensafi, jie.fu} @nus.edu.sg
% Thanks from Li Cheng, Kittipat Kampa and Matthew Zeiler
% ########################################################################


    % =================== classification for GMM =========================
    % In this section, we classify each superpixel/node independently using GMM.
    % The posterior at each node is used as a soft evidence in H1.
    % Later on, we will will use the mean and covariance in this process as
    % the intial value for ITSBN.
    % ====================================================================
    if iteration > 1
        pw = ones(1,C{1+1,2})/C{1+1,2};
        gmm_obj = gmdistribution(mu_c,Sigma_c,pw);
    end

    gmm_posterior = posterior(gmm_obj,Y);
    [max_postr, class_result] = max(gmm_posterior,[],2);
    
    
    
    % ============== plot the initial clustering using GMM ===============
    % ====================================================================
    figure(fig_GMM_feature_space);
    title('GMM clustering in feature space'); hold on; daspect([1 1 1]);
    color_cls = ['g','r','b','m','k'];
    for c = 1:C{1+1,2}
        class_index = class_result == c;
        d_plot = Y(class_index,:);
        color_index = mod(c,length(color_cls))+1;
        if iteration == 1
            plot(d_plot(:,1),d_plot(:,2),'*','Color',color_cls(color_index));
        end
        plot(mu_c(c,1),mu_c(c,2),'x','Color',color_cls(color_index));
        h = plotGaussianContour2D(mu_c(c,:)',Sigma_c(:,:,c),1,color_cls(color_index),300);
    end
%     print('-depsc','-r200',['gmm_result_',num2str(exp_number),'.eps']);
%     print('-djpeg','-r100',['gmm_result_',num2str(exp_number),'.jpg']);
    %     h = gcf; saveas(h,['gmm_fit_',num2str(exp_number),'.fig'])
    
    % ============== plot the GMM segmentation output image ==============
    % ====================================================================
    if iteration == 1
        %##################################################
        figure(fig_GMM_segmentation); daspect([1 1 1]); 
        [I_overlay] = fn_displaySegmentationResult(class_result, Y_index', node_label_hier{1,3}, img_RGB);
        % imagesc(img_cluster_result_gmm); daspect([1 1 1]);
        imagesc(I_overlay); daspect([1 1 1]); title('segmentation result using GMM');
        %     print('-depsc','-r200',['gmm_clustering_',num2str(exp_number),'.eps']);
        print('-djpeg','-r100',['gmm_clustering_',num2str(exp_number),'.jpg']);
        %     h = gcf; saveas(h,['gmm_clustering_',num2str(exp_number),'.fig'])
        movefile(['./gmm_clustering_',num2str(exp_number),'.jpg'], ['./',imagename]);
        
        
        % ======= save to segs (for evaluation) =========
        [seg_result] = fn_displaySegmentationResult(class_result, Y_index', node_label_hier{1,3});
        segs = cell(1,1);
        segs{1,1} = seg_result;
        save(imagename,'segs');
        %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        %@
        %movefile(['./',imagename,'.mat'], ['D:\']);
        
    end
    % ====================================================================