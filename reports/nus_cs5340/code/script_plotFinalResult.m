% script_plotFinalResult
% ########################################################################
% ########################## PLOT FINAL RESULT ############################
% Here, we will plot the final clusters in the feature space (2D though),
% and also the cost function F too.
% =========================================================================
% ########################################################################


% ================ plot the FINAL classification result ==================
figure(fig_final_GMM_feature_space); hold on; daspect([1 1 1]); title('final classification result');
plot(Y(:,1),Y(:,2),'.-','Color',0.7*[1 1 1]); daspect([1 1 1]);
for c = 1:C{1+1,2}
    class_index = class_result == c;
    d_plot = Y(class_index,:);
    color_index = mod(c,length(color_cls))+1;
    plot(d_plot(:,1),d_plot(:,2),'o','Color',color_cls(color_index));
    plot(mu_c(c,1),mu_c(c,2),'x','Color',color_cls(color_index));
    h = plotGaussianContour2D(mu_c(c,:)',Sigma_c(:,:,c),1,color_cls(color_index),300);
end
% print('-depsc','-r200',['final_classification result_',num2str(exp_number),'.eps']);
print('-djpeg','-r100',['final_classification result_',num2str(exp_number),'.jpg']);
movefile(['./final_classification result_',num2str(exp_number),'.jpg'], ['./',imagename]);




% =================== print the cost function curve =======================
figure(fig_cost_function);
%     print('-depsc','-r200',['cost_function_',num2str(exp_number),'.eps']);
print('-djpeg','-r100',['cost_function_',num2str(exp_number),'.jpg']);
%     h = gcf; saveas(h,['cost_function_',num2str(exp_number),'.fig'])
movefile(['./cost_function_',num2str(exp_number),'.jpg'], ['./',imagename]);


% ================ PLOT THE CONTOUR ===================================
acc_detected_contour = 0*segs{1,1};
for l = (L-2):-1:1
    segm_output = segs{l,1};
    detected_contour = fn_segment2boundary(segm_output, 'off');
    acc_detected_contour = detected_contour + acc_detected_contour;
end
% plot the figure
figure(fig_ITSBN_boundary); imagesc(acc_detected_contour);
daspect([1 1 1]);
nValues = 128;  %# The number of unique values in the colormap
map = [linspace(1,0,nValues)' linspace(1,0,nValues)' linspace(1,0,nValues)'];  %'# 128-by-3 colormap
colormap(jet); %colormap(map);
colorbar;
set(gca,'xtick',[]);
set(gca,'ytick',[]);

%     print('-depsc','-r200',['boundary_',num2str(exp_number),'.eps']);
print('-djpeg','-r100',['boundary_',num2str(exp_number),'.jpg']);
%     h = gcf; saveas(h,['boundary_',num2str(exp_number),'.fig'])
movefile(['./boundary_',num2str(exp_number),'.jpg'], ['./',imagename]);

% =============== PLOT SEGMENTATION RESULT ============================
figure(fig_ITSBN_display);
cnt_plot = 1;
for l = (L-2):-1:1
    segm_output = segs{l,1};
    Irgb_avg = fn_segment2avgcolor(img_RGB,segm_output);
    subplot(2,2,cnt_plot); imshow(Irgb_avg);
    cnt_plot = cnt_plot + 1;
end
%     print('-depsc','-r200',['display_',num2str(exp_number),'.eps']);
print('-djpeg','-r100',['display_',num2str(exp_number),'.jpg']);
%     h = gcf; saveas(h,['display_',num2str(exp_number),'.fig'])
movefile(['./display_',num2str(exp_number),'.jpg'], ['./',imagename]);
disp(['The results (tree structure) are saved in the folder /',imagename]);