function detected_contour = fn_segment2boundary(segm_output, figure_toggle)

% This code convert a segmentation output into a boundary image by
% differentiating the segments.

region_seg = double(segm_output); % load the detected boundary
diff_row = diff(region_seg,1,1).^2;
diff_col = diff(region_seg,1,2).^2;
detected_contour = region_seg*0;
detected_contour(1:end-1,1:end) = diff_row;
detected_contour(1:end,1:end-1) = detected_contour(1:end,1:end-1) + diff_col;
detected_contour(detected_contour == 0) = 0;
detected_contour(detected_contour ~= 0) = 1;

if strcmp(figure_toggle,'on')
    % plot the contour of the results
    nValues = 128;  %# The number of unique values in the colormap
    map = [linspace(1,0,nValues)' linspace(1,0,nValues)' linspace(1,0,nValues)'];  %'# 128-by-3 colormap
    figure; imagesc(region_seg);
    figure; imagesc(detected_contour);
    daspect([1 1 1]); colormap(map);
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
end