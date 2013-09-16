function a = fn_pseudoDiag(r,c)
% #########################################################################
% This function produce a psudo diagonal matrix such that all the entries
% in the matrix are 0 except the entries crossed by the diagonal line
% between the top-left to the bottom-right are 1.
% For example
% fn_pseudoDiag(5,3) gives
%      1     0     0
%      1     1     0
%      0     1     0
%      0     1     1
%      0     0     1
% #########################################################################
% Kittipat (Bot) Kampa
% May 26, 2011
% #########################


w = [r -c];

[col_ind, row_ind] = meshgrid([0:1:c],[0:1:r]);

coor = [col_ind(:), row_ind(:)]'; % x and y respectively
prod = w*coor;

upper = coor(:,prod<0);
lower = coor(:,prod>0);

upper_sub = upper + repmat([1 0]', 1, size(upper,2));
lower_sub = lower + repmat([0 1]', 1, size(upper,2));

common_sub = intersect(upper_sub',lower_sub','rows');

% convert from x-y to row-col
a = zeros(r,c);
a(sub2ind(size(a),common_sub(:,2),common_sub(:,1))) = 1;

% % ============= visualize the separation =================
% % uncomment when need to see the plot
% figure; hold on; daspect([1 1 1]);
% plot(upper(1,:),upper(2,:),'b*');
% plot(lower(1,:),lower(2,:),'ro');
% figure; imagesc(a); daspect([1 1 1]);
% % ========================================================
