function [] = makeTreeFromLabel(imagename)

% ===== user input =====
% imagename = '113016'; % house
% ======================

[s,mess,messid] = mkdir(['./',imagename]);
disp(mess);
addpath(['./',imagename]);
load([imagename,'_img_label_hier']);

n_level = size(img_label_hier,1); % n_level: # of level from H(L-2) to H(1)
K = 1; % K: number of nodes so far in the network including the root node (node#1)


H = cell(n_level+2+1,2); % store index of each level
% The reason why we have n_level+2+1 levels prepare because
% n_level is from H(L-2) to H(1)
% +2 is for H(L-1) and H(0)
% +1 is for the phantom node 0

% level n_level+1 is the index of root level, the number of superpixel is of course 1
H{n_level+1+1,1} = n_level+1; 
H{n_level+1+1,2} = 1;

    
num_region_each_level = zeros(n_level,1); % # of superpixel in each level
for level = 1:n_level
    K = K + img_label_hier{level,2};
    num_region_each_level(level,1) = img_label_hier{level,2};
end

% ########################################################################
% ########################################################################
% ########################################################################
% # In this section we will assign the node label to the superpixelID
% ########################################################################
% ------------ version 2.0.2 ------------------------
% Now K is the total number of superpixels from root H(L-1) to H1
% So, the actual total number os superpixels is K+num_region_each_level(1,1)
% In addition, we have 1 phantom node for roots (node0), so we have one
% more for column K +num_region_each_level(1,1) +1
% Z = zeros(K +num_region_each_level(1,1) , K +num_region_each_level(1,1) +1); % version 2.0.2 structure matrix child x parent
%
% ------------ version 2.1.0 ------------------------
% In version 2.1.0, we will NOT use H(0) because we know that the hidden
% variable can be softevidence in BNT. Therefore, our structure is composed
% of K nodes, where K is the total number of superpixels from root H(L-1)
% to H1
Z = zeros(K , K +1); % version 2.1.0 structure matrix child x parent
% ========================================================================
node_index = 1; % start with index of the root, node_index also serves as the last node index so far when adding up the network too
node_label_hier = img_label_hier;

% ################ REORDER INDICES ##################
for level = n_level:-1:1
    img_label_hier{level,3} = fn_regularizeIndices(img_label_hier{level,3});
end


% ################ MAKE ITSBN ######################
for level = n_level:-1:1 % We do it top-down style, from level H(L-2) down to H(1) 
    I_child_label = img_label_hier{level,3} +node_index;
    list_label = unique(I_child_label)'; % list of the node index in the current level
    H{level+1,1} = level; % level H[level]
    H{level+1,2} = list_label; % 
    if level == n_level % --- for level H(L-2)
        node_label_hier{level,3} = I_child_label;
        Z(list_label,1) = 1; % connect level H(L-2) to the root node in H(L-1)
    else % for other levels
        node_label_hier{level,3} = I_child_label;
        I_parent_label = node_label_hier{level+1,3}; % node index of the corresponding parent level (level above I_child_label)
        for label = list_label
            % --- connect child superpixel to its corresponding parent ----
            % - Here we overlap the parent level on the child level and the
            % parent whose index are overlapped with the child the most
            % will be connected to the child. 
            parent_region_index = I_parent_label(I_child_label == label);
            root_index = mode(parent_region_index,1); % root is from the mode
            if sum(Z(label,:)~=0,2) == 0
                % sometime there are more than 1 mode, and we will pick the
                % smaller number, so we use root_index(1)
                Z(label,root_index(1)) = 1; % connect the child to its corresponding parent
            else
                disp(['node ',num2str(label),'has more than 1 parent']);
            end
            % -------------------------------------------------------------
        end
    end
    % === update node_index, from this we will know how many nodes are use
    % to make the network so far. node_index is the last index in the
    % previous level
    
    % app#2
    % the offset of the index n level l is obtained from the max index in
    % the level l+1. This is more stable than the previous approach (app#1)
    node_index = max(I_child_label(:),[],1);
    
    % app#1
    % node_index = node_index + img_label_hier{level,2}; % This is not
    % stable because the superpixel indices are NOT always increasing by 1.
    % Sometimes, the indices skip, for example 1, 2, 3, ..., 47, 49, 50
    % which yields duplicate parents. 
    
end



% % ---- add the level H(0) <evidence level> -----
% evidence_label = img_label_hier{1,3} +node_index; % node index for level H(0)
% % evidence_label: is a matrix with the same size as the original image.
% % Each element is the corresponding node index for level H(0)
% I_child_label = unique(evidence_label)';
% H{0+1,1} = 0; % level root
% H{0+1,2} = I_child_label;
% % -- connect the child to the right-above parent --
% for i = I_child_label
%     Z(i,i-num_region_each_level(1,1)) = 1;
% end

% --- draw the network -----
figure; draw_graph(Z(:,1:end-1)');
figure; imagesc(node_label_hier{1,3});

% ---- add phantom parent H(L) -----
H{n_level+2+1,1} = n_level+2;
H{n_level+2+1,2} = K +num_region_each_level(1,1)+1;

% === At this point, we will have the complete structure of the Bayesian
% network read for BNT
% ########################################################################
% ########################################################################
% ########################################################################


save([imagename,'_package'], 'Z', 'n_level', 'H', 'node_label_hier'); % in 2.1.0 we don't use 'evidence_label',
% We also move 'Y', 'Y_index' to feature extraction routine

% move the output to the corresponding folder
movefile(['./',imagename,'_package.mat'], ['./',imagename]);
disp(['The results (tree structure) are saved in the folder /',imagename]);

% --- remove the path ----
rmpath(['./',imagename]);