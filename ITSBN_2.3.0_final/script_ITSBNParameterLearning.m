    % --- script_ITSBNParameterLearning -----
    % This code is stem from the function code fn_ITSBNImageSegm.m.
    % Basically, this code is just a script (macro) cropped from the main
    % function. The reason to separate this code from the main functin is
    % to improve the readability of the main code. In all cases, you can
    % copy and past this code back to the main function without any risk.
    
    
    %=================== PARAMETER LEARNING for ITSBN =====================
    % In this section, we learn the parameters mu_c, Sigma_c and phi_lvu
    % in order to update the ITSBN. The update equations are not very stable,
    % so I add some regularizers to them. 
    % =====================================================================
    pxi_g_evid = zeros(length(H{1+1,2}),C{1+1,2});
    tmp_cnt = 1;
    for i = H{1+1,2} % index of each node in H1
        pxi_g_evid_tmp = marginal_nodes(engine_bnet2,i);
        pxi_g_evid(tmp_cnt,:) = pxi_g_evid_tmp.T';
        tmp_cnt = tmp_cnt + 1;
    end
    
    % --- mu ----
    tmp_cnt = 1;
    mu_c_hat = zeros(D,C{1+1,2});
    for i = H{1+1,2} % index of each node in H1
        mu_c_hat = mu_c_hat + repmat( Y(tmp_cnt,:)', 1, C{1+1,2}) .* repmat( pxi_g_evid(tmp_cnt,:), D, 1);
        tmp_cnt = tmp_cnt + 1;
    end
    mu_c_new = ( mu_c_hat ./ repmat( sum( pxi_g_evid, 1), D, 1) )';
    
    % --- Sigma ---
    Sigma_c_hat = zeros(D,D,C{1+1,2});
    for c = 1:C{1+1,2}
        tmp_cnt = 1;
        for i = H{1+1,2} % index of each node in H1
            tmp1 = Y(tmp_cnt,:)-mu_c_new(c,:);
            Sigma_c_hat(:,:,c) = Sigma_c_hat(:,:,c) + pxi_g_evid(tmp_cnt,c) * (tmp1' * tmp1);
            tmp_cnt = tmp_cnt + 1;
        end
    end
    % normalize each Sigma_c
    tmp2 = sum( pxi_g_evid, 1);
    for c = 1:C{1+1,2}
        Sigma_tmp = Sigma_c_hat(:,:,c)/tmp2(1,c);
        if det(Sigma_tmp) < 1e-7
            Sigma_tmp = Sigma_tmp + Sigma_c_loading_factor * eye(D); % 1e-3 is for factor loading
        end
        Sigma_c_hat(:,:,c) = Sigma_tmp;
    end
    Sigma_c_new = Sigma_c_hat;
    
    
    % --- phi_l ----
    % for level 1 to L-2
    CPT_new = CPT;
    for l = 1:(L-2)
        j_index = H{l+1,2};
        CPT_new{l+1,2} = zeros(size(CPT_new{l+1,2})); % CPT_new{l+1,2}*0;
        for j = j_index
            i = find(Z(j,:)==1); % i_index
            pxjxi_g_evid_tmp = marginal_nodes(engine_bnet2,[j i]);
            pxjxi_g_evid = pxjxi_g_evid_tmp.T;
            pxjxi_g_evid = pxjxi_g_evid +added_message_stabilizer; % for stability
%             % ---- for test ------
%             if l == 4
%                 disp(['j= ',num2str(j)]);
%                 pxjxi_g_evid for test only
%             end
%             % --------------------
            % CPT_new{l+1,2} = CPT_new{l+1,2} + pxjxi_g_evid; % original...vulnerable to wiped-out CPT
            % CPT_new{l+1,2} = CPT_new{l+1,2} + pxjxi_g_evid + added_diag_cpt * eye(C); % tweak by adding some diagonal terms
            CPT_new{l+1,2} = CPT_new{l+1,2} + pxjxi_g_evid + added_diag_cpt * fn_pseudoDiag(C{l +1,2},C{l+1 +1,2}); % tweak by adding some diagonal terms
        end
        % renormalize the CPT
        CPT_new{l+1,2} = CPT_new{l+1,2}./repmat( sum(CPT_new{l+1,2}, 1)+ added_renormalize_cpt_factor, C{l +1,2}, 1); % add small number to prevent NaN
%         CPT_new{l+1,2} = CPT_new{l+1,2}./repmat( sum(CPT_new{l+1,2}, 1), C, 1); % original ..vulnerable to NaN
    end
    % for level L-1 (root)
    pxjxi_g_evid_tmp = marginal_nodes(engine_bnet2,[1]);
    pxjxi_g_evid = pxjxi_g_evid_tmp.T;
    pxjxi_g_evid = pxjxi_g_evid +added_message_stabilizer; % for stability
    CPT_new{L-1+1,2} = pxjxi_g_evid';
    % renormalize the CPT
    CPT_new{L-1+1,2} = CPT_new{L-1+1,2}/sum(CPT_new{L-1+1,2}, 2);
    
    % backup old parameters
    CPT_old = CPT; 
    mu_c_old = mu_c; 
    Sigma_c_old = Sigma_c; 
    
    % update parameters
    CPT = CPT_new; 
    mu_c = mu_c_new; 
    Sigma_c = Sigma_c_new;