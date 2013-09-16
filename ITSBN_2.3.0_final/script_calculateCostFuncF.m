% --------- script_calculateCostFuncF --------------
% This code is stem from the function code fn_ITSBNImageSegm.m.
% Basically, this code is just a script (macro) cropped from the main
% function. The reason to separate this code from the main functin is
% to improve the readability of the main code. In all cases, you can
% copy and past this code back to the main function without any risk.

    % =============== calculating the cost function F() =================
    % This section, we calculate the cost function F which is the
    % expectation of the complete log-likelihood. The value of F is used to
    % indicate when to stop the algorithm.
    % There are 2 terms involving in the cost function:
    % 1) term#1: log-likelihood term which is modeled using the multivariate
    % Gaussian. 
    % 2) term#2: The tree-structured prior term which involves image-class
    % label.
    % The 2 terms will be combined into the cost function F at the end
    % ===================================================================
    
    % ======== term#1 (Gaussian term) ========
    % This is the log-likelihood term
    % =======================================
    pxi_g_evid = zeros(length(H{1+1,2}),C{1+1,2});
    tmp_cnt = 1;
    for i = H{1+1,2} % index of each node in H1
        pxi_g_evid_tmp = marginal_nodes(engine_bnet2,i);
        pxi_g_evid(tmp_cnt,:) = pxi_g_evid_tmp.T';
        tmp_cnt = tmp_cnt + 1;
    end
    logNorm = zeros(length(H{1+1,2}),C{1+1,2});
    for c = 1:C{1+1,2}
        logNorm(:,c) = log( mvnpdf( Y, mu_c(c,:), Lambda_c(:,:,c)^-1) );
    end
    % replace -Inf in logNorm with the minimum value
    tmp1 = min( logNorm(~isinf(logNorm)),[], 1); % non-inf minimum value of log normal
    logNorm(isinf(logNorm)) = tmp1;
    term1 = sum( sum( logNorm .* pxi_g_evid , 2), 1);
    
    
    % ======== term#2 (discrete) =========
    % This is the tree-structured prior term.
    % Level 1 to L-2
    % =======================================
    term2 = 0;
    for l = 1:(L-2)
        j_index = H{l+1,2};
        i_index = H{l+1+1,2};
        for j = j_index
            for i = i_index
                if Z(j,i) ~= 0
                    pxjxi_g_evid_tmp = marginal_nodes(engine_bnet2,[j i]);
                    pxjxi_g_evid = pxjxi_g_evid_tmp.T;
                    tmp1 = sum( sum( pxjxi_g_evid .* log(CPT{l+1,2} + 1e-4 ), 1), 2); % +1e-4 was added for stability
                    term2 = term2 + tmp1;
                end
            end
        end
    end
    % Level L-1 (root)
    pxjxi_g_evid_tmp = marginal_nodes(engine_bnet2,[1]);
    pxjxi_g_evid = pxjxi_g_evid_tmp.T;
    pxjxi_g_evid = pxjxi_g_evid + added_message_stabilizer; % for stability
    tmp1 = sum( pxjxi_g_evid' .* log(CPT{L-1+1,2}), 2);
    term2 = term2 + tmp1;
    
    
    % ======= the lower bound F(theta) ===========
    % Combine term#1 and term#2
    % =======================================
    F_new = term1 + term2;
    F_old = F; F = F_new;
    if isnan(F_new)
        disp('F_new = NaN!!!');
    end
    figure(fig_cost_function); hold on; title('lower bound F(q)');
    plot(iteration, F, 'k.-');
    xlabel('iteration'); ylabel('F(q)');