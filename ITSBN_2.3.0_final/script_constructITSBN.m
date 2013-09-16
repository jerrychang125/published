% --------- script_constructITSBN --------------
% This code is stem from the function code fn_ITSBNImageSegm.m.
% Basically, this code is just a script (macro) cropped from the main
% function. The reason to separate this code from the main functin is
% to improve the readability of the main code. In all cases, you can
% copy and past this code back to the main function without any risk.
    
    
    % ===================== CONSTRUCT ITSBN ==============================
    % In this section, we will construct a tree-structure bayesian networks
    % using BNT. The summary is following:
    % 1) We prepare and update the CPT of the ITSBN
    % 2) Prepare soft and hard evidence for the ITSBN
    % 3) Make an inference engine out of those prepared information
    % The inference engine will be used later when needing to calculate the
    % posterior marginal or the expectation of each node in the network
    % ================== put/update CPTs to the network ==================
    % # add CPT to nodes in level H(1) to H(L-2)
    % ====================================================================
    for l = 1:(L-2) % version 2.1.0. In version 2.0.2 we have H(0) so we use 0:(L-2)
        for j = H{l+1,2}
            bnet1.CPD{j} = tabular_CPD(bnet1, j, CPT{l+1,2});
        end
    end
    % ---- add CPT to the root node at the level H(L-1)
    bnet1.CPD{1} = tabular_CPD(bnet1, 1, CPT{L-1+1,2}); 
    % Make engine for inference
    tic;
    engine_bnet1 = jtree_inf_engine(bnet1);
    disp(['jtree_inf_engine ',num2str(toc),' sec']);

    
    % ================= MAKE INFERENCE ENGINE for ITSBN ==================
    % ======== this uses the fact that we have Y_index and Y, so we can put
    % each feature vector to the corresponding softevidence in every level
    % ====================================================================
    
    % prepare the soft and hard evidence
    hard_evidence = cell(1,K);
    soft_evidence = cell(1,K);
    evidence_node_list = unique(node_label_hier{1,3})'; % superpixel index of H(1)
    for j = evidence_node_list
        soft_evidence{j} = gmm_posterior(Y_index==j,:);
    end

    % input the soft and hard evidence to the engine
    tic;
    [engine_bnet2, ll2] = enter_evidence(engine_bnet1, hard_evidence, 'soft', soft_evidence);
    disp(['TSBN spend ',num2str(toc),' sec']);
    disp(['loglikelihood: ', num2str(ll2)]);