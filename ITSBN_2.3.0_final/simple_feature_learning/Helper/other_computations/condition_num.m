%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Computes the condiiton number for the first layer of the Deconvolution
% Networks.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @outparam \li \e zsample feature maps \li \e ysample the input maps
% \li \e F the filters \li \e xdim the x dimension of the input \li \e the y
% dimension of the input \li \e lambda the coefficient on the reconstruction
% term \li \e alpha the sparsity norm.
%
% @other_comp_file @copybrief condition_num.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


n = length(zsample)
max_eig = eigs(@(zsample)computeAtAx(zsample,ysample,F,xdim,ydim,lambda,alpha), n, 'LM', 1);
min_eig = eigs(@(zsample)computeAtAx(zsample,ysample,F,xdim,ydim,lambda,alpha), n, 'SM', 1);


condition_number = max_eig/min_eig



