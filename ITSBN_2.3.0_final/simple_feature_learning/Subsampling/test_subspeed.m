%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This tests the speed of differnt pooling implementations.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @pooling_file @copybrief test_subspeed.m
% @test @copybrief test_subspeed.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%> 
% @file test_subspeed.m
%
% @pre The variables newdata, epochs, etc. must be set externally.
%
% @test 

profile on
A = randn(50,50,100,50);


tic
[Amaxes,Ainds] = avg_pool(A,[2 2]);

newA = reverse_avg_pool(Amaxes,Ainds,[2 2],[size(A,1) size(A,2)]);
% Alooped = loop_max_pool(A,[2 2]);
t=toc


tic
% [A2maxes,A2inds] = fast_max_pool(A,[2 2]);
% A2 = loop_max_pool(A,[2 2]);
% Alooped2 = fast_loop_max_pool(A,[2 2]);
% newA2 = reverse_prob_max_pool(Amaxes,Ainds,[2 2],[size(A,1) size(A,2)]);
newA2 = loop_avg_pool(A,[2 2]);
t2=toc

norm(newA(:)-newA2(:))


t/t2
'speedup'

profile viewer