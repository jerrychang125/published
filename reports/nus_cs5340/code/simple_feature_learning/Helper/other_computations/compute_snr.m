%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Compute the Signal-to-Noise Ratio between to images.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @other_comp_file @copybrief compute_snr.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief compute_snr.m
%
% @param A the ground truth (or original) image.
% @param B the corrupted (or reconstructed) image.
% @retval snr the signal to noise ratio between the two input images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function snr = compute_snr(A,B)
snr = 10*log10( sum(sum(sum((A-mean(A(:))).^2))) / ...
    sum(sum(sum((A - B).^2))) );
end