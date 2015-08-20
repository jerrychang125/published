%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Draws samples from the distribution of the feature map, z, over their
% activations for a number of trainin examples. Note: this function works on one
% feature map at a time.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @top_down_file @copybrief sample_z_map.m
% @other_comp_file @copybrief sample_z_map.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief sample_z_map.m
%
% @param num_samples integer - # of sample z maps
% @param z_activations z_size_x x z_size_y x #ntrain images matrix -  activations of z map from training images
% @param threshold float - absolute value below which activations=0
% @retval z_out z_size x num_samples -- sampled z maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z_out] = sample_z_map(num_samples,z_activations,threshold)

NUM_BINS = 100;

% get size
[z_size_x,z_size_y,nTrain] = size(z_activations);

% get max,min of activations
z_min = min(z_activations(:));
z_max = max(z_activations(:));

% specify bin locations
bins = linspace(z_min,z_max,NUM_BINS);

% now build histogram
z_hist = hist(z_activations(:),bins);

% normalize histogram
z_hist_n = z_hist / sum(z_hist);

% build culumative distribution function
z_cum = cumsum(z_hist_n);

% add tiny offset to get distinct values for interp1 function
z_cum = z_cum + [1:length(z_cum)]*1e-6;

% generate random numbers from 0..1 uniform
samples = rand(z_size_x,z_size_y,num_samples);

% lookup in cumulative distribution
% z_out = interp1(z_cum,bins,samples(:),'linear','extrap');
z_out = interp1(z_cum,bins,samples(:));

z_out(~isfinite(z_out)) = 0;

% threshold
tmp_ind = find(abs(z_out(:))<threshold);
z_out(tmp_ind) = 0;
% keyboard
% reshape
z_out = reshape(z_out,[z_size_x z_size_y num_samples]);


