%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Loads various epoch10_layer1.mat files, computes their total scalled
% energy for clean (first half of samples) and noisy (second half of
% sample) for different lambda values and then plots the result.
% Assumes you are in the directory of each Run_##
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @plotting_file @copybrief plot_clearn_vs_noise1.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set of runs you want to plot.
for run=1:10
    
    load(strcat('./Run_',num2str(run),'/epoch10_layer1.mat'))
    
    %% Load the parameters from the model struct.
    layer = model.layer;
    lambda = model.lambda(layer);
    kappa(1) = 1;
    beta = model.Binitial*(model.betaT*model.Bmultiplier);
    lambda_input = model.lambda_input;
    
    % Number of clean samples is half the total number of samples.
    num_clean = size(model.reg_error,2)/2;
    
    % To see for the given epoch.
    epoch = 10;
    
    
    cleanE(run) = lambda/2*mean(model.pix_update_rec_error(epoch,1:num_clean),2)+...
        kappa(layer)*mean(model.reg_error(epoch,1:num_clean),2)+...
        lambda_input*mean(model.update_noise_rec_error(epoch,1:num_clean),2)+...
        (beta/2)/kappa(layer)*mean(model.beta_rec_error(epoch,1:num_clean),2);

    noisyE(run) = lambda/2*mean(model.pix_update_rec_error(epoch,num_clean+1:end),2)+...
        kappa(layer)*mean(model.reg_error(epoch,num_clean+1:end),2)+...
        lambda_input*mean(model.update_noise_rec_error(epoch,num_clean+1:end),2)+...
        (beta/2)/kappa(layer)*mean(model.beta_rec_error(epoch,num_clean+1:end),2);    
    
end


%% Plot the results.
f = figure; clf;
plot(log10(cleanE),'-bs')
hold on
plot(log10(noisyE),'-ro')
title('Total Energy of Clean versus Noisy Images Epoch 10')
xlabel('lambda increasing')
ylabel('total energy')

f = figure; clf;
plot(noisyE./cleanE,'-g')
title('Ratio of the Noisy Energy over Clean Energy Epoch 10')
xlabel('lambda increasing')
ylabel('ratio of energies')
