clear all
clc
close all

% Gaussian#1
u1 = [4.9025
   22.3335];
S1 = [1.8066    0.4998
    0.4998    1.9996];

% Gaussian#2
u2 = [17.0114
   23.5313];
S2 = [2.3107    0.4998
    0.4998    2.4994];

% Gaussian#3
u3 = [37.4230
   24.5666];
S3 = [3.2988    -0.9993
    -0.9993    1.9983];

% put all the Gaussian parameters in a matrix
u = zeros(2,1,3);
S = zeros(2,2,3);
u(:,:,1) = u1;
u(:,:,2) = u2;
u(:,:,3) = u3;
S(:,:,1) = S1;
S(:,:,2) = S2;
S(:,:,3) = S3;

% range of x-y that we want to plot
rangeX = -5:0.5:45;
rangeY = 15:0.5:35;


C = ['g','b','m']; % the color of each Gaussian
numPoint = 2000; % density of the ellipse contour
FigID = 11; % figure ID of the plot (positive integer)

% now we call the function
plotContourBackgroundGaussian2D(u,S,rangeX,rangeY,C, numPoint, FigID);

% add the detail of the plot here
figure(FigID);
colormap hot
xlabel('x coordinate');
ylabel('y coordinate');
title('estimated locations of the targets');



