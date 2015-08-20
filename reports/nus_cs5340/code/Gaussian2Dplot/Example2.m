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

% range of x-y that we want to plot
rangeX = -5:0.5:45;
rangeY = 15:0.5:35;

% plot the gaussian background
[xI,yI] = meshgrid(rangeX,rangeY);
matrixSize = size(xI);
xI = xI(:);
yI = yI(:);
zI1 = xI*0;
zI2 = xI*0;
zI3 = xI*0;
numInterpPnt = length(xI);
for i = 1:numInterpPnt
zI1(i) = Gaussian2D(xI(i),yI(i),u1,S1);
end

for i = 1:numInterpPnt
zI2(i) = Gaussian2D(xI(i),yI(i),u2,S2);
end

for i = 1:numInterpPnt
zI3(i) = Gaussian2D(xI(i),yI(i),u3,S3);
end

xI = reshape(xI,matrixSize);
yI = reshape(yI,matrixSize);
zI1 = reshape(zI1,matrixSize);
zI2 = reshape(zI2,matrixSize);
zI3 = reshape(zI3,matrixSize);

zI = zI1 + zI2 + zI3;

figure; pcolor(xI,yI,zI); shading interp
colormap hot
daspect([1 1 1]);

% then plot the contour on top of the shaded plot
numPoint = 2000;

C = 'g';
h = plotGaussianContour2D(u1,S1,C,numPoint);

C = 'b';
h = plotGaussianContour2D(u2,S2,C,numPoint);

C = 'm';
h = plotGaussianContour2D(u3,S3,C,numPoint);
xlabel('x coordinate');
ylabel('y coordinate');
title('estimated locations of the targets');
