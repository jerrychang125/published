function []=plotContourBackgroundGaussian2D(u,S,rangeX,rangeY,C, numPoint, FigID)

% written by Kittipat "Bot" Kampa, University of Florida, based on the
% ellipse.m original written by D.G. Long, Brigham Young University

% plot the gaussian background
[xI,yI] = meshgrid(rangeX,rangeY);
matrixSize = size(xI);
xI = xI(:);
yI = yI(:);
zI = zeros(size(xI,1),size(xI,2),size(u,3));
numInterpPnt = length(xI);

for n = 1:size(u,3)
    for i = 1:numInterpPnt
        zI(i,1,n) = Gaussian2D(xI(i),yI(i),u(:,:,n),S(:,:,n));
    end
end

xI = reshape(xI,matrixSize);
yI = reshape(yI,matrixSize);

zIsum = sum(zI,3);
zIsum = reshape(zIsum,matrixSize);

figure(FigID); pcolor(xI,yI,zIsum); shading interp
colormap copper
daspect([1 1 1]);

% then plot the contour on top of the shaded plot
for n = 1:size(u,3)
    h = plotGaussianContour2D(u(:,:,n),S(:,:,n),C(n),numPoint);
end