function h = plotGaussianContour2D(u,S,radius_scale,C,numPoint);
% plot the contour of 1 std of 2D Gaussian
% u = mean vector
% S = covariance matrix
% C = color
% numPoint = number of points to show

% written by Kittipat "Bot" Kampa, University of Florida, based on the
% ellipse.m original written by D.G. Long, Brigham Young University

[U,Sigma,V] = svd(S);
angRad = atan(U(2,1)/U(1,1));
% figure; ezmesh(@(x,y)Gaussian2D(x,y,u,S)); % daspect([1 1 1])
h=ellipse(radius_scale*sqrt(Sigma(1,1)),radius_scale*sqrt(Sigma(2,2)),angRad,u(1,1),u(2,1),C,numPoint);

% example code
% u = [0 0]';
% S = [2 -1;-1 1];
% C = 'r';
% numPoint = 300;
% figure;
% h = plotGaussianContour2D(u,S,C,numPoint);
% 
% u = [1 1]';
% S = [2 1;1 1];
% C = 'b';
% h = plotGaussianContour2D(u,S,C,numPoint);