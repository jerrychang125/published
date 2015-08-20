function z = Gaussian2D(x,y,u,S)
d = 2;
X = [x y]';
z = (1/((2*pi)^(d/2)*det(S)))*exp(-0.5*(X-u)'*inv(S)*(X-u));

% -- Example code ---
% plot surface
% ezsurf(@(x,y)Gaussian2D(x,y,[0 0]',[2 -1;-1 1])); daspect([1 1 1])

% plot the surface and contour
% ezsurfc(@(x,y)Gaussian2D(x,y,[0 0]',[2 -1;-1 1])); daspect([1 1 1])

