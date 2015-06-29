%% Gives point on GCL parametric surface

function [x,y,z] = layer_eq_point(layer,u,v)

% Define equations
x=(5.3-1*sin(u)+(1.00+layer*0.138)*cos(v+0.10*(pi-u))).*cos(u)*-500;
y=((5.5-2*sin(u)+(0.9+layer*0.114)*cos(v+0.10*(pi-u))).*sin(u))*750;
z=(sin(v+-0.03*(pi-u)))*(664+layer*114)+2500*sin(u);