%% Creates the GCL septotemporal line with 10000 points

function [x,y,z] = layer_eq(u,v,layer)

% Define equations
x=(5.3-1.*sin(u)+(1.00+layer.*0.138).*cos(v)).*cos(u).*-500;
y=((5.5-2.*sin(u)+(0.9+layer.*0.114).*cos(v)).*sin(u)).*750;
z=(sin(v+-0.13*(pi-u))).*(663+layer*114)+2500*sin(u);