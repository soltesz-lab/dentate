%% Grid cell activity map.
%% Equation is
%% 1/3 * cos[ 4*pi/(sqrt(3)*l) {cos(t-rot), sin(t-rot)} dot (x-offset)  ] for t = -pi/3, 0, pi/3
function out = grid_activity(X,Y,l,rot,xoff,yoff,k)

  out = zeros(size(l));
        
  for tt=[-pi/3.0, 0.0, pi/3.0]

    x1 = (2*pi/(sqrt(3)*k))*l;
    x2 = cos(tt-rot);
    x3 = sin(tt-rot);
    x4 = X - xoff;
    x5 = Y - yoff;

    t1 = x2.*x4 + x3.*x5;
    out = out + 1/3 * cos(x1 .* t1);

  end
  
  out(find(out < 0)) = 0;

