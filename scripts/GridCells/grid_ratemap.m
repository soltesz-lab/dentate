%% Grid cell activity map.
%%
%% Equation from de Almeida and Lisman JNeurosci 2009:
%%
%% G(r,lambda,theta,c) = g(\sum_{k=1..3} cos{(4*pi)/sqrt(3*lambda) u (t + theta) \dot (r - c)})
%% where
%%   t = -pi/6.0, pi/6.0, pi/2.0
%%   u(t) = [cos(t) sin(t)] is the unitary vector pointing to the direction t
%%   g(x) = exp{a (x - b)} - 1, gain function with b = -3/2 and a = 0.3

function ratemap = grid_ratemap(X,Y,lambda,theta,xoff,yoff)

  a = 0.3;
  b = -3/2;

  n = size(lambda);
  ratemap = zeros(n);
        
  for tt=[-pi/6.0, pi/6.0, pi/2.0]

      %x0 = *lambda);
    x0 = lambda*sqrt(3);
    x1 = ((4*pi) * ones(n)) ./ x0;
    x2 = cos(tt-theta);
    x3 = sin(tt-theta);
    x4 = X - xoff;
    x5 = Y - yoff;

    t1 = x2.*x4 + x3.*x5;
    ratemap = ratemap + cos(x1 .* t1);

  end
  
  ratemap = exp(a .* (ratemap - b)) - 1;

