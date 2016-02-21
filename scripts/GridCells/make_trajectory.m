
function [tX,tY] = make_trajectory(X, Y, N, Xstart, Ystart, vmax, tend, grid_unit)

  size_len = sqrt(grid_unit)

  Xmax = size(X,1)
  Ymax = size(Y,1)

  tX = zeros(N, tend);
  tY = zeros(N, tend);
  
  tX(:, 1) = Xstart;
  tY(:, 1) = Ystart;

  xrescale = 1;
  yrescale = 1;
  xdir = 1; ydir = 1;

  for t = 2:tend

    vx = randi(vmax, N, 1) - vmax/2;
    tX(:,t) = tX(:,t-1) + xdir*vx;

    vy = randi(vmax, N, 1) - vmax/2;
    tY(:,t) = tY(:,t-1) + ydir*vy;

    for i = 1:N
        if tX(i,t) > Xmax
           tX(i,t) = tX(i,t) - randi(xrescale*vmax, 1, 1);
           yrescale = randi(4,1,1);
           xrescale = 1;
           xdir = -xdir;
        elseif tX(i,t) < 1
          tX(i,t) = tX(i,t) + randi(xrescale*vmax, 1, 1);
          yrescale = randi(4,1,1);
          xrescale = 1;
          xdir = -xdir;
        end

        if tY(i,t) > Ymax
           tY(i,t) = tY(i,t) - randi(yrescale*vmax, 1, 1);
           yrescale = 1;
           xrescale = randi(4,1,1);
           ydir = -ydir;
        elseif tY(i,t) < 1
           tY(i,t) = tY(i,t) + randi(yrescale*vmax, 1, 1);
           yrescale = 1;
           xrescale = randi(4,1,1);
           ydir = -ydir;
        end
    end
  end
