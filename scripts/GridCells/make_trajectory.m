
function [Xpos,Ypos] = make_trajectory(X, Y, N, Xstart, Ystart, vmax, tend, grid_unit)

  size_len = sqrt(grid_unit)

  Xmax = size(X,1)
  Ymax = size(Y,1)

  Xpos = zeros(N, tend);
  Ypos = zeros(N, tend);
  
  Xpos(:, 1) = Xstart;
  Ypos(:, 1) = Ystart;

  xrescale = 1; yrescale = 1;
  xdir = 1; ydir = 1;

  for t = 2:tend

    vx = randi(vmax, N, 1) - vmax/2;
    Xpos(:,t) = Xpos(:,t-1) + xdir*vx;

    vy = randi(vmax, N, 1) - vmax/2;
    Ypos(:,t) = Ypos(:,t-1) + ydir*vy;

    for i = 1:N
        if Xpos(i,t) > Xmax
           Xpos(i,t) = Xpos(i,t) - randi(xrescale*vmax, 1, 1);
           yrescale = randi(4,1,1);
           xrescale = 1;
           xdir = -xdir;
        elseif Xpos(i,t) < 1
          Xpos(i,t) = Xpos(i,t) + randi(xrescale*vmax, 1, 1);
          yrescale = randi(4,1,1);
          xrescale = 1;
          xdir = -xdir;
        end

        if Ypos(i,t) > Ymax
           Ypos(i,t) = Ypos(i,t) - randi(yrescale*vmax, 1, 1);
           yrescale = 1;
           xrescale = randi(4,1,1);
           ydir = -ydir;
        elseif Ypos(i,t) < 1
           Ypos(i,t) = Ypos(i,t) + randi(yrescale*vmax, 1, 1);
           yrescale = 1;
           xrescale = randi(4,1,1);
           ydir = -ydir;
        end
    end
  end
