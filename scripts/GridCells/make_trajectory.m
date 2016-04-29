
function [Xpos,Ypos] = make_trajectory(X, Y, Xstart, Ystart, tend, dt, gain, grid_unit)

  N = tend/dt

  Xmax = size(X,2)
  Ymax = size(Y,2)

  Xpos = zeros(1, N);
  Ypos = zeros(1, N);
  
  Xpos(1, 1) = Xstart;
  Ypos(1, 1) = Ystart;

  xrescale = 1; yrescale = 1;
  xdir = 1; ydir = 1;
  t_changedir = randi(N);

  i = 1;
  for t = 2:N

    vx = randi(gain) - gain/2;

    Xpos(:,t) = Xpos(:,t-1) + xdir*vx;

    vy = randi(gain) - gain/2;
    Ypos(:,t) = Ypos(:,t-1) + ydir*vy;

    if t > t_changedir
      xdir = randi(4); ydir = randi(4);
      t_rest = N - t;
      if t_rest > 1
        t_changedir = t + randi(t_rest);
      end
    end
    while (Xpos(i,t) > Xmax  || Xpos(i,t) < 1)
      if Xpos(i,t) > Xmax
        Xpos(i,t) = Xpos(i,t) - randi(xrescale*gain, 1, 1);
        yrescale = randi(4,1,1);
        xrescale = 1;
        xdir = -xdir;
      elseif Xpos(i,t) < 1
        Xpos(i,t) = Xpos(i,t) + randi(xrescale*gain, 1, 1);
        yrescale = randi(4,1,1);
        xrescale = 1;
        xdir = -xdir;
      end
    end
    
    while (Ypos(i,t) > Ymax || Ypos(i,t) < 1)
      if Ypos(i,t) > Ymax
        Ypos(i,t) = Ypos(i,t) - randi(yrescale*gain, 1, 1);
        yrescale = 1;
        xrescale = randi(4,1,1);
        ydir = -ydir;
      elseif Ypos(i,t) < 1
        Ypos(i,t) = Ypos(i,t) + randi(yrescale*gain, 1, 1);
        yrescale = 1;
        xrescale = randi(4,1,1);
        ydir = -ydir;
      end
    end
  end
