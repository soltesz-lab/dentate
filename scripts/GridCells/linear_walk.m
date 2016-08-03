
function [Xpos,Ypos] = linear_walk(W,tend,dt)

nsteps = floor(tend/dt);
velocity = rand()/2;

Xpos = zeros(nsteps,1);
Ypos = zeros(nsteps,1);
headDir = zeros(nsteps,1)';

Xpos(1) = 0;
Ypos(1) = 0;

for i = 2:nsteps
    % max acceleration is .1 cm/ms^2
    dv = max(min(normrnd(0,.05),.1),-.1); 
    
    % max velocity is .5 cm/ms
    velocity = min(max(velocity + dv,0),0.05) * dt;
        
    % Don't let trajectory go outside of the boundry, if it would then randomly
    % rotate to the left or right
    leftOrRight = round(rand());
    if (leftOrRight == 0)
        leftOrRight = -1;
    end
    
    Xpos(i) = Xpos(i-1)+cos(headDir(i-1))*velocity; 
    Ypos(i) = Ypos(i-1);

end
Xpos = Xpos + W/2;
