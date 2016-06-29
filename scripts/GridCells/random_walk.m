
function [Xpos,Ypos] = random_walk(W,tend,dt)

nsteps = floor(tend/dt);
enclosureRadius = W/2;
velocity = rand()/2;

Xpos = zeros(nsteps,1);
Ypos = zeros(nsteps,1);
headDir = zeros(nsteps,1)';

Xpos(1) = 0;
Ypos(1) = 0;
headDir(1) = rand()*2*pi;

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
    
    while (sqrt((Xpos(i-1) + cos(headDir(i-1))*velocity)^2 ...
                + (Ypos(i-1) + sin(headDir(i-1))*velocity)^2)  > enclosureRadius)
        
        headDir(i-1) = headDir(i-1) + leftOrRight*pi/100;
        
    end
    Xpos(i) = Xpos(i-1)+cos(headDir(i-1))*velocity; 
    Ypos(i) = Ypos(i-1)+sin(headDir(i-1))*velocity; 
    headDir(i) = mod(headDir(i-1) + (rand()-.5)/5*pi/2,2*pi);
end
Xpos = Xpos + W/2;
Ypos = Ypos + W/2;