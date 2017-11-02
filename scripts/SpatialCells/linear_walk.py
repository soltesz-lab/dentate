import math
import numpy as np

def linear_walk(tend,dt):

    np.random.seed(seed=21)

    ## max acceleration is .1 cm/ms^2
    max_accel = 0.1
    ## max velocity is .05 cm/ms
    max_vel   = 0.05
    
    nsteps   = int(math.floor(tend/dt))
    velocity = np.random.rand()/2

    Xpos = np.zeros(nsteps,)
    Ypos = np.zeros(nsteps,)
    headDir = np.zeros(nsteps,)
    
    for i in xrange(1,nsteps):
        a  = np.random.randn()+0.05
        dv = max(min(a,max_accel),-max_accel); 
        velocity = min(max(velocity + dv,0.0),max_vel) * dt
        
        Xpos[i] = Xpos[i-1] + np.cos(headDir[i-1]) * velocity 
        Ypos[i] = Ypos[i-1]

    return (Xpos,Ypos)
 
