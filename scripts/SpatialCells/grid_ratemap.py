
import numpy as np

## Grid cell activity map.
##
## Equation from de Almeida and Lisman JNeurosci 2009:
##
## G(r,lambda,theta,c) = g(\sum_{k=1..3} cos{(4*pi)/sqrt(3*lambda) u (t + theta) \dot (r - c)})
## where
##   r: spatial position (x,y)
##   c: spatial offset (xoff,yoff)
##   t = -pi/6.0, pi/6.0, pi/2.0
##   u(t) = [cos(t) sin(t)] is the unitary vector pointing to the direction t
##   g(x) = exp{a (x - b)} - 1, gain function with b = -3/2 and a = 0.3

def gain (a,b,x) = exp(a * (x - b))
def u (t) = np.array([np.cos(t) np.sin(t)])
  
def ratemap (pos,lam,theta,offset)

  a = 0.3;
  b = -3/2;

  tsum = 0.0
        
  for t in [-pi/6.0, pi/6.0, pi/2.0]:

    k = (4.0 * np.pi)/np.sqrt(3.0 * lam)
    tsum = tsum + np.cos (k * np.dot(u(t + theta), pos - offset))
  
  return gain(a, b, tsum)

