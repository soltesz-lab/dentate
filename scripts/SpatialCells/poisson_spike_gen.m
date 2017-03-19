function [spikes, ts] = poisson_spike_gen(rates, dt, tend)

  nsteps = floor(tend/dt)
  fr = interp(rates, floor(nsteps/size(rates,2)));
  xs = rand(1,nsteps);
  spikes = find(xs < fr*dt);
  ts = 0:dt:tend-dt;



