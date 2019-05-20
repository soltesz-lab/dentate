"""Routines to keep track of simulation computation time and terminate the simulation if not enough time has been allocated."""

import time
from neuron import h
from dentate import utils

# This logger will inherit its settings from the root logger, created in dentate.env
logger = utils.get_module_logger(__name__)

class SimTimeEvent:

    def __init__(self, pc, max_walltime_hours, results_write_time, setup_time, dt_status=1.0, dt_checksimtime=5.0):
        self.pc  = pc
        wt = time.time()
        self.walltime_status = wt
        self.walltime_checksimtime = wt
        self.dt_status = dt_status
        self.setup_time = setup_time
        self.tcsum = 0.
        self.tcma = 0.
        self.nsimsteps = 0
        self.max_walltime_hours = max_walltime_hours
        self.results_write_time = results_write_time
        self.dt_checksimtime = dt_checksimtime
        self.fih_checksimtime = h.FInitializeHandler(1, self.checksimtime)
        self.fih_simstatus = h.FInitializeHandler(1, self.simstatus)
        if (int(pc.id()) == 0):
            logger.info("max wall time hours = %g" % max_walltime_hours)
            logger.info("wall time = %g" % wt)
            logger.info("dt = %g" % h.dt)
            logger.info("tstop = %g" % h.tstop)

    def reset(self):
        self.tcsum = 0.
        self.tcma = 0.
        self.nsimsteps = 0
        self.walltime_status = time.time()
        self.fih_checksimtime = h.FInitializeHandler(1, self.checksimtime)
        self.fih_simstatus = h.FInitializeHandler(1, self.simstatus)
    
    def simstatus(self):
        wt = time.time()
        if h.t > 0.:
            if (int(self.pc.id()) == 0):
                logger.info("*** rank 0 computation time at t=%.2f ms was %.2f s" % (h.t, wt-self.walltime_status))
        else:
            init_time = wt-self.walltime_status
            max_init_time = self.pc.allreduce(init_time, 2) ## maximum value
            self.setup_time += max_init_time
        self.walltime_status = wt
        if ((h.t + self.dt_status) < h.tstop):
            h.cvode.event(h.t + self.dt_status, self.simstatus)

    def checksimtime(self):
        wt = time.time()
        if (h.t > 0):
            tt = wt - self.walltime_checksimtime
            ## cumulative moving average simulation time per dt_checksimtime
            self.tcma = self.tcma + (tt - self.tcma) / (self.nsimsteps + 1)
            self.tcsum = self.tcsum + tt
            ## remaining physical time
            trem = h.tstop - h.t
            ## remaining simulation time
            tsimrem = self.max_walltime_hours*3600 - self.tcsum - self.setup_time
            min_tsimrem = self.pc.allreduce(tsimrem, 3) ## minimum value
            ## simulation time necessary to complete the simulation
            tsimneeded = (trem/self.dt_checksimtime)*self.tcma+self.results_write_time
            max_tsimneeded = self.pc.allreduce(tsimneeded, 2) ## maximum value
            if (int(self.pc.id()) == 0): 
                logger.info("*** remaining computation time is %.2f s and remaining simulation time is %.2f ms" % (tsimrem, trem))
                logger.info("*** estimated computation time to completion is %.2f s" % max_tsimneeded)
                logger.info("*** computation time so far is %.2f s" % (self.tcsum + self.setup_time))
            ## if not enough time, reduce tstop and perform collective operations to set minimum (earliest) tstop across all ranks
            if (max_tsimneeded > min_tsimrem):
                tstop1 = int((tsimrem - self.results_write_time)/(self.tcma/self.dt_checksimtime)) + h.t
                min_tstop = self.pc.allreduce(tstop1, 3) ## minimum value
                if (int(self.pc.id()) == 0):
                        logger.info("*** not enough time to complete %.2f ms simulation, simulation will likely stop around %2.f ms" % (h.tstop, min_tstop))
                if (min_tstop <= h.t):
                    h.tstop = h.t + h.dt
                else:
                    h.tstop = min_tstop
                    h.cvode.event(h.tstop)
        self.nsimsteps = self.nsimsteps + 1
        self.walltime_checksimtime = wt
        if (h.t + self.dt_checksimtime < h.tstop):
            h.cvode.event(h.t + self.dt_checksimtime, self.checksimtime)

