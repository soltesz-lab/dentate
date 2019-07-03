"""Routines to keep track of simulation computation time and terminate the simulation if not enough time has been allocated."""
import time

from dentate.utils import get_module_logger, old_div
from neuron import h

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


class SimTimeEvent(object):

    def __init__(self, pc, max_walltime_hours, results_write_time, setup_time, dt_status=1.0, dt_checksimtime=5.0):
        if (int(pc.id()) == 0):
            logger.info("*** allocated wall time is %.2f hours" % (max_walltime_hours))
        wt = time.time()
        self.pc = pc
        self.walltime_status = wt
        self.walltime_checksimtime = wt
        self.dt_status = dt_status
        self.tcsum = 0.
        self.tcma = 0.
        self.nsimsteps = 0
        self.walltime_max = max_walltime_hours * 3600. - setup_time
        self.results_write_time = results_write_time
        self.dt_checksimtime = dt_checksimtime
        self.fih_checksimtime = h.FInitializeHandler(1, self.checksimtime)
        self.fih_simstatus = h.FInitializeHandler(1, self.simstatus)
        if (int(self.pc.id()) == 0):
            logger.info("*** max wall time is %.2f s; max setup time was %.2f s" % (self.walltime_max, setup_time))

    def reset(self):
        wt = time.time()
        self.walltime_max = self.walltime_max - self.tcsum
        self.tcsum = 0.
        self.tcma = 0.
        self.nsimsteps = 0
        self.walltime_status = wt
        self.walltime_checksimtime = wt
        self.fih_checksimtime = h.FInitializeHandler(1, self.checksimtime)
        self.fih_simstatus = h.FInitializeHandler(1, self.simstatus)

    def simstatus(self):
        wt = time.time()
        if h.t > 0.:
            if (int(self.pc.id()) == 0):
                logger.info("*** rank 0 computation time at t=%.2f ms was %.2f s" % (h.t, wt - self.walltime_status))
        self.walltime_status = wt
        if ((h.t + self.dt_status) < h.tstop):
            h.cvode.event(h.t + self.dt_status, self.simstatus)

    def checksimtime(self):
        wt = time.time()
        if (h.t > 0):
            tt = wt - self.walltime_checksimtime
            ## cumulative moving average wall time time per dt_checksimtime
            self.tcma = self.tcma + old_div((tt - self.tcma), (self.nsimsteps + 1))
            self.tcsum = self.tcsum + tt
            ## remaining physical time
            trem = h.tstop - h.t
            ## remaining wall time
            walltime_rem = self.walltime_max - self.tcsum
            walltime_rem_min = self.pc.allreduce(walltime_rem, 3)  ## minimum value
            ## wall time necessary to complete the simulation
            walltime_needed = (old_div(trem, self.dt_checksimtime)) * self.tcma + self.results_write_time
            walltime_needed_max = self.pc.allreduce(walltime_needed, 2)  ## maximum value
            if (int(self.pc.id()) == 0):
                logger.info("*** remaining computation time is %.2f s and remaining simulation time is %.2f ms" % (
                walltime_rem, trem))
                logger.info("*** estimated computation time to completion is %.2f s" % walltime_needed_max)
                logger.info("*** computation time so far is %.2f s" % self.tcsum)
            ## if not enough time, reduce tstop and perform collective operations to set minimum (earliest) tstop across all ranks
            if (walltime_needed_max > walltime_rem_min):
                tstop1 = int(
                    old_div((walltime_rem - self.results_write_time), (old_div(self.tcma, self.dt_checksimtime)))) + h.t
                min_tstop = self.pc.allreduce(tstop1, 3)  ## minimum value
                if (int(self.pc.id()) == 0):
                    logger.info(
                        "*** not enough time to complete %.2f ms simulation, simulation will likely stop around %2.f ms" % (
                        h.tstop, min_tstop))
                if (min_tstop <= h.t):
                    h.tstop = h.t + h.dt
                else:
                    h.tstop = min_tstop
                    h.cvode.event(h.tstop)
            self.nsimsteps = self.nsimsteps + 1
        else:
            init_time = wt - self.walltime_checksimtime
            max_init_time = self.pc.allreduce(init_time, 2)  ## maximum value
            self.tcsum += max_init_time
            if (int(self.pc.id()) == 0):
                logger.info("*** max init time at t=%.2f ms was %.2f s" % (h.t, max_init_time))
                logger.info("*** computation time so far is %.2f and total computation time is %.2f s" % (
                self.tcsum, self.walltime_max))
        self.walltime_checksimtime = wt
        if (h.t + self.dt_checksimtime < h.tstop):
            h.cvode.event(h.t + self.dt_checksimtime, self.checksimtime)
