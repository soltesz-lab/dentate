
from neuron import h

class SimTimeEvent:

    def __init__(self, pc, max_walltime_hours, results_write_time, dt_status=1.0, dt_checksimtime=5.0):
        self.pc  = pc
        self.walltime_status = 0
        self.dt_status = dt_status
        self.tcsum = 0.
        self.tcma = 0.
        self.nsimsteps = 0
        self.walltime_checksimtime = 0
        self.max_walltime_hrs = max_walltime_hours
        self.results_write_time = results_write_time
        self.dt_checksimtime = dt_checksimtime
        self.fih_checksimtime = h.FInitializeHandler(1, self.checksimtime)
        if (int(env.pc.id()) == 0):
            print "dt = %g" % h.dt
            print "tstop = %g" % h.tstop
            h.fih_simstatus = h.FInitializeHandler(1, self.simstatus)

    def simstatus(self):
        wt = h.startsw()
        if (self.walltime_status > 0):
            if (int(self.pc.id()) == 0):
                print "*** rank 0 computation time at t=%g ms was %g s\n" % (t, wt-self.walltime_status)
        self.walltime_status = wt
        if ((h.t + self.dt_status) < h.tstop):
            h.cvode.event(h.t + self.dt_status, self.simstatus)

    def checksimtime(self):
        wt = startsw()
        if (t > 0):
            tt = wt - self.walltime_checksimtime
            ## cumulative moving average simulation time per dt_checksimtime
            self.tcma = self.tcma + (tt - self.tcma) / (self.nsimsteps + 1)
            self.tcsum = self.tcsum + tt
            ## remaining physical time
            trem = h.tstop - h.t
            ## remaining simulation time
            tsimrem = max_walltime_hrs*3600 - self.tcsum - h.mkcellstime - h.connectcellstime - h.connectgjstime
            min_tsimrem = self.pc.allreduce(tsimrem, 3) ## minimum value
            ## simulation time necessary to complete the simulation
            tsimneeded = (trem/self.dt_checksimtime)*self.tcma+self.results_write_time
            max_tsimneeded = self.pc.allreduce(tsimneeded, 2) ## maximum value
            if (int(self.pc.id()) == 0): 
                print "*** remaining computation time is %g s and remaining simulation time is %g ms" % (tsimrem, trem)
                print "*** estimated computation time to completion is %g s" % max_tsimneeded

            if (max_tsimneeded > min_tsimrem):
                tstop1 = int((tsimrem - self.results_write_time)/(self.tcma/self.dt_checksimtime)) + h.t
                min_tstop = self.pc.allreduce(tstop1, 3) ## minimum value
                if (int(self.pc.id()) == 0):
                        print "*** not enough time to complete %g ms simulation, simulation will likely stop around %g ms" % (tstop, min_tstop)
                if (min_tstop <= h.t):
                    h.tstop = h.t + h.dt
            else:
                h.tstop = min_tstop
                cvode.event(h.tstop)
        self.nsimsteps = self.nsimsteps + 1
        self.walltime_checksimtime = wt
        if (h.t + self.dt_checksimtime < h.tstop):
            cvode.event(h.t + self.dt_checksimtime, self.checksimtime)

