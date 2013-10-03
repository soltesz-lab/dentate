#
#
#   File to test dentate project
#
#   To execute this type of file, type '..\..\..\nC.bat -python XXX.py' (Windows)
#   or '../../../nC.sh -python XXX.py' (Linux/Mac). Note: you may have to update the
#   NC_HOME and NC_MAX_MEMORY variables in nC.bat/nC.sh
#
#   Author: Padraig Gleeson
#
#   This file has been developed as part of the neuroConstruct project
#   This work has been funded by the Medical Research Council and the
#   Wellcome Trust
#
#

import sys
import os

try:
    from java.io import File
except ImportError:
    print "Note: this file should be run using ..\\..\\..\\nC.bat -python XXX.py' or '../../../nC.sh -python XXX.py'"
    print "See http://www.neuroconstruct.org/docs/python.html for more details"
    quit()

sys.path.append(os.environ["NC_HOME"]+"/pythonNeuroML/nCUtils")

import ncutils as nc # Many useful functions such as SimManager.runMultipleSims found here

projFile = File(os.getcwd(), "../Dentate.ncx")



##############  Main settings  ##################

simConfigs = []

simConfigs.append("Default Simulation Configuration")

simDt =                 0.01

simulators =            ["NEURON"]

numConcurrentSims =     4

varTimestepNeuron =     False
varTimestepTolerance =  0.00001

plotSims =              True
plotVoltageOnly =       True
runInBackground =       True
analyseSims =           True

verbose = True

#############################################


def testAll(argv=None):
    if argv is None:
        argv = sys.argv

    print "Loading project from "+ projFile.getCanonicalPath()


    simManager = nc.SimulationManager(projFile,
                                      numConcurrentSims = numConcurrentSims,
                                      verbose = verbose)

    simManager.runMultipleSims(simConfigs =           simConfigs,
                               simDt =                simDt,
                               simulators =           simulators,
                               runInBackground =      runInBackground,
                               varTimestepNeuron =    varTimestepNeuron,
                               varTimestepTolerance = varTimestepTolerance)

    simManager.reloadSims(plotVoltageOnly =   plotVoltageOnly,
                          analyseSims =       analyseSims)

    # Times for 0.002
    '''spikeTimesToCheck = {'MC_2cells_0': [208.616, 228.986, 249.076, 269.178, 289.37, 309.674, 330.094, 350.644, 371.324, 392.148, 413.124, 434.262, 455.572, 477.068, 498.758, 520.658, 542.78, 565.136, 587.74, 610.606, 633.752, 657.19, 680.936, 705.01, 729.428, 754.208, 779.372],
                         'HC_2cells_0': [203.14, 209.936, 216.602, 223.352, 230.18, 237.076, 244.026, 251.026, 258.07, 265.16, 272.296, 279.478, 286.712, 294.004, 301.356, 308.776, 316.27, 323.846, 331.51, 339.27, 347.134, 355.112, 363.21, 371.44, 379.812, 388.334, 397.018, 405.876, 414.918, 424.16, 433.614, 443.294, 453.216, 463.394, 473.848, 484.598, 495.658, 507.054, 518.806, 530.94, 543.478, 556.452, 569.886, 583.812, 598.264, 613.274, 628.876, 645.106, 661.998, 679.588, 697.908, 716.986, 736.844, 757.502, 778.966],
                         'GC_2cells_0': [211.06, 228.346, 248.88, 275.776, 310.672, 353.33, 402.16, 455.576, 512.614, 572.762, 635.624, 700.69, 767.282],
                         'BC_2cells_0': [204.46, 212.558, 220.798, 229.154, 237.606, 246.14, 254.748, 263.422, 272.15, 280.928, 289.75, 298.612, 307.506, 316.432, 325.384, 334.36, 343.358, 352.374, 361.408, 370.458, 379.522, 388.598, 397.686, 406.784, 415.892, 425.008, 434.134, 443.266, 452.406, 461.552, 470.702, 479.86, 489.022, 498.19, 507.362, 516.538, 525.716, 534.9, 544.088, 553.278, 562.472, 571.668, 580.866, 590.068, 599.274, 608.48, 617.688, 626.9, 636.114, 645.328, 654.546, 663.766, 672.986, 682.208, 691.432, 700.658, 709.884, 719.114, 728.344, 737.574, 746.806, 756.04, 765.274, 774.51, 783.748, 792.986]}
    '''
    spikeTimesToCheck = {'MC_2cells_0': [208.63, 228.99, 249.06, 269.14, 289.3, 309.58, 329.98, 350.51, 371.17, 391.97, 412.93, 434.04, 455.33, 476.81, 498.48, 520.36, 542.46, 564.8, 587.39, 610.23, 633.36, 656.77, 680.5, 704.55, 728.95, 753.71, 778.85],
                         'HC_2cells_0': [203.15, 209.95, 216.62, 223.37, 230.2, 237.09, 244.04, 251.04, 258.08, 265.16, 272.29, 279.46, 286.68, 293.95, 301.29, 308.68, 316.15, 323.69, 331.31, 339.03, 346.85, 354.77, 362.8, 370.96, 379.25, 387.68, 396.27, 405.02, 413.94, 423.05, 432.36, 441.87, 451.62, 461.6, 471.84, 482.35, 493.16, 504.27, 515.71, 527.51, 539.67, 552.24, 565.23, 578.67, 592.59, 607.02, 621.99, 637.54, 653.69, 670.48, 687.94, 706.1, 724.98, 744.61, 765.0, 786.16],
                         'GC_2cells_0': [211.08, 228.37, 248.85, 275.57, 310.12, 352.3, 400.58, 453.41, 509.84, 569.35, 631.55, 695.97, 761.97],
                         'BC_2cells_0': [204.47, 212.58, 220.83, 229.2, 237.66, 246.2, 254.82, 263.5, 272.24, 281.02, 289.85, 298.72, 307.63, 316.56, 325.52, 334.5, 343.51, 352.53, 361.57, 370.63, 379.7, 388.78, 397.88, 406.98, 416.1, 425.22, 434.35, 443.49, 452.64, 461.79, 470.95, 480.12, 489.29, 498.46, 507.64, 516.83, 526.01, 535.21, 544.4, 553.6, 562.8, 572.0, 581.21, 590.42, 599.63, 608.85, 618.07, 627.28, 636.51, 645.73, 654.96, 664.18, 673.41, 682.64, 691.87, 701.11, 710.34, 719.58, 728.82, 738.06, 747.3, 756.54, 765.78, 775.03, 784.27, 793.52]}

    spikeTimeAccuracy = 0.05

    report = simManager.checkSims(spikeTimesToCheck = spikeTimesToCheck,
                                  spikeTimeAccuracy = spikeTimeAccuracy)

    print report

    return report


if __name__ == "__main__":
    testAll()
