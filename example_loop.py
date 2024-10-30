import os
from pathlib import Path

from main import *

def BS1_scheduler(stepNumber):
    if stepNumber == -1:
        return -pi/4
    return -pi/4

### Paramaters ###
for i, gamma in enumerate(np.linspace(0, np.pi, 30)):
    nSteps = 4 # number of steps for walk. Sequence for walk is aBBO 45 deg -> 0 deg -> 45 deg ->...                  
    alphaSq = 0.08 # intensity / mean photon number of coherent state
    r = 0.12 # squeezing parameter
    eta = 1 # overall efficiency
    # gamma = 0 # phase shift between H,V due to group delay in aBBO
    mm = 1 # mode matching (HOM visibility)
    n_noise = 0 #5e-6 # dark count prob per pump pulse in each mode
    max_photons = 1 # maximum number of photons detected

    ### Run simulation ###

    pn_ideal = computeWalkOutput(nSteps, r, alphaSq, eta, gamma, max_photons, n_noise, BS1_scheduler)


    # Keep only b-photon or a-photon modes (tracing-over non detected modes)
    pn_ideal_a, pn_ideal_b = traceOverModes(pn_ideal)


    ### Plotting specific outcomes ###

    # look at 1-photon a subspace
    oneFolds_ideal_a = filterProbDict(pn_ideal_a)
    # look at 1-photon b subspace
    oneFolds_ideal_b = filterProbDict(pn_ideal_b)

    # plot
    plot_destdir = os.path.join(str(Path().absolute()), f'plots/theta{BS1_scheduler(-1):.2f}/nsteps{nSteps}/')
    if not os.path.exists(plot_destdir):
        os.makedirs(plot_destdir)
    utbe_plot(oneFolds_ideal_a, plot_destdir, postfix=f'nsteps{nSteps}_gamma{gamma:.2f}_a')
    utbe_plot(oneFolds_ideal_b, plot_destdir, postfix=f'nsteps{nSteps}_gamma{gamma:.2f}_b')
    print(f"finished loop {i}, with gamma = {gamma:.2f}")

