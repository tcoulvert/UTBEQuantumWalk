import os
from pathlib import Path

from main import *

def BS1_scheduler(stepNumber):
    if stepNumber == -1:
        return -pi/4
    return -pi/4

def gamma_scheduler(stepNumber, rng=np.random.default_rng(seed=None)):
    # return 0
    return np.pi * rng.random()

### Paramaters ###
full_ideal_a, full_ideal_b = {}, {}
for trial in range(500):
    nSteps = 5 # number of steps for walk. Sequence for walk is aBBO 45 deg -> 0 deg -> 45 deg ->...                  
    alphaSq = 0.08 # intensity / mean photon number of coherent state
    r = 0.12 # squeezing parameter
    eta = 1 # overall efficiency
    # gamma = 0 # phase shift between H,V due to group delay in aBBO
    mm = 1 # mode matching (HOM visibility)
    n_noise = 0 #5e-6 # dark count prob per pump pulse in each mode
    max_photons = 1 # maximum number of photons detected

    ### Run simulation ###

    pn_ideal = computeWalkOutput(nSteps, r, alphaSq, eta, max_photons, n_noise, BS1_scheduler, gamma_scheduler)


    # Keep only b-photon or a-photon modes (tracing-over non detected modes)
    pn_ideal_a, pn_ideal_b = traceOverModes(pn_ideal)


    ### Plotting specific outcomes ###

    # look at 1-photon a subspace
    oneFolds_ideal_a = filterProbDict(pn_ideal_a)
    # look at 1-photon b subspace
    oneFolds_ideal_b = filterProbDict(pn_ideal_b)

    if trial == 0:
        full_ideal_a = {key: [oneFolds_ideal_a[key]] for key in oneFolds_ideal_a.keys()}
        full_ideal_b = {key: [oneFolds_ideal_b[key]] for key in oneFolds_ideal_b.keys()}
    else:
        for key in oneFolds_ideal_a.keys():
            full_ideal_a[key].append(oneFolds_ideal_a[key])
        for key in oneFolds_ideal_b.keys():
            full_ideal_b[key].append(oneFolds_ideal_b[key])

sorted_keys = [key for key in full_ideal_a.keys()]
sorted_keys.sort()
mean_ideal_a = [np.mean(full_ideal_a[key]) for key in sorted_keys]
std_ideal_a = [np.std(full_ideal_a[key]) for key in sorted_keys]
mean_ideal_b = [np.mean(full_ideal_b[key]) for key in sorted_keys]
std_ideal_b = [np.std(full_ideal_b[key]) for key in sorted_keys]


# plot
plot_destdir = os.path.join(str(Path().absolute()), f'plots/theta{BS1_scheduler(-1):.2f}_gammaRNG/nsteps{nSteps}/')
if not os.path.exists(plot_destdir):
    os.makedirs(plot_destdir)
utbe_plot_list(mean_ideal_a, plot_destdir, yerr=std_ideal_a, labels=sorted_keys, postfix=f'nsteps{nSteps}_gammaSeedNONE_ntrials{trial+1}_a')
utbe_plot_list(mean_ideal_b, plot_destdir, yerr=std_ideal_b, labels=sorted_keys, postfix=f'nsteps{nSteps}_gammaSeedNONE_ntrials{trial+1}_b')

