from main import *

### Paramaters ###

nSteps = 4 # number of steps for walk. Sequence for walk is aBBO 45 deg -> 0 deg -> 45 deg ->...                  
alphaSq = 1 # intensity / mean photon number of coherent state
r = 0.12 # squeezing parameter
eta = 1 # overall efficiency
gamma = 0 # phase shift between H,V due to group delay in aBBO
mm = 1 # mode matching (HOM visibility)
n_noise = 0 #5e-6 # dark count prob per pump pulse in each mode
max_photons = 2 # maximum number of photons detected

### Run simulation ###

pn_ideal = computeWalkOutput(nSteps, r, alphaSq, eta, gamma, max_photons, n_noise)


# Keep only b-photon or a-photon modes (tracing-over non detected modes)
pn_ideal_a, pn_ideal_b = traceOverModes(pn_ideal)


### Plotting specific outcomes ###

# look at 1-photon a subspace
oneFolds_ideal_a = filterProbDict(pn_ideal_a)
# look at 1-photon b subspace
oneFolds_ideal_b = filterProbDict(pn_ideal_b)

# plot
utbe_plot(oneFolds_ideal_a, postfix=f'nsteps{nSteps}_a')
utbe_plot(oneFolds_ideal_b, postfix=f'nsteps{nSteps}_b')

