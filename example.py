from pathlib import Path

from main import *

def BS1_scheduler(stepNumber):
    if stepNumber == -1:
        return -pi/4
    return -pi/4

def gamma_scheduler(stepNumber, rng=np.random.default_rng(seed=None)):
    return 0
    # return np.pi * rng.random()

# for nSteps in range(1, 11):
#     ### Paramaters ###
#     # nSteps = 10 # number of steps for walk. Sequence for walk is aBBO 45 deg -> 0 deg -> 45 deg ->...                  
#     alphaSq = 0.08 # intensity / mean photon number of coherent state
#     r = 0.12 # squeezing parameter
#     eta = 1 # overall efficiency
#     # gamma = 0 # phase shift between H,V due to group delay in aBBO
#     mm = 1 # mode matching (HOM visibility)
#     n_noise = 0 #5e-6 # dark count prob per pump pulse in each mode
#     max_photons = 1 # maximum number of photons detected

#     ### Run simulation ###

#     pn_ideal = computeWalkOutput(nSteps, r, alphaSq, eta, max_photons, n_noise, BS1_scheduler, gamma_scheduler)


#     # Keep only b-photon or a-photon modes (tracing-over non detected modes)
#     pn_ideal_a, pn_ideal_b = traceOverModes(pn_ideal)


#     ### Plotting specific outcomes ###

#     # look at 1-photon a subspace
#     oneFolds_ideal_a = filterProbDict(pn_ideal_a)
#     # look at 1-photon b subspace
#     oneFolds_ideal_b = filterProbDict(pn_ideal_b)

#     # plot
#     plot_destdir = os.path.join(str(Path().absolute()), f'new_plots/theta{BS1_scheduler(-1):.2f}_gamma0.0/nsteps{nSteps}/')
#     if not os.path.exists(plot_destdir):
#         os.makedirs(plot_destdir)
#     utbe_plot_dict(oneFolds_ideal_a, plot_destdir, postfix=f'nsteps{nSteps}_a')
#     utbe_plot_dict(oneFolds_ideal_b, plot_destdir, postfix=f'nsteps{nSteps}_b')

H_arr = np.loadtxt("hardware_output/fastAxisHistogram.csv", delimiter=",")
V_arr = np.loadtxt("hardware_output/slowAxisHistogram.csv", delimiter=",")

slice_number = 40

# time-bin window = 1.5ns (1500ps)
mask_H = lambda numpy_arr: np.logical_and(numpy_arr[:, 0] < 13.9, numpy_arr[:, 0] > 12)
mask_V = lambda numpy_arr: np.logical_and(numpy_arr[:, 0] < 19, numpy_arr[:, 0] > 16)

print(f"num H_arr bins to integrate = {np.sum(H_arr[mask_H(H_arr), 1] > 3e3)}")
# print(H_arr[mask_H(H_arr), 1])
print(f"num V_arr bins to integrate = {np.sum(V_arr[mask_V(V_arr), 1] > 3e3)}")
# print(V_arr[mask_V(V_arr), 1])
plt.figure()
# plt.bar(H_arr[mask_H(H_arr), 0], H_arr[mask_H(H_arr), 1], label='H mode')
plt.bar(V_arr[mask_V(V_arr), 0], V_arr[mask_V(V_arr), 1], label='V mode')
plt.legend()
# plt.xticks([i*20 for i in range((round(H_arr[:, 0][-1]) // 20)+1)])
plt.yscale('log')
plt.savefig('output_fig_log_from_data.png')

# plt.figure()
# plt.bar(H_arr[:, 0], H_arr[:, 1], label='H mode')
# plt.bar(V_arr[:, 0], V_arr[:, 1], label='V mode')
# plt.legend()
# plt.savefig('output_fig_linear_from_data.png')

