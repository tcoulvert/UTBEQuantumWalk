from pathlib import Path

from main import *

def BS1_scheduler(stepNumber):
    if stepNumber == -1:
        return -pi/4
    return -pi/4

# def gamma_scheduler(stepNumber, rng=np.random.default_rng(seed=None)):
#     return 0
#     # return np.pi * rng.random()

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

qw_windows = [(0, 50), (50, 90), (100, 140), (150, 190), (200, 240), (250, 290), (300, 340), (350, 390), (400, 440), (450, 490)]

# time-bin window = 0.65ns (650ps), time steps of 0.05ns (50ps) -> 13 timesteps per timebin
mask_HV = lambda numpy_arr, low, high: np.logical_and(numpy_arr[:, 0] > low, numpy_arr[:, 0] < high)

for step_i, (low, high) in enumerate(qw_windows):
    plt.figure()
    plt.bar(H_arr[mask_HV(H_arr, low, high), 0], H_arr[mask_HV(H_arr, low, high), 1], label='H mode')
    plt.bar(V_arr[mask_HV(V_arr, low, high), 0], V_arr[mask_HV(V_arr, low, high), 1], label='V mode')
    plt.legend()
    plt.hlines([np.mean(H_arr[mask_HV(H_arr, low, high), 1][:100]), np.mean(V_arr[mask_HV(V_arr, low, high), 1][:100])], [low, low], [high, high], label=['H noise floor', 'V noise floor'])
    plt.yscale('log')
    plt.savefig(f'output_data_step{step_i+1}.png')

# plt.figure()
# plt.bar(H_arr[:, 0], H_arr[:, 1], label='H mode')
# plt.bar(V_arr[:, 0], V_arr[:, 1], label='V mode')
# plt.yscale('log')
# plt.savefig('output_data.png')

# plt.figure()
# plt.bar(H_arr[:, 0], H_arr[:, 1], label='H mode')
# plt.bar(V_arr[:, 0], V_arr[:, 1], label='V mode')
# plt.legend()
# plt.savefig('output_data_linear.png')


def find_local_max(qw_idx, arr):
   
    qw_window_bool = np.logical_and(arr[:, 0] > qw_windows[qw_idx][0], arr[:, 0] < qw_windows[qw_idx][1])
    noise_floor, noise_std = np.mean(arr[qw_window_bool, 1][:100]), 5*np.std(arr[qw_window_bool, 1][:100])

    sub_array = arr[:, 1]
    local_max_idxs = [0 for _ in range(qw_idx+2)]

    for timestep in range(1, len(sub_array) - 1):
        if not qw_window_bool[timestep]:
            continue

        if (
            sub_array[timestep] > noise_floor+noise_std
            and sub_array[timestep] > sub_array[timestep-1]
            and sub_array[timestep] > sub_array[timestep+1]
        ):
            local_max_idxs[0] = timestep
            break

    for timebin in range(1, qw_idx+2):
        local_max_idxs[timebin] = local_max_idxs[0] + (timebin * (29 if timebin%4 != 0 else 30))

    return local_max_idxs, noise_floor, noise_std


nstep_walks_arr_of_dicts = []
for nstep_walk in range(1, 11):
    nstep_walks_arr_of_dicts.append({})


    plt.figure()
    for mode_name, mode in [('H', H_arr), ('V', V_arr)]:
        local_maxima, noise_floor, noise_std = find_local_max(nstep_walk-1, mode)

        integrated_timebins = []
        for timebin in range(nstep_walk+1):
            integrated_timebin = np.sum(
                mode[
                    local_maxima[timebin] - 6 : local_maxima[timebin] + 7
                ]
            ) - np.sum([noise_floor for _ in range(13)])
            integrated_timebin = integrated_timebin if integrated_timebin > 0 else 0
            integrated_timebins.append(integrated_timebin)

        normalized_timebins = [float(timebin / np.sum(integrated_timebins)) for timebin in integrated_timebins]
        nstep_walks_arr_of_dicts[-1][mode_name] = normalized_timebins

        
        plt.bar(mode[mask_HV(mode, qw_windows[nstep_walk-1][0], qw_windows[nstep_walk-1][1]), 0], mode[mask_HV(mode, qw_windows[nstep_walk-1][0], qw_windows[nstep_walk-1][1]), 1], label=mode_name+' mode')
        plt.vlines([mode[:, 0][local_max] for local_max in local_maxima], [0 for _ in local_maxima], [np.max(mode[mask_HV(mode, qw_windows[nstep_walk-1][0], qw_windows[nstep_walk-1][1]), 1]) for _ in local_maxima], label=['local max' for local_max in local_maxima], color=['black' for local_max in local_maxima])
        # plt.vlines([mode[:, 0][local_max-6] for local_max in local_maxima], [0 for _ in local_maxima], [np.max(mode[mask_HV(mode, qw_windows[nstep_walk-1][0], qw_windows[nstep_walk-1][1]), 1]) for _ in local_maxima], color=['red' for local_max in local_maxima])
        # plt.vlines([mode[:, 0][local_max+6] for local_max in local_maxima], [0 for _ in local_maxima], [np.max(mode[mask_HV(mode, qw_windows[nstep_walk-1][0], qw_windows[nstep_walk-1][1]), 1]) for _ in local_maxima], color=['blue' for local_max in local_maxima])
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'output_data_step{nstep_walk}_integrated.png')

    plot_destdir = os.path.join(str(Path().absolute()), f'new_plots/theta{BS1_scheduler(-1):.2f}_gamma0.0/nsteps{nstep_walk}/')
    if not os.path.exists(plot_destdir):
        os.makedirs(plot_destdir)
    utbe_plot_list(nstep_walks_arr_of_dicts[-1]['H'], plot_destdir, postfix=f'DATA_nsteps{nstep_walk}_a', labels=range(len(nstep_walks_arr_of_dicts[-1]['H'])))
    utbe_plot_list(nstep_walks_arr_of_dicts[-1]['V'], plot_destdir, postfix=f'DATA_nsteps{nstep_walk}_b', labels=range(len(nstep_walks_arr_of_dicts[-1]['V'])))

    print('='*60)
    print('='*60)
    print(f"QWalk nsteps = {nstep_walk}")
    print('-'*60)
    print(nstep_walks_arr_of_dicts[-1])
    
print('='*60)
print('='*60)
print(nstep_walks_arr_of_dicts)

        

        


