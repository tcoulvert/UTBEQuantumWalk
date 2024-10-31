import argparse
import copy
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})


def main(filenames):
    for filename in filenames:
        print(filename)
        loaded_data = np.load(filename)

        timestampArray = loaded_data['timestamps'] / 60
        histIndexArray1 = loaded_data['histIndex1']
        histDataArray1 = loaded_data['histData1']
        histIndexArray2 = loaded_data['histIndex2']
        histDataArray2 = loaded_data['histData2']

        qw_steps_1and2 = np.sum(histDataArray2, axis=0)[500:]
        qw_steps_3and4 = np.sum(histDataArray1, axis=0)[500:]

        # fig, (ax1,ax2) = plt.subplots(2)
        # ax1.plot(range(len(qw_steps_1and2)), qw_steps_1and2)
        # ax2.plot(range(len(qw_steps_3and4)), qw_steps_3and4)

        # plot_destdir = os.path.join(str(Path().absolute()), 'plots')
        # if not os.path.exists(plot_destdir):
        #     os.makedirs(plot_destdir)
        # plt.savefig(os.path.join(plot_destdir, f"QW_1to4Steps_bias{filename[filename.find('V')-4:filename.find('V')+1]}.png"))

        # Naive attempt to find time-bins #
        threshold_med = np.median(np.concatenate([qw_steps_1and2, qw_steps_3and4]))
        threshold_dev = np.median(np.abs(np.concatenate([qw_steps_1and2, qw_steps_3and4]) - threshold_med))
        threshold = threshold_med + 7*threshold_dev
        timebins = dict.fromkeys(range(len([qw_steps_1and2, qw_steps_3and4])))
        pruned_timebins = copy.deepcopy(timebins)
        for qw_idx, qw in enumerate([qw_steps_1and2, qw_steps_3and4]):
            timebins[qw_idx] = list()  # 2 b/c 2 qw steps in each histogram

            above_threshold_flag = False
            timebin_start_idx, timebin_end_idx = 0, 0

            for i in range(len(qw)-1):
                if (
                    qw[i] < threshold and not above_threshold_flag  # Cuts out standard noise
                ) or (
                    qw[i] >= threshold and qw[i+1] < threshold and not above_threshold_flag  # Ignore large single-bin noise fluctuations
                ):
                    continue
                elif qw[i] >= threshold and qw[i+1] >= threshold and not above_threshold_flag:  # Start of timebin found!
                    above_threshold_flag = True
                    timebin_start_idx = i
                elif qw[i] >= threshold and above_threshold_flag:  # Ignore bins where we're inside a timebin as we only want the bin edges
                    continue
                elif qw[i] < threshold and above_threshold_flag:  # End of timebin found!
                    above_threshold_flag = False
                    timebin_end_idx = i

                    timebins[qw_idx].append([timebin_start_idx, timebin_end_idx])

            # Pune timebins to remove spurious timebins at the end of data
            timebins[qw_idx] = np.array(timebins[qw_idx])
            timebin_start_difs = timebins[qw_idx][1:, 0] - timebins[qw_idx][:-1, 0]
            new_steps = np.arange(len(timebins[qw_idx]))[np.concatenate([[False], timebin_start_difs > 1000])]
            if len(new_steps) == 1:  # Should only be one new step b/c only 2 qw step type per histogram
                pruned_timebins[qw_idx] = {
                    0: timebins[qw_idx][:new_steps[0]],
                    1: timebins[qw_idx][new_steps[0]:]
                }
            elif len(new_steps) > 1:
                pruned_timebins[qw_idx] = {
                    0: timebins[qw_idx][:new_steps[0]],
                    1: timebins[qw_idx][new_steps[0]:new_steps[1]]
                }
            else:
                pruned_timebins[qw_idx] = {
                    0: timebins[qw_idx],
                    1: [[None, None]]
                }

        # Plot data with initial found timebin edges
        fig, (ax1,ax2) = plt.subplots(2)
        fig.suptitle('Data histograms with initial timebins before pruning')
        ax1.plot(range(len(qw_steps_1and2)), qw_steps_1and2)
        for i, (timebin_start_idx, timebin_end_idx) in enumerate(timebins[0], 1):
            ax1.vlines(timebin_start_idx, 0, np.max(qw_steps_1and2), linestyle='dashed', color=cmap_petroff10[i], label=f'idx = {timebin_start_idx}')
            ax1.vlines(timebin_end_idx, 0, np.max(qw_steps_1and2), linestyle='dashed', color=cmap_petroff10[i], label=f'idx = {timebin_end_idx}')
        ax1.legend(bbox_to_anchor=(1, 1))
        ax1.set_title('QW steps 1 and 2')
        ax2.plot(range(len(qw_steps_3and4)), qw_steps_3and4)
        for i, (timebin_start_idx, timebin_end_idx) in enumerate(timebins[1]):
            ax2.vlines(timebin_start_idx, 0, np.max(qw_steps_3and4), linestyle='dashed', color=cmap_petroff10[i], label=f'idx = {timebin_start_idx}')
            ax2.vlines(timebin_end_idx, 0, np.max(qw_steps_3and4), linestyle='dashed', color=cmap_petroff10[i], label=f'idx = {timebin_end_idx}')
        ax2.legend(bbox_to_anchor=(1, 0.2))
        ax2.set_title('QW steps 3 and 4')

        plot_destdir = os.path.join(str(Path().absolute()), 'plots')
        if not os.path.exists(plot_destdir):
            os.makedirs(plot_destdir)
        plt.savefig(os.path.join(plot_destdir, f"QW_1to4Steps_bias{filename[filename.find('V')-4:filename.find('V')+1]}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(plot_destdir, f"QW_1to4Steps_bias{filename[filename.find('V')-4:filename.find('V')+1]}.pdf"), bbox_inches='tight')
        plt.close()

        # Build prob distributions for QW steps 1-4
        prob_dists = {}
        for qw_idx, qw in enumerate([qw_steps_1and2, qw_steps_3and4]):

            prob_dists[qw_idx] = {}
            for step_idx in range(2):
                
                tbins = {}
                for tb_idx, (start_idx, end_idx) in enumerate(pruned_timebins[qw_idx][step_idx]):
                    if start_idx is None:
                        tbins[tb_idx] = 0
                    tbins[tb_idx] = np.sum(qw[start_idx:end_idx])

                total_count = np.sum([val for val in tbins.values()])
                prob_dists[qw_idx][step_idx] = np.array([val / total_count for val in tbins.values()])

        # Plot prob distributions
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)
        plt.title('Probability distributions')
        ax1.bar(range(len(prob_dists[0][0])), prob_dists[0][0], label='QW Step 1')
        ax2.bar(range(len(prob_dists[0][1])),prob_dists[0][1], label='QW Step 2')
        ax3.bar(range(len(prob_dists[1][0])),prob_dists[1][0], label='QW Step 3')
        ax4.bar(range(len(prob_dists[1][1])),prob_dists[1][1], label='QW Step 4')

        ax1.legend(bbox_to_anchor=(1, 1))
        ax1.set_title('QW step 1')
        ax2.legend(bbox_to_anchor=(1, 0.8))
        ax2.set_title('QW step 2')
        ax1.legend(bbox_to_anchor=(1, 0.6))
        ax3.set_title('QW step 3')
        ax4.legend(bbox_to_anchor=(1, 0.4))
        ax4.set_title('QW step 4')
        
        plt.savefig(os.path.join(plot_destdir, f"ProbDists_1to4Steps_bias{filename[filename.find('V')-4:filename.find('V')+1]}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(plot_destdir, f"ProbDists_1to4Steps_bias{filename[filename.find('V')-4:filename.find('V')+1]}.pdf"), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='Process the quantum random walk data.'
    # )
    # parser.add_argument('--dump', dest='output_dir_path', action='store', default=f'{str(Path().absolute())}/../output_sim/',
    #     help='Name of the output path in which the processed parquets will be stored.'
    # )
    # args = parser.parse_args()

    filenames = glob.glob(os.path.join(str(Path().absolute()), 'hardware_output/*.npz'))
    filenames.sort()
    main(filenames)

