import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import csv



file_name = os.path.join(str(Path().absolute()), 'hardware_output/QW_w_switch_as_BS1_DCbias_5-00V_RF_zeros.npz')

loaded_data = np.load(file_name)

timestampArray = loaded_data['timestamps']/60
histIndexArray1 = loaded_data['histIndex1']
histDataArray1 = loaded_data['histData1']
histIndexArray2 = loaded_data['histIndex2']
histDataArray2 = loaded_data['histData2']

fig, (ax1,ax2) = plt.subplots(2)
#ax1.plot(histDataArray2[0,:])
#ax2.plot(histDataArray1[0,:])

ax1.plot(histIndexArray2[0,:],np.sum(histDataArray2,axis=0))
ax2.plot(histIndexArray1[0,:],np.sum(histDataArray1,axis=0))

ax1.set_xlim([10000,50000])
ax2.set_xlim([20000,70000])

ax1.set_ylim([0,22000])
ax2.set_ylim([0,800])

plt.savefig('test.png')


'''
fig, (ax1,ax2) = plt.subplots(2)
for frame in range(len(timestampArray)):
    ax1.clear()
    ax2.clear()

    ax1.set_xlim([0,50000])
    ax2.set_xlim([0,50000])

    ax1.set_ylim([0,700])
    ax2.set_ylim([0,25])

    ax1.plot(histIndexArray2[frame,:], histDataArray2[frame,:])
    ax2.plot(histIndexArray1[frame,:], histDataArray1[frame,:])

    plt.pause(2)

plt.show()
'''

'''
# STEP 1
step1_dist = np.zeros((len(timestampArray),2))
for i in range(len(timestampArray)):
    step1_tb1 = np.sum(histDataArray2[i,33:53])
    step1_tb2 = np.sum(histDataArray2[i,63:83])
    step1_tot = step1_tb1+step1_tb2
    #step1_dist[i,:] = np.array([step1_tb1/step1_tot, step1_tb2/step1_tot])
    step1_dist[i,:] = np.array([step1_tb1, step1_tb2])

# STEP 2
step2_dist = np.zeros((len(timestampArray),3))
for i in range(len(timestampArray)):
    step2_tb1 = np.sum(histDataArray2[i,389:409])
    step2_tb2 = np.sum(histDataArray2[i,418:438])
    step2_tb3 = np.sum(histDataArray2[i,447:467])
    step2_tot = step2_tb1+step2_tb2+step2_tb3
    #step2_dist[i,:] = np.array([step2_tb1/step2_tot, step2_tb2/step2_tot, step2_tb3/step2_tot])
    step2_dist[i,:] = np.array([step2_tb1, step2_tb2, step2_tb3])


# STEP 3
step3_dist = np.zeros((len(timestampArray),4))
for i in range(len(timestampArray)):
    step3_tb1 = np.sum(histDataArray1[i,144:164])
    step3_tb2 = np.sum(histDataArray1[i,178:198])
    step3_tb3 = np.sum(histDataArray1[i,208:228])
    step3_tb4 = np.sum(histDataArray1[i,236:256])
    step3_tot = step3_tb1+step3_tb2+step3_tb3+step3_tb4
    #step3_dist[i,:] = np.array([step3_tb1/step3_tot, step3_tb2/step3_tot, step3_tb3/step3_tot, step3_tb4/step3_tot])
    step3_dist[i,:] = np.array([step3_tb1, step3_tb2, step3_tb3, step3_tb4])

fig, (ax1,ax2,ax3) = plt.subplots(3)
ax1.plot(timestampArray[:],step1_dist[:,0])
ax1.plot(timestampArray[:],step1_dist[:,1])

ax2.plot(timestampArray[:],step2_dist[:,0])
ax2.plot(timestampArray[:],step2_dist[:,1])
ax2.plot(timestampArray[:],step2_dist[:,2])

ax3.plot(timestampArray[:],step3_dist[:,0])
ax3.plot(timestampArray[:],step3_dist[:,1])
ax3.plot(timestampArray[:],step3_dist[:,2])
ax3.plot(timestampArray[:],step3_dist[:,3])

ax1.set_xlim([0,20])
ax2.set_xlim([0,20])
ax3.set_xlim([0,20])

#ax1.set_ylim([0,1])
#ax2.set_ylim([0,1])
#ax3.set_ylim([0,1])

ax3.set(xlabel='Time (min)')

plt.show()
'''
