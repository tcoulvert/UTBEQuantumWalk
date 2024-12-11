import os

from numpy import exp, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt

import strawberryfields as sf
from strawberryfields.ops import *


def utbe_plot_list(oneFolds_ideal, plot_destdir, postfix='', yerr=None, labels=None):
    fig, ax = plt.subplots(figsize = (12,8))
    ax.bar(
       np.arange(len(oneFolds_ideal))+0.1, oneFolds_ideal, yerr=yerr,
       color='tab:blue', width=0.2
    )
    ax.set_xticks(range(len(oneFolds_ideal)))
    ax.set_xticklabels(labels, rotation=65)
    plt.title('Walk output')
    plt.ylabel('Probability')
    plt.xlabel('Detection outcome (t0,t1,t2,...)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_destdir, f'output{"_"+postfix if postfix != "" else ""}.png'))
    plt.close()
    
def utbe_plot_dict(oneFolds_ideal, plot_destdir, postfix=''):
    fig, ax = plt.subplots(figsize = (12,8))
    ax.bar(
       np.arange(len(oneFolds_ideal))+0.1, 
       oneFolds_ideal.values(), 
       color='tab:blue', width=0.2
    )
    ax.set_xticks(range(len(oneFolds_ideal)))
    ax.set_xticklabels(list(oneFolds_ideal.keys()), rotation=65)
    plt.title('Walk output')
    plt.ylabel('Probability')
    plt.xlabel('Detection outcome (t0,t1,t2,...)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_destdir, f'output{"_"+postfix if postfix != "" else ""}.png'))
    plt.close()


def generate_photon_outcomes(N, max_photons=2):
    
    '''Helper function to create a label for the possible
    detection probabilities of max_photons across N modes.
    
    For example, if N=3 and max_photons=2, we expect outcomes to be:
    {(0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,1,1), (1,1,0), (1,0,1), (2,0,0),
    (0,2,0), (0,0,2)}
    
    Input
    
    int N:             Number of detection modes
    int max_photons:   Maximum number of photons detectable across the N modes
    
    Output
    
    list outcomes:     List of N-tuples which are the detection labels.
    
    '''
    
    outcomes = []

    def generate_outcomes_helper(current_outcome, remaining_photons, current_mode):
        if current_mode == N:
            if remaining_photons >= 0:
                outcomes.append(tuple(current_outcome))
            return None

        
        for photons_in_mode in range(min(max_photons + 1, remaining_photons + 1)):
            new_outcome = current_outcome + [photons_in_mode]
            generate_outcomes_helper(new_outcome, remaining_photons - photons_in_mode, current_mode + 1)

    generate_outcomes_helper([], max_photons, 0)
    return outcomes


def normalizeProbDict(p):
    
    '''Helper function which normalizes the prob dictionary p'''
    
    p = {key: value /  sum(p.values()) for key, value in p.items()}
    
    return p
    

def filterProbDict(pDict, num_photons=1):
    
    '''Helper function which returns a normalized probDict only containing the
    num_photons subspace. Useful mainly for plotting.'''
    
    filtered_probabilities = {key: value for key, value in pDict.items() if sum(key) == num_photons}
    
    return normalizeProbDict(filtered_probabilities)

def traceOverModes(pDict):
    
    '''Helper function which returns two probDicts only containing the
    H and V photons subspace by tracing over non-detected photons.'''
    
    a_probabilities = {}
    b_probabilities = {}
    
    for key, value in pDict.items():
        a_outcome = key[::2] # keep every second element starting from idx = 0
        b_outcome = key[1::2] # keep every second element starting from idx = 1
        
        a_probabilities[a_outcome] = a_probabilities.get(a_outcome,0) + value
        b_probabilities[b_outcome] = b_probabilities.get(b_outcome,0) + value
    
    return normalizeProbDict(a_probabilities) , normalizeProbDict(b_probabilities)
    

def computeWalkOutput(nSteps, r, alphaSq, eta, max_photons, n_noise, BS1_scheduler, gamma_scheduler, etaFock=1):
    
    '''Main function which computes the walk output photon statistics, 
    including most experimental imperfections. Uses strawberryfields in the 
    Gaussian backend.
    
    The output photon statistics are stored in a dictionary pn. 
    An example entry is pn = {(2,0,0,0) : 0.23} where the key (2,0,0,0) is 
    the detection label and value 0.23 is the corresponding probability.
    The prob for a desired detection label (a,b,c,d,...) can be obtained by 
    calling pn[ (a,b,c,d,...) ].
    
    **Detection/mode labels follow this convention:
    
    (a,b,c,d,...) = (H;t0, V;t0, H;t1, V;t1, ...)
    
    The walk circuit can be modified as needed. Note that mode 0 is  
    reserved for the herald in the strwaberyfields circuit. But the herald is 
    not included in the output detection labels. Currently the walk is assumed
    to be aBBO[45deg] -> aBBO[0deg] -> aBBO[45deg] ... where 45 deg is defined
    wrt {H,V} basis.
    
    Input
    
    int nSteps:      Total number of steps for the walk.
    float r:         Squeezing parameter for TMSV source
    float alphaSq:   Coherent state intensity / mean photon number
    float eta:       Efficiency of setup (applies equal loss before all detectors)
    float gamma:     Phase between H and V pol due to BBO
    int max_photons: Max number of photons detected at output of walk.
    float eta_fock:  Transmission of heralded photon. Useful for computing walk
                     with imperfect mode matching.
    
    Output
    
    dict pn:         Normalized prob distb containing output walk statistics.
    
    
    '''        
    
    nModes = 2 * (nSteps + 1)  # Start with {|H;t0>,|V;t0>}. 
                               # Each subsequent step introduces 2 new modes
                          
    alpha = np.sqrt(alphaSq)
    
    sf.hbar = 2
    prog = sf.Program(nModes+1)  # the + 1 is due to herald mode
    eng = sf.Engine("gaussian")
    
    with prog.context as q:
        
        # Initializing states input to walk
        # Let 0 be the herald mode

        # Initializing states input to walk
        Coherent(alpha)         | q[1] # Make sure that this mode is the same as the "detection" mode (and start of modes set to vac in QW)

        # Quantum walk
        for stepNumber in range(1, nSteps+1): # stepNumber 0 does nothing... ranges from stepNumber = 0 to nSteps

            for k in range(0, stepNumber+1, 1):
              # Mix modes {a,b} with same time bin (basis change)
              theta1 = BS1_scheduler(stepNumber)
              BSgate(theta=theta1, phi=0)  | (q[2*k+1], q[2*k+2])

            for k in range(stepNumber-1, -1, -1):
              # Apply time shift to {b} modes
              BSgate(theta=pi/2, phi=gamma_scheduler(stepNumber)) | (q[2*k+2], q[2*k+4])

            if stepNumber != nSteps:
                for k in range(0, stepNumber+1, 1):
                    # Undo basis change, back to {a,b}
                    BSgate(theta=pi/4, phi=0)  | (q[2*k+1], q[2*k+2])


            # if stepNumber < nSteps:
            #   for k in range(stepNumber-1, -1, -1): ##this is the only part I don't currently understand... seems I need to fill vac from bottom up???
            #   #for k in range(0, stepNumber+1, 1):
            #     Vac          | q[2*k+1]
            #     Vac          | q[2*k+3]
        
        
        # Apply loss + dark counts to all channels (including herald!)        
        # for i in range(nModes+1):
        #     ThermalLossChannel(eta, n_noise)  | q[i]

    # Run SF engine
    results = eng.run(prog)
    state = results.state
    
    
    # Compute vacuum, 1-folds
    
    pn = {}
    
    allLabels = generate_photon_outcomes(nModes, max_photons)
   
    for k in range(len(allLabels)):
        detLabel = [0] + list(allLabels[k]) # add herald in mode 0 for computing pn
        
        # We are assuming projection onto Fock states rather than
        # click detectors... Should be an okay approx as long as <n> << 1.
        
        pn[allLabels[k]] = state.fock_prob(detLabel)
        
    return pn

