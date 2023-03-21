# Core 
import io

# Data Science
import numpy as np
import scipy 
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hypyp.ext.mpl3d import glm
from hypyp.ext.mpl3d.mesh import Mesh
from hypyp.ext.mpl3d.camera import Camera

# MNE 
import mne

# HyPyP
from hypyp import prep 
from hypyp import analyses
from hypyp import stats
from hypyp import viz
from hypyp import utils

# From Mother-InfantEEG_complete notebook
import os
from copy import copy
from collections import OrderedDict
from mne.datasets import eegbci
from mne import Epochs, pick_types, find_events, pick_types, set_eeg_reference

# other
from copy import copy
import re 
import json 

# Loading the data 

# Identify the channels corresponding to the mother and child
def identify_person(dyad_data):
    r_i = re.compile(".*_Baby$")
    r_m = re.compile(".*_Mom$")
    infant_channels = [chan for chan in list(filter(r_i.match, dyad_data.info["ch_names"]))]
    mother_channels = [chan for chan in list(filter(r_m.match, dyad_data.info["ch_names"]))]

    return infant_channels, mother_channels


# Create and save mother and infant data in two separate files
def separate_files(dyad_path, i_channels, m_channels):
    dyad_data = mne.io.read_raw_edf(dyad_path, preload = False, stim_channel = 'auto', verbose = False)
    idx = re.findall(r'\d+', str(dyad_path)[13:])[0]
    infant_path = f"Infant{idx}_RestingState.fif"
    mother_path = f"Mother{idx}_RestingState.fif"
    infant_file = dyad_data.save(infant_path, i_channels, overwrite = True)
    mother_file = dyad_data.save(mother_path, m_channels, overwrite = True)
    return infant_file, mother_file, infant_path, mother_path


# Rename channels: delete empty electrodes and remove _Baby and _Mom from channel names
# Takes as input separated infant and mother files
def rename_channel_names(data, baby):
    #old_channels = list(filter(lambda x: "EMPTY" not in x, data.info["ch_names"]))
    old_channels = list(data.info["ch_names"])
    if baby is True:
        new_channels_baby = [chan[:-5] if chan[-5:] == "_Baby"else chan for chan in old_channels]
        old_to_new_names = {}
        for old, new in zip(old_channels, new_channels_baby):
            old_to_new_names[old] = new
        data.rename_channels(mapping = old_to_new_names)
    else:
        new_channels_mom = [chan[:-4] if chan[-4:] == "_Mom" else chan for chan in old_channels]
        old_to_new_names = {}
        for old, new in zip(old_channels, new_channels_mom):
            old_to_new_names[old] = new

        data.rename_channels(mapping = old_to_new_names)
    
    return data

# Combine the functions above to read the data 
def read_data(dyad_path):
    # Read the dyad data 
    print(dyad_path)
    dyad_data = mne.io.read_raw_edf(dyad_path, preload = False, stim_channel = 'auto', verbose = False)
    
    # Identify channels belonging to the infant and mother
    infant_channels, mother_channels = identify_person(dyad_data)

    # Separate the files based on the channels 
    infant_file, mother_file, infant_path, mother_path = separate_files(dyad_path, infant_channels, mother_channels)

    # Read the newly created files 
    infant_data = mne.io.read_raw(infant_path, preload = True, verbose = False)
    mother_data = mne.io.read_raw(mother_path, preload = True, verbose = False)

    # Rename the channels (remove _Baby and _Mom from channel names)
    rename_channel_names(infant_data, True)
    rename_channel_names(mother_data, False)

    # Set montage
    infant_data.set_montage('biosemi64') 
    mother_data.set_montage('biosemi64') 

    return infant_file, mother_file, dyad_data, infant_data, mother_data

# infant_file, mother_file, dyad_data, rawBaby, rawMom = read_data(os.path.join(firstDyadPath, firstDyadDir[4]))

# Epoch the data
def get_epochs(rawBaby, rawMom):
    # Define the duration of the epoch (in seconds)
    epoch_duration = rawBaby.times[-1] - rawBaby.times[0]  # Duration of the continuous data

    # Create the long epoch
    rawBabyEpochs = mne.make_fixed_length_epochs(rawBaby, duration=epoch_duration, preload=True)
    rawMomEpochs = mne.make_fixed_length_epochs(rawMom, duration=epoch_duration, preload=True)

    # # Downsample the data to reduce the computation time. Using a 512 Hz rate or higher should be enough 
    # # for most high frequency analyses (Turk et al., 2022)
    # print('Original sampling rate:', rawBabyEpochs.info['sfreq'], 'Hz')
    rawBabyEpochs.resample(250)
    rawMomEpochs.resample(250)
    # print('New sampling rate:', rawBabyEpochs.info['sfreq'], 'Hz')
    return rawBabyEpochs, rawMomEpochs

# rawBabyEpochs, rawMomEpochs = get_epochs(rawBaby, rawMom)
# rawMomEpochs.info

def set_parameters():
    # overlap: theta - 4-5, alpha - 8-9
    freq_bands = {'Theta-Baby': [3, 5], 
                'Theta-Mom': [4, 7],
                'Alpha-Baby': [6, 9], 
                'Alpha-Mom': [8, 12]} 
                
    full_freq = { 'full_frq': [3, 12]}

    return freq_bands

freq_bands = set_parameters()

def compute_plv(rawBabyEpochs, rawMomEpochs, freq_bands):
    data_inter = np.array([rawBabyEpochs, rawMomEpochs])
    print(data_inter.shape)
    plv = analyses.pair_connectivity(data = data_inter, sampling_rate=rawBabyEpochs.info["sfreq"], frequencies = freq_bands, mode = "plv", epochs_average = True)
    return plv

# plv = compute_plv(rawBabyEpochs, rawMomEpochs, freq_bands)

# The diagonal elements of the matrix correspond to the PLV of each channel with itself and are always equal to 1
# Let's remove them 
def clean_plvmatrix(plv):
    plv_clean = plv.copy()
    for i in range(4):
        plv_clean[i] -= np.diag(np.diag(plv_clean[i]))
    return plv_clean


def get_plv_perband(rawBabyEpochs, plv_clean):
    n_ch = len(rawBabyEpochs.info['ch_names'])
    theta_baby, theta_mom, alpha_baby, alpha_mom = plv_clean[:, 0:n_ch, n_ch:2*n_ch]
    return theta_baby, theta_mom, alpha_baby, alpha_mom

dataPath = os.path.join(os.getcwd(), "dyad_data/preprocessed_data") # path to the folder with all participants 
#dataPath = "/home/u692590/thesis/dyad_data/preprocessed_data"
allDyadsDir = os.listdir(dataPath) # folder with all participants 
firstDyadPath = os.path.join(dataPath, allDyadsDir[0]) # path to the first dyad
firstDyadDir = os.listdir(firstDyadPath) # folder of the first dyad
freq_bands = set_parameters()

# create a dictionary per each phase of the experiment to store the data for each participant and frequency band
# {participant: {stage: {'theta_baby':theta_baby, ...}}}
results = {}
# iterating over dyads
non_equal_channels = []
for i, file in enumerate(allDyadsDir): # loop over mother-infant pairs folders
    dyadPath = os.path.join(dataPath, allDyadsDir[i]) 
    dyadDir = sorted(os.listdir(dyadPath))
    stage_dict = {}
    # iterating over stages 
    for j, dyad in enumerate(dyadDir): # loop over each dyad's (sorted) folder
        stagePath = os.path.join(dyadPath, dyadDir[j])
        infant_file, mother_file, dyad_data, rawBaby, rawMom = read_data(stagePath)
        rawBabyEpochs, rawMomEpochs = get_epochs(rawBaby, rawMom)
        if len(rawBabyEpochs.info["ch_names"]) < 64 or len(rawBabyEpochs.info["ch_names"]) < 64:
            non_equal_channels.append((f"person {i}", f"stage{j+1}"))
            break
        plv = compute_plv(rawBabyEpochs, rawMomEpochs, freq_bands)
        plv_clean = clean_plvmatrix(plv)
        theta_baby, theta_mom, alpha_baby, alpha_mom = get_plv_perband(rawBabyEpochs, plv_clean)
        stage_dict[j+1] = {'theta_baby': theta_baby.tolist(), 'theta_mom': theta_mom.tolist(), 'alpha_baby': alpha_baby.tolist(), 'alpha_mom': alpha_mom.tolist()}
    if len(stage_dict) == 0:
        continue
    else:
        results[i] = stage_dict

print(non_equal_channels)
# save results to json 
with open("results.json", "w") as output_file:
    json.dump(results, output_file)
    
