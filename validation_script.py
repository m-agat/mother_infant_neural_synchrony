from load_data import DataLoader, Infant, Mom
from plv import PLV, pseudoPLV
import os
import mne
import numpy as np
import scipy.signal as sig
import json
import re

# path to the folder with all participants
dataPath = "/home/u692590/thesis/dyad_data/preprocessed_data"
# dataPath = "/home/agata/Desktop/thesis/dyad_data/preprocessed_data/"

plv_results_theta = {}  # Dictionary to store PLV results for theta frequency band 
plv_results_alpha = {}  # Dictionary to store PLV results for alpha frequency band

# Loop over each stage (subfolder) within the participant's path
for participant in sorted(os.listdir(dataPath)):
    participantPath = os.path.join(dataPath, participant)
    participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]

    stage_dict_theta = {} # Dictionary to store PLV results for theta frequency at each stage
    stage_dict_alpha = {} #  Dictionary to store PLV results for alpha frequency at each stage

    for sfp_stage in sorted(os.listdir(participantPath)):
        sfp_stagePath = os.path.join(participantPath, sfp_stage)
        stage = re.findall('-[0-5]-', str(sfp_stagePath))[0]

        dyad = DataLoader(sfp_stagePath)
        dyad.read_data()

        baby = Infant(dyad.infant_path)
        baby.read_data()

        mom = Mom(dyad.mother_path)
        mom.read_data()

        # Drop channels that are not present in both infant and mother data
        for chan_baby in baby.epochs.info['ch_names']:
            if chan_baby not in mom.epochs.info['ch_names']:
                baby.epochs.drop_channels(chan_baby)
        for chan_mom in mom.epochs.info['ch_names']:
            if chan_mom not in baby.epochs.info['ch_names']:
                mom.epochs.drop_channels(chan_mom)

        # Calculate the PLV
        plv=PLV(baby.epochs, mom.epochs)
        plv.get_plv() 
        plv_theta = plv.theta_baby # n_segments x n_channels x n_channels 
        plv_alpha = plv.alpha_baby

        # Array to store counts of real PLV > surr PLV for theta frequenc
        compared_theta = np.zeros_like(plv_theta)
        compared_alpha = np.zeros_like(plv_alpha)

        # Calculate surrogate PLVs
        n_surrogates = 200
        for surr in range(n_surrogates):
            # compute surrogate PLVs
            surrPLV = pseudoPLV(baby.epochs, mom.epochs)
            surrPLV.get_plv()

            # theta comparison
            surr_theta = surrPLV.theta_baby
            mask_theta = plv_theta > surr_theta
            compared_theta[mask_theta] += 1

            # alpha comparison
            surr_alpha = surrPLV.alpha_baby
            mask_alpha = plv_alpha > surr_alpha
            compared_alpha[mask_alpha] += 1

        threshold = np.full_like(plv_theta, 0.95*n_surrogates) # Calculate the threshold for significant PLV

        # Create a mask for significant PLV in each frequency band
        mask_t = compared_theta > threshold
        mask_a = compared_alpha > threshold

        # Set non-significant theta PLV values to NaN
        plv_theta = np.where(mask_t, plv_theta, np.nan)
        plv_alpha = np.where(mask_a, plv_alpha, np.nan)


        # Convert PLV arrays to lists and store in stage_dict_theta/alpha
        stage_dict_theta[stage] = plv_theta.tolist()
        stage_dict_alpha[stage] = plv_alpha.tolist()

    plv_results_theta[participant_idx] = stage_dict_theta 
    plv_results_alpha[participant_idx] = stage_dict_alpha


with open("results_theta_plv.json", "w") as results_file:
    json.dump(plv_results_theta, results_file)
    
with open("results_alpha_plv.json", "w") as results_file:
    json.dump(plv_results_alpha, results_file)

    