import os
import mne
import numpy as np
import scipy.signal as sig
from hypyp import analyses
from copy import copy
import json
import re
import pickle
import random

# local classes
from load_data import DataLoader
from load_data import Infant
from load_data import Mom
# from plv import PLV


class PLV:
    def __init__(self, baby_epochs, mom_epochs):
        '''
        Computes PLV for one dyad
        '''
        self.babyEpochs = baby_epochs
        self.momEpochs = mom_epochs
        self.data = np.array([baby_epochs.get_data(), mom_epochs.get_data()])
        self.freq_bands = {'Theta-Baby': [3, 5],
                           'Theta-Mom': [4, 7],
                           'Alpha-Baby': [6, 9],
                           'Alpha-Mom': [8, 12]}
        self.sampling_rate = self.babyEpochs.info["sfreq"]

    def get_plv(self):
        # computing the analytic signal using the hilbert transform
        complex_signal = analyses.compute_freq_bands(
            self.data, self.sampling_rate, self.freq_bands)
        self.plv = analyses.compute_sync(
            complex_signal, mode='plv', epochs_average=False) 
        n_ch = len(self.babyEpochs.info['ch_names']) 
        # slicing the last two dimensions to get the inter-connectivity values
        self.theta_baby, self.theta_mom, self.alpha_baby, self.alpha_mom = self.plv[
            :, :, 0:n_ch, n_ch:2*n_ch]


# path to the folder with all participants
dataPath = "/home/u692590/thesis/dyad_data/preprocessed_data"
thesisPath = "/home/u692590/thesis"
# dataPath = "/home/agata/Desktop/thesis/dyad_data/preprocessed_data/"
# thesisPath = os.getcwd()

# Define pattern to find mothers' files
mom_fp1 = [f for f in os.listdir(
    thesisPath) if re.match(r"Mother\d+_1\.fif", f)]
mom_sf1 = [f for f in os.listdir(
    thesisPath) if re.match(r"Mother\d+_2\.fif", f)]
mom_fp2 = [f for f in os.listdir(
    thesisPath) if re.match(r"Mother\d+_3\.fif", f)]
mom_sf2 = [f for f in os.listdir(
    thesisPath) if re.match(r"Mother\d+_4\.fif", f)]
mom_ru = [f for f in os.listdir(
    thesisPath) if re.match(r"Mother\d+_5\.fif", f)]

mom_files = [mom_fp1, mom_sf1, mom_fp2, mom_sf2, mom_ru]

plv_results_theta = {}
plv_results_alpha = {}
all_nans_theta = {}
all_nans_alpha = {}
for participant in sorted(os.listdir(dataPath)):
    participantPath = os.path.join(dataPath, participant)
    participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]
    stage_dict_theta = {}
    stage_dict_alpha = {}
    nan_per_condition_theta = {}
    nan_per_condition_alpha = {}
    for i, sfp_stage in enumerate(sorted(os.listdir(participantPath))):
        sfp_stagePath = os.path.join(participantPath, sfp_stage)
        stage = re.findall('-[0-5]-', str(sfp_stagePath))[0]
        dyad = DataLoader(sfp_stagePath)
        dyad.read_data()
        baby = Infant(dyad.infant_path)
        baby.read_data()
        mom = Mom(dyad.mother_path)
        mom.read_data()
        for chan_baby in baby.epochs.info['ch_names']:
            if chan_baby not in mom.epochs.info['ch_names']:
                baby.epochs.drop_channels(chan_baby)
        for chan_mom in mom.epochs.info['ch_names']:
            if chan_mom not in baby.epochs.info['ch_names']:
                mom.epochs.drop_channels(chan_mom)
        plv = PLV(baby.epochs, mom.epochs)
        plv.get_plv()
        # plv_theta = np.mean(np.array(plv.inter_con_theta), axis = 0)
        # plv_alpha = np.mean(np.array(plv.inter_con_alpha), axis = 0)
        plv_theta = plv.theta_baby
        plv_alpha = plv.alpha_baby

        # placeholders for the counts how many time real PLV > surr PLV
        compared_theta = np.zeros_like(plv_theta)
        compared_alpha = np.zeros_like(plv_alpha)
        n_surrogates = 1
        for surr in range(n_surrogates):
            # compute surrogate PLVs
            print([file for file in mom_files[i] if file != dyad.mother_path])
            mom = Mom(random.choice([file for file in mom_files[i] if file != dyad.mother_path]))
            mom.read_data()
            mne.epochs.equalize_epoch_counts([baby.epochs, mom.epochs])
            surrPLV = PLV(baby.epochs, mom.epochs)
            surrPLV.get_plv()

            # theta comparison
            surr_theta = surrPLV.theta_baby
           
            # Calculate the difference in shape between the arrays
            diff = np.subtract(plv_theta.shape, surr_theta.shape)

            # Pad the second array with zeros to match the shape of the first array
            surr_theta = np.pad(surr_theta, ((0, diff[0]), (0, diff[1]), (0, diff[2])), mode='constant', constant_values=0)

            mask_theta = plv_theta >= surr_theta
            compared_theta[mask_theta] += 1

            # alpha comparison
            surr_alpha = surrPLV.alpha_baby
            # Calculate the difference in shape between the arrays
            diff = np.subtract(plv_alpha.shape, surr_alpha.shape)
            # Pad the second array with zeros to match the shape of the first array
            surr_alpha = np.pad(surr_alpha, ((0, diff[0]), (0, diff[1]), (0, diff[2])), mode='constant', constant_values=0)

            mask_alpha = plv_alpha >= surr_alpha
            compared_alpha[mask_alpha] += 1

        threshold = np.full_like(plv_theta, 0.95*n_surrogates)
        mask_t = compared_theta >= threshold
        mask_a = compared_alpha >= threshold
        plv_theta = np.where(mask_t, plv_theta, np.nan)
        plv_alpha = np.where(mask_a, plv_alpha, np.nan)

        stage_dict_theta[stage] = plv_theta.tolist()
        stage_dict_alpha[stage] = plv_alpha.tolist()
        nan_per_condition_theta[stage] = np.count_nonzero(np.isnan(np.array(plv_theta)))/(np.count_nonzero(np.isnan(np.array(plv_theta))) + np.count_nonzero(~np.isnan(np.array(plv_theta))))
        nan_per_condition_alpha[stage] = np.count_nonzero(np.isnan(np.array(plv_alpha)))/(np.count_nonzero(np.isnan(np.array(plv_alpha))) + np.count_nonzero(~np.isnan(np.array(plv_alpha))))

    plv_results_theta[participant_idx] = stage_dict_theta
    plv_results_alpha[participant_idx] = stage_dict_alpha
    all_nans_theta[participant_idx] = nan_per_condition_theta
    all_nans_alpha[participant_idx] = nan_per_condition_alpha


with open("results_theta_validated_epochs.json", "w") as results_file:
    json.dump(plv_results_theta, results_file)

with open("results_alpha_validated_epochs.json", "w") as results_file:
    json.dump(plv_results_alpha, results_file)
