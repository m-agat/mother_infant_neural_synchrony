
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
from plv import pseudoPLV


# # path to the folder with all participants
# dataPath = "/home/u692590/thesis/dyad_data/preprocessed_data"
dataPath = dataPath = os.path.join(os.getcwd(), "dyad_data/preprocessed_data")

# path to the folder with all participants
dataPath = "/home/u692590/thesis/dyad_data/preprocessed_data"

def create_surrogate_data():
    plv_results_theta = {}
    plv_results_alpha = {}
    for participant in sorted(os.listdir(dataPath)):
        participantPath = os.path.join(dataPath, participant)
        participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]
        stage_dict_theta = {}
        stage_dict_alpha = {}
        for sfp_stage in sorted(os.listdir(participantPath)):
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
            plv=pseudoPLV(baby.epochs, mom.epochs)
            plv.get_plv_alpha()
            plv.get_plv_theta()
            # plv_theta = np.mean(np.array(plv.inter_con_theta), axis = 0)
            # plv_alpha = np.mean(np.array(plv.inter_con_alpha), axis = 0)
            plv_theta = plv.inter_con_theta
            plv_alpha = plv.inter_con_alpha
            stage_dict_theta[stage] = plv_theta
            stage_dict_alpha[stage] = plv_alpha
        plv_results_theta[participant_idx]=stage_dict_theta
        plv_results_alpha[participant_idx]=stage_dict_alpha

        matrix_theta = np.zeros((40, 5, 12, 12))

        for i, participant in enumerate(plv_results_theta.keys()):
            for j, condition in enumerate(plv_results_theta[participant].keys()):
                matrix_theta[i, j] = np.array(plv_results_theta[participant][condition])

        matrix_alpha = np.zeros((40, 5, 12, 12))

        for i, participant in enumerate(plv_results_alpha.keys()):
            for j, condition in enumerate(plv_results_alpha[participant].keys()):
                matrix_alpha[i, j] = np.array(plv_results_alpha[participant][condition])

        return matrix_theta


theta_matrix = np.load('theta_matrix.npy')


n_surrogates = 3
pseudo_dyads_theta = [create_surrogate_data() for _ in range(n_surrogates)]
                           

significant_synchrony = np.zeros((40, 5, 12, 12))

for pseudo_dyad_matrix in pseudo_dyads_theta:
    mask = theta_matrix > pseudo_dyad_matrix
    significant_synchrony[mask] += 1

np.save('pseudo_plvs', significant_synchrony)

# with open("plv_validated_theta.json", "w") as results_file:
#     json.dump(significant_synchrony, results_file)