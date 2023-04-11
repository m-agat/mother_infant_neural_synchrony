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
from plv import PLV

# path to the folder with all participants
dataPath = "/home/u692590/thesis/dyad_data/preprocessed_data"
# dataPath = dataPath = os.path.join(os.getcwd(), "dyad_data/preprocessed_data")
plv_results = {}
baby_epochs = []
mom_epochs = []
for participant in sorted(os.listdir(dataPath)):
    participantPath = os.path.join(dataPath, participant)
    participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]
    for sfp_stage in sorted(os.listdir(participantPath)):
        sfp_stagePath = os.path.join(participantPath, sfp_stage)
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
        baby_epochs.append(baby.epochs)
        mom_epochs.append(mom.epochs)

for participant in sorted(os.listdir(dataPath)):
    participantPath = os.path.join(dataPath, participant)
    participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]
    random.shuffle(baby_epochs)
    random.shuffle(mom_epochs)
    stage_dict = {}
    for sfp_stage in sorted(os.listdir(participantPath)):
        sfp_stagePath = os.path.join(participantPath, sfp_stage)
        stage = re.findall('-[0-5]-', str(sfp_stagePath))[0]
        baby = random.choice(baby_epochs)
        mom = random.choice(mom_epochs)
        mne.epochs.equalize_epoch_counts([baby, mom])
        plv=PLV(baby, mom)
        plv.get_plv_alpha()
        plv.get_plv_theta()
        stage_dict[stage]={'theta': 
            plv.inter_con_theta.tolist(), 'alpha': plv.inter_con_alpha.tolist()}
    plv_results[participant_idx]=stage_dict


with open("pseudo_results.json", "w") as results_file:
    json.dump(plv_results, results_file)

    
