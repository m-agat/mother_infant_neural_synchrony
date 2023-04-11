import os
import json
import mne
import numpy as np
import scipy.signal as sig
from load_data import DataLoader
from load_data import Infant
from load_data import Mom
from hypyp import analyses
from copy import copy


class PLI:
    def __init__(self, baby_epochs, mom_epochs):
        '''
        Computes PLV for one dyad
        '''
        self.babyEpochs = baby_epochs
        self.momEpochs = mom_epochs
        self.data = np.array([np.array(baby_epochs), np.array(mom_epochs)])
        self.freq_bands = {'Theta-Baby': [3, 5],
                           'Theta-Mom': [4, 7],
                           'Alpha-Baby': [6, 9],
                           'Alpha-Mom': [8, 12]}
        self.sampling_rate = self.babyEpochs.info["sfreq"]


    def get_pli(self):
        complex_signal = analyses.compute_freq_bands(self.data, self.sampling_rate, self.freq_bands)
        self.pli = analyses.compute_sync(complex_signal, mode='pli')
        n_ch = len(self.babyEpochs.info['ch_names'])
        # pli_clean = self.pli.copy()
        # for i in range(self.pli.shape[0]):
        #     pli_clean[i] -= np.diag(np.diag(pli_clean[i]))
        self.theta_baby, self.theta_mom, self.alpha_baby, self.alpha_mom = self.pli[:, 0:n_ch, n_ch:2*n_ch]


import os
import mne
import numpy as np
import scipy.signal as sig
from hypyp import analyses
from copy import copy
import json
import re

# local classes
from load_data import DataLoader
from load_data import Infant
from load_data import Mom

'''
Compute PLIs for all dyads 
'''
# dataPath = os.path.join(os.getcwd(), "dyad_data/preprocessed_data")
dataPath = "/home/u692590/thesis/dyad_data/preprocessed_data"
pli_results = {}
for participant in sorted(os.listdir(dataPath)):
    participantPath = os.path.join(dataPath, participant)
    participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]
    stage_dict = {}
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
        pli= PLI(baby.epochs, mom.epochs)
        pli.get_pli()
        stage_dict[stage]={'theta': 
            pli.theta_baby.tolist(), 'alpha': pli.theta_mom.tolist()}
    pli_results[participant_idx]=stage_dict


with open("results_pli.json", "w") as results_file:
    json.dump(pli_results, results_file)
