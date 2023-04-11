import os
import mne
import numpy as np
import scipy.signal as sig
from hypyp import analyses
from copy import copy
import json
import re
import pandas as pd

# local classes
from load_data import DataLoader
from load_data import Infant
from load_data import Mom
from plv import PLV

# path to the folder with all participants
dataPath = os.path.join(os.getcwd(), "dyad_data/preprocessed_data")
plv_results = {}
missing_channels = {}
for participant in sorted(os.listdir(dataPath)):
    participantPath = os.path.join(dataPath, participant)
    participant_idx = re.findall(r'\d+', str(participantPath))[0]
    sfp_stagePath = os.path.join(participantPath, os.listdir(participantPath)[0])
    stage = re.findall('-[0-5]-', str(sfp_stagePath))[0]
    dyad = DataLoader(sfp_stagePath)
    dyad.read_data()
    baby = Infant(dyad.infant_path)
    baby.read_data()
    mom = Mom(dyad.mother_path)
    mom.read_data()
    missing_channels[participant_idx]= []
    for chan_baby in baby.epochs.info['ch_names']:
        if chan_baby not in mom.epochs.info['ch_names']:
            missing_channels[participant_idx].append(f'Baby {chan_baby}')
    for chan_mom in mom.epochs.info['ch_names']:
        if chan_mom not in baby.epochs.info['ch_names']:
            missing_channels[participant_idx].append(f'Mom {chan_mom}')


missing_channels_df = pd.DataFrame(missing_channels)
print(missing_channels_df)