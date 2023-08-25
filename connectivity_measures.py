import mne
import numpy as np
import scipy.signal as sig
from hypyp import analyses


class connectivityMeasure:
    '''
    Computes a functional connectivity measure for one dyad
    '''
    def __init__(self, baby_epochs, mom_epochs, mode):
        self.babyEpochs = baby_epochs
        self.momEpochs = mom_epochs
        self.mode = mode # plv, pli, wpli
        self.data = np.array([baby_epochs.get_data(), mom_epochs.get_data()])
        self.freq_bands = {'Theta-Baby': [3, 5],
                           'Theta-Mom': [4, 7],
                           'Alpha-Baby': [6, 9],
                           'Alpha-Mom': [8, 12]}
        self.sampling_rate = self.babyEpochs.info["sfreq"]


    def calculate_sync(self):
        complex_signal = analyses.compute_freq_bands(self.data, self.sampling_rate, self.freq_bands)
        if self.mode == 'plv':
            self.plv = analyses.compute_sync(complex_signal, mode='plv', epochs_average=False)
            sync_data = self.plv
        elif self.mode == 'pli':
            self.pli = analyses.compute_sync(complex_signal, mode='pli', epochs_average=False)
            sync_data = self.pli
        elif self.mode == 'wpli':
            self.wpli = analyses.compute_sync(complex_signal, mode='wpli', epochs_average=False)
            sync_data = self.wpli
        else:
            raise ValueError("Invalid mode provided")

        n_ch = len(self.babyEpochs.info['ch_names'])
        self.theta_baby, self.theta_mom, self.alpha_baby, self.alpha_mom = sync_data[:, 0:n_ch, n_ch:2*n_ch]


class pseudoConnectivityMeasure:
    '''
    Computes PLV for one dyad shuffling the Hilbert transform to introduce randomization
    '''
    def __init__(self, baby_epochs, mom_epochs, mode):
        self.babyEpochs = baby_epochs
        self.momEpochs = mom_epochs
        self.mode = mode
        self.data = np.array([np.array(baby_epochs), np.array(mom_epochs)])
        self.freq_bands = {'Theta-Baby': [3, 5],
                           'Theta-Mom': [4, 7],
                           'Alpha-Baby': [6, 9],
                           'Alpha-Mom': [8, 12]}
        self.sampling_rate = self.babyEpochs.info["sfreq"]


    def calculate_surrogate_sync(self):
        '''
        Method calculating the single frequency connectivity measure with shuffling of the Hilbert transform
        The code was adapted from the HyPyP library code
        '''
        # filtering and hilbert transform
        complex_signal = []
        for freq_band in self.freq_bands.values():
            filtered = np.array([mne.filter.filter_data(self.data[participant],
                                                        self.sampling_rate, l_freq=freq_band[0], h_freq=freq_band[1],
                                                        verbose=False)
                                for participant in range(2)
                                # for each participant
                                ])
            hilb = np.angle(sig.hilbert(filtered))
            np.take(hilb,np.random.permutation(hilb.shape[1]),axis=1,out=hilb)
            complex_signal.append(hilb)

        complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])

        if self.mode == 'plv':
            self.plv = analyses.compute_sync(complex_signal, mode='plv', epochs_average=False)
            sync_data = self.plv
        elif self.mode == 'pli':
            self.pli = analyses.compute_sync(complex_signal, mode='pli', epochs_average=False)
            sync_data = self.pli
        elif self.mode == 'wpli':
            self.wpli = analyses.compute_sync(complex_signal, mode='wpli', epochs_average=False)
            sync_data = self.wpli
        else:
            raise ValueError("Invalid mode provided")

        n_ch = len(self.babyEpochs.info['ch_names'])
        self.theta_baby, self.theta_mom, self.alpha_baby, self.alpha_mom = sync_data[:, 0:n_ch, n_ch:2*n_ch]

        