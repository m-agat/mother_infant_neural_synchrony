import os
import json
import mne
import numpy as np
import scipy.signal as sig
from load_data import DataLoader
from load_data import Infant
from load_data import Mom
from hypyp import analyses
from hypyp import utils
from copy import copy

# # path to the folder with all participants
# dataPath = os.path.join(os.getcwd(), "dyad_data/preprocessed_data")
# allDyadsDir = os.listdir(dataPath)  # folder with all participants
# dyadPath = os.path.join(dataPath, allDyadsDir[0])
# dyadDir = sorted(os.listdir(dyadPath))
# stagePath = os.path.join(dyadPath, dyadDir[4])

# dyad = DataLoader(stagePath)
# dyad.read_data()

# baby = Infant("Infant036_1.fif")
# baby.read_data()

# mom = Mom("Mother036_1.fif")
# mom.read_data()


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
        complex_signal = analyses.compute_freq_bands(self.data, self.sampling_rate, self.freq_bands)
        self.plv = analyses.compute_sync(complex_signal, mode='plv', epochs_average=False)
        n_ch = len(self.babyEpochs.info['ch_names'])
        self.theta_baby, self.theta_mom, self.alpha_baby, self.alpha_mom = self.plv[:, 0:n_ch, n_ch:2*n_ch]

    
    def get_plv_alpha(self):
        # Define frequency bands
        assert self.data[0].shape[0] == self.data[1].shape[0], "Two data streams should have the same number of trials."
        freq_bands = {'alpha': [6, 12]}
        complex_signal = []
        # filtering and hilbert transform
        for freq_band in freq_bands.values():
            filtered = np.array([mne.filter.filter_data(self.data[participant],
                                                        self.sampling_rate, l_freq=freq_band[0], h_freq=freq_band[1],
                                                        verbose=False)
                                for participant in range(2)
                                # for each participant
                                 ])
            hilb = np.angle(sig.hilbert(filtered))
            complex_signal.append(hilb)

        complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])
        
        freq_range1 = [6, 9]  # alpha frequency band for baby
        freq_range2 = [8, 12]  # alpha frequency band for mom

        n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

        complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
        transpose_axes = (0, 1, 3, 2)
        phase = complex_signal / np.abs(complex_signal)

        r = np.mean(freq_range2)/np.mean(freq_range1)
        freq_range = [np.min(freq_range1), np.max(freq_range2)]
        freqsn = freq_range
        freqsm = [f * r for f in freqsn]
        n_mult = (freqsn[0] + freqsm[0]) / (2 * freqsn[0])
        m_mult = (freqsm[0] + freqsn[0]) / (2 * freqsm[0])

        phase[:, :, :, :n_ch] = n_mult * phase[:, :, :, :n_ch]
        phase[:, :, :, n_ch:] = m_mult * phase[:, :, :, n_ch:]

        c = np.real(phase)
        s = np.imag(phase)
        dphi = analyses._multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con_alpha = abs(dphi/n_samp)
        con_alpha = np.nanmean(con_alpha, axis=1)

        n_channels = len(self.babyEpochs.info['ch_names'])
        self.inter_con_alpha = con_alpha[:, 0:n_channels, n_channels: 2*n_channels]
        


    def get_plv_theta(self):
        assert self.data[0].shape[0] == self.data[1].shape[0], "Two data streams should have the same number of trials."
        freq_bands = {'theta': [3, 7]}
        complex_signal = []
        # filtering and hilbert transform
        for freq_band in freq_bands.values():
            filtered = np.array([mne.filter.filter_data(self.data[participant],
                                                        self.sampling_rate, l_freq=freq_band[0], h_freq=freq_band[1],
                                                        verbose=False)
                                for participant in range(2)
                                # for each participant
                                 ])
            hilb = sig.hilbert(filtered)
            complex_signal.append(hilb)

        complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])

        # Define frequency bands
        
        freq_range1 = [3, 5]  # theta frequency band for baby
        freq_range2 = [4, 7]  # theta frequency band for mom

        n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

        complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
        transpose_axes = (0, 1, 3, 2)
        phase = complex_signal / np.abs(complex_signal)

        r = np.mean(freq_range2)/np.mean(freq_range1)
        freq_range = [np.min(freq_range1), np.max(freq_range2)]
        freqsn = freq_range
        freqsm = [f * r for f in freqsn]
        n_mult = (freqsn[0] + freqsm[0]) / (2 * freqsn[0])
        m_mult = (freqsm[0] + freqsn[0]) / (2 * freqsm[0])

        phase[:, :, :, :n_ch] = n_mult * phase[:, :, :, :n_ch]
        phase[:, :, :, n_ch:] = m_mult * phase[:, :, :, n_ch:]

        c = np.real(phase)
        s = np.imag(phase)
        dphi = analyses._multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con_theta = abs(dphi) / n_samp
        con_theta = np.nanmean(con_theta, axis=1)

        n_channels = len(self.babyEpochs.info['ch_names'])
        self.inter_con_theta = con_theta[:, 0:n_channels, n_channels: 2*n_channels]
        


class pseudoPLV:
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


    def get_plv(self):
        complex_signal = analyses.compute_freq_bands(self.data, self.sampling_rate, self.freq_bands)
        self.plv = analyses.compute_sync(complex_signal, mode='plv')
        n_ch = len(self.babyEpochs.info['ch_names'])
        self.theta_baby, self.theta_mom, self.alpha_baby, self.alpha_mom = self.plv[:, 0:n_ch, n_ch:2*n_ch]

    
    def get_plv_alpha(self):
        # Define frequency bands
        assert self.data[0].shape[0] == self.data[1].shape[0], "Two data streams should have the same number of trials."
        freq_bands = {'alpha': [6, 12]}
        complex_signal = []
        # filtering and hilbert transform
        for freq_band in freq_bands.values():
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
        
        freq_range1 = [6, 9]  # alpha frequency band for baby
        freq_range2 = [8, 12]  # alpha frequency band for mom

        n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

        complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
        transpose_axes = (0, 1, 3, 2)
        phase = complex_signal / np.abs(complex_signal)

        r = np.mean(freq_range2)/np.mean(freq_range1)
        freq_range = [np.min(freq_range1), np.max(freq_range2)]
        freqsn = freq_range
        freqsm = [f * r for f in freqsn]
        n_mult = (freqsn[0] + freqsm[0]) / (2 * freqsn[0])
        m_mult = (freqsm[0] + freqsn[0]) / (2 * freqsm[0])

        phase[:, :, :, :n_ch] = n_mult * phase[:, :, :, :n_ch]
        phase[:, :, :, n_ch:] = m_mult * phase[:, :, :, n_ch:]

        c = np.real(phase)
        s = np.imag(phase)
        dphi = analyses._multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con_alpha = abs(dphi) / n_samp
        con_alpha = np.nanmean(con_alpha, axis=1)

        n_channels = len(self.babyEpochs.info['ch_names'])
        self.inter_con_alpha = con_alpha[:, 0:n_channels, n_channels: 2*n_channels]
        


    def get_plv_theta(self):
        assert self.data[0].shape[0] == self.data[1].shape[0], "Two data streams should have the same number of trials."
        freq_bands = {'theta': [3, 7]}
        complex_signal = []
        # filtering and hilbert transform
        for freq_band in freq_bands.values():
            filtered = np.array([mne.filter.filter_data(self.data[participant],
                                                        self.sampling_rate, l_freq=freq_band[0], h_freq=freq_band[1],
                                                        verbose=False)
                                for participant in range(2)
                                # for each participant
                                 ])
            hilb = sig.hilbert(filtered)
            np.take(hilb,np.random.permutation(hilb.shape[1]),axis=1,out=hilb)
            complex_signal.append(hilb)

        complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])

        # Define frequency bands
        
        freq_range1 = [3, 5]  # theta frequency band for baby
        freq_range2 = [4, 7]  # theta frequency band for mom

        n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

        complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
        transpose_axes = (0, 1, 3, 2)
        phase = complex_signal / np.abs(complex_signal)

        r = np.mean(freq_range2)/np.mean(freq_range1)
        freq_range = [np.min(freq_range1), np.max(freq_range2)]
        freqsn = freq_range
        freqsm = [f * r for f in freqsn]
        n_mult = (freqsn[0] + freqsm[0]) / (2 * freqsn[0])
        m_mult = (freqsm[0] + freqsn[0]) / (2 * freqsm[0])

        phase[:, :, :, :n_ch] = n_mult * phase[:, :, :, :n_ch]
        phase[:, :, :, n_ch:] = m_mult * phase[:, :, :, n_ch:]

        c = np.real(phase)
        s = np.imag(phase)
        dphi = analyses._multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con_theta = abs(dphi) / n_samp
        con_theta = np.nanmean(con_theta, axis=1)

        n_channels = len(self.babyEpochs.info['ch_names'])
        self.inter_con_theta = con_theta[:, 0:n_channels, n_channels: 2*n_channels]
