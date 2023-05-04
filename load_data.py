import io
import mne
import re 

class DataLoader:
    def __init__(self, path):
        '''Load dyad data'''
        self.path = path
        self.dyad_data = mne.io.read_raw_edf(self.path, preload = False, stim_channel = 'auto', verbose = False)
        self.infant_data = None
        self.mother_data = None
        self.infant_file = None
        self.mother_file = None
        self.infant_channels = []
        self.mother_channels = []
        self.infant_path = ""
        self.mother_path = ""

    def identify_person(self):
        r_i = re.compile(".*_Baby$")
        r_m = re.compile(".*_Mom$")
        self.infant_channels = [chan for chan in list(filter(r_i.match, self.dyad_data.info["ch_names"]))]
        self.mother_channels = [chan for chan in list(filter(r_m.match, self.dyad_data.info["ch_names"]))]

    def separate_files(self):
        idx = re.findall(r'\d+', str(self.path)[14:])[0]
        cond = re.findall('-[0-5]-', str(self.path))[0]
        self.infant_path = f"Infant{idx}_{cond[1]}.fif"
        self.mother_path = f"Mother{idx}_{cond[1]}.fif"
        self.infant_file = self.dyad_data.save(self.infant_path, self.infant_channels, overwrite = True)
        self.mother_file = self.dyad_data.save(self.mother_path, self.mother_channels, overwrite = True)
    
    def read_data(self):
        self.identify_person()       
        self.separate_files()


class Participant:
    def __init__(self, path):
        self.path = path
        self.data = mne.io.read_raw(self.path, preload = True, verbose = False)
        self.epochs = None # shape: (n_epochs, n_channels, n_samples_per_epoch)

    def set_montage(self):
        self.data.set_montage('biosemi64') 

    def subset_channels(self):
        # Define the list of electrode names you want to keep
        electrodes_to_keep = ['F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'T7', 'T8', 'P3', 'P4', 'P7', 'P8']

        # Use pick_channels to keep only the desired electrodes
        self.data.pick_channels(electrodes_to_keep)

    def get_epochs(self):
        # Define the duration of the epoch (in seconds)
        epoch_duration = 2 #self.data.times[-1] - self.data.times[0]  # Duration of the continuous data
        # Create the long epoch
        self.data.filter(3, 12)
        self.epochs = mne.make_fixed_length_epochs(self.data, duration=epoch_duration, preload=True)
        # # Downsample the data to reduce the computation time. 
        self.epochs.resample(250)
        
    
class Infant(Participant):
    def __init__(self, path):
        super().__init__(path)

    def rename_channels(self):
        old_channels = list(self.data.info["ch_names"])
        new_channels_baby = [chan[:-5] if chan[-5:] == "_Baby" else chan for chan in old_channels]
        old_to_new_names = {old: new for old, new in zip(old_channels, new_channels_baby)}
        self.data.rename_channels(mapping=old_to_new_names)
        
    def read_data(self):
        self.rename_channels()
        self.set_montage()
        self.subset_channels()
        self.get_epochs()


class Mom(Participant):
    def __init__(self, path):
        super().__init__(path)

    def rename_channels(self):
        old_channels = list(self.data.info["ch_names"])
        new_channels_mom = [chan[:-4] if chan[-4:] == "_Mom" else chan for chan in old_channels]
        old_to_new_names = {old: new for old, new in zip(old_channels, new_channels_mom)}
        self.data.rename_channels(mapping=old_to_new_names)

    def read_data(self):
        self.rename_channels()
        self.set_montage()
        self.subset_channels()
        self.get_epochs()
