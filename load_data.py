import mne
import re
import os

class DataLoader:
    
    '''Loads data from the dyad file'''

    def __init__(self, path):
        self.path = path
        self.dyad_data = mne.io.read_raw_edf(
            self.path, preload=False, stim_channel='auto', verbose=False)
        self.infant_data = None
        self.mother_data = None
        self.infant_file = None
        self.mother_file = None
        self.infant_channels = []
        self.mother_channels = []
        self.infant_path = ""
        self.mother_path = ""

    def identify_person(self):

        '''Identifies the person (infant or mother) based on the channel names in the dyad data.'''

        r_i = re.compile(".*_Baby$")
        r_m = re.compile(".*_Mom$")
        self.infant_channels = [chan for chan in list(
            filter(r_i.match, self.dyad_data.info["ch_names"]))]
        self.mother_channels = [chan for chan in list(
            filter(r_m.match, self.dyad_data.info["ch_names"]))]

    def separate_files(self):

        '''Separates the dyad data into individual files for the infant and mother.'''

        idx = re.findall(r'\d+', str(self.path)[14:])[0]
        cond = re.findall('-[0-5]-', str(self.path))[0]
        self.infant_path = f"/media/agata/My Passport/dyad_data/new/split/Infant{idx}_{cond[1]}.fif"
        self.mother_path = f"/media/agata/My Passport/dyad_data/new/split/Mother{idx}_{cond[1]}.fif"
        # self.infant_path = f"{os.getcwd()}/dyad_data/split/Infant{idx}_{cond[1]}.fif"
        # self.mother_path = f"{os.getcwd()}/dyad_data/split/Mother{idx}_{cond[1]}.fif"
        self.infant_file = self.dyad_data.save(
            self.infant_path, self.infant_channels, overwrite=True)
        self.mother_file = self.dyad_data.save(
            self.mother_path, self.mother_channels, overwrite=True)

    def read_data(self):

        '''Identifies the person and separates the data into individual files.'''

        self.identify_person()
        self.separate_files()


class Participant:

    '''This class represents a participant (infant or mother). 
    It provides methods for setting the montage, subsetting channels, and obtaining epochs.'''

    def __init__(self, path):
        self.path = path
        self.data = mne.io.read_raw(self.path, preload=True, verbose=False) 
        self.epochs = None

    def set_montage(self):

        '''Sets the montage of the participant's data to 'biosemi64'.'''

        self.data.set_montage('biosemi64')

    def subset_channels(self):
        
        '''Retains only a subset of channels defined by the electrodes_to_keep list.'''

        # Define the list of electrode names you want to keep
        electrodes_to_keep = ['F3', 'F4', 'F7', 'F8',
                              'C3', 'C4', 'T7', 'T8', 'P3', 'P4', 'P7', 'P8']

        # Use pick_channels to keep only the desired electrodes
        self.data.pick_channels(electrodes_to_keep)

    def get_epochs(self):

        '''Creates epochs from the participant's data with a fixed length of epoch_duration seconds. 
        The data is then downsampled to reduce computation time.'''

        # Define the duration of the epoch (in seconds)
        epoch_duration = 2
        # Create the long epoch
        self.data.filter(3, 12)
        self.epochs = mne.make_fixed_length_epochs(
            self.data, duration=epoch_duration, preload=True)
        # Downsample the data to reduce the computation time.
        self.epochs.resample(250)


class Infant(Participant):

    '''This class represents an infant. 
    It inherits from the Participant class and provides additional methods specific to infants.
    '''
    
    def __init__(self, path):
        super().__init__(path)

    def rename_channels(self):
        '''Renames the channels in the infant's data by removing the '_Baby' suffix.'''

        old_channels = list(self.data.info["ch_names"])
        new_channels_baby = [chan[:-5] if chan[-5:] ==
                             "_Baby" else chan for chan in old_channels]
        old_to_new_names = {old: new for old,
                            new in zip(old_channels, new_channels_baby)}
        self.data.rename_channels(mapping=old_to_new_names)

    def read_data(self):
        '''Renames channels, sets the montage, subsets channels, and obtains epochs for the infant's data.'''

        self.rename_channels()
        self.set_montage()
        self.subset_channels()
        self.get_epochs()


class Mom(Participant):

    '''This class represents a mother. 
    It inherits from the Participant class and provides additional methods specific to mothers.
    '''

    def __init__(self, path):
        super().__init__(path)

    def rename_channels(self):
        '''Renames the channels in the mother's data by removing the '_Mom' suffix.'''

        old_channels = list(self.data.info["ch_names"])
        new_channels_mom = [chan[:-4] if chan[-4:] ==
                            "_Mom" else chan for chan in old_channels]
        old_to_new_names = {old: new for old,
                            new in zip(old_channels, new_channels_mom)}
        self.data.rename_channels(mapping=old_to_new_names)

    def read_data(self):
        '''Renames channels, sets the montage, subsets channels, and obtains epochs for the infant's data.'''

        self.rename_channels()
        self.set_montage()
        self.subset_channels()
        self.get_epochs()
