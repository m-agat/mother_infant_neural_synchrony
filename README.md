# Thesis

### **load_data.py**
This code defines classes and functions for loading and processing data from dyad recordings in the MNE (Magnetoencephalography and Electroencephalography) library.

The `DataLoader` class is responsible for loading dyad data from an EDF file. It initializes with a path to the EDF file and loads the raw data using the `mne.io.read_raw_edf` function. The `identify_person` method uses regular expressions to identify the channels corresponding to the infant and mother in the data. The `separate_files` method extracts the data for the infant and mother into separate files by saving them as FIF files. Finally, the `read_data` method calls the previous two methods to identify the persons in the data and separate the files.

The `Participant` class represents a generic participant in the dyad recording. It initializes with a path to the participant's data file and loads the data using the `mne.io.read_raw` function. The `set_montage` method sets the electrode montage for the data. The `subset_channels` method selects a subset of channels to keep based on a predefined list. The `get_epochs` method filters the data, creates epochs of a fixed duration, and resamples them.

The `Infant` class extends the `Participant` class and adds additional functionality specific to the infant participant. The `rename_channels` method renames the channels by removing the "_Baby" suffix from their names. The `read_data` method calls the necessary methods to preprocess the infant data, including renaming channels, setting the montage, selecting channels, and creating epochs.

The `Mom` class also extends the `Participant` class and provides functionality specific to the mother participant. The `rename_channels` method removes the "_Mom" suffix from the channel names, and the `read_data` method preprocesses the mother data.


### **plv.py**
The code defines two classes, `PLV` and `pseudoPLV`, for computing Phase-Locking Value (PLV) measures for a dyad consisting of a baby and a mom. 

The `PLV` class computes PLV for a given dyad. It initializes with `baby_epochs` and `mom_epochs`, which represent the EEG epochs for the baby and mom, respectively. The class calculates PLV for the given dyad by following these steps:

1. Constructs a data array containing the EEG data for both the baby and mom.
2. Defines frequency bands of interest, including theta and alpha bands for both the baby and mom.
3. Computes complex-valued analytic signals using the `compute_freq_bands` function from the `hypyp.analyses` module.
4. Computes PLV measures using the `compute_sync` function from the `hypyp.analyses` module, with the mode set to 'plv' and epochs averaging disabled.
5. Extracts PLV values for each frequency band of interest (theta and alpha) for both the baby and mom.

The `pseudoPLV` class is similar to `PLV` but includes an additional step of shuffling the phases of the analytic signals before calculating PLV. This step is performed for the alpha and theta frequency bands.

plv_preprocessed.ipynb - notebook with computations of PLVs for preprocessed data (1 dyad) <br>
gpu_code.py - python file with code that iterates over all dyads available 
