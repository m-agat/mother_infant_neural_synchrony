# The Impact of Smartphone Distraction on Mother-Infant Neural Synchrony During Social Interactions :woman_feeding_baby::brain:
## Agata Mosińska - Thesis 2023

### Abstract 
Face-to-face interactions between mothers and infants are crucial for infant development, however, these days, they are often disrupted by mother's smartphone distraction. This study aimed to investigate the effects of maternal smartphone distraction on mother-infant brain-to-brain synchrony. Thirty-three mother-infant pairs participated in a modified Still Face Paradigm, incorporating smartphone distraction. Dual-EEG was employed to measure mother-infant neural synchrony which was subsequently quantified using phase locking value analysis.

The analysis focused on the alpha and theta frequency bands, known for their role in social interactions. Results revealed that the Still Face Paradigm disrupted mother-infant synchrony in the theta frequency band, while no notable differences between conditions were found in the alpha band. These findings align with previous research emphasizing the theta band's sensitivity to the disruption of interaction compared to the alpha band.

The study found no evidence of the experiment's progression impacting neural synchrony, contradicting prior research. This suggests that mother-infant synchrony can be restored when the mother re-engages in the interaction. 

Overall, the findings underline the importance of minimizing technological distractions to promote healthy neural synchrony and foster positive mother-infant relationships.

&nbsp;
## Table of contents 
- [Data Loading](#data-loading)
- [Connectivity Measures](#connectivity-measures)
- [Validation](#validation)
- [Statistical Analysis](#statistical-analysis)

&nbsp;
# Data Loading 

## **load_data.py**

### :file_folder: DataLoader 

This class is used to load dyad data. It provides methods for identifying the person (infant or mother), separating the data into individual files, and reading the data.

#### `__init__(self, path)`

Initializes the DataLoader object.

- `path`: The path to the dyad data.

#### `identify_person()`

Identifies the person (infant or mother) based on the channel names in the dyad data.

#### `separate_files()`

Separates the dyad data into individual files for the infant and mother.

#### `read_data()`

Identifies the person and separates the data into individual files.

---

### :superhero_woman: Participant 

This class represents a participant (infant or mother). It provides methods for setting the montage, subsetting channels, and obtaining epochs.


#### `__init__(self, path)`

Initializes the Participant object.

- `path`: The path to the participant's data.

#### `set_montage()`

Sets the montage of the participant's data to 'biosemi64'.

#### `subset_channels()`

Retains only a subset of channels defined by the `electrodes_to_keep` list.

#### `get_epochs()`

Creates epochs from the participant's data with a fixed length of `epoch_duration` seconds. The data is then downsampled to reduce computation time.

---

### :baby: Infant (inherits from Participant) 

This class represents an infant participant. It inherits from the Participant class and provides additional methods specific to infants.

#### `__init__(self, path)`

Initializes the Infant object.

- `path`: The path to the infant's data.

#### `rename_channels()`

Renames the channels in the infant's data by removing the '_Baby' suffix.

#### `read_data()`

Renames channels, sets the montage, subsets channels, and obtains epochs for the infant's data.

---

### :woman_feeding_baby: Mom (inherits from Participant) 

This class represents a mother participant. It inherits from the Participant class and provides additional methods specific to mothers.

#### `__init__(self, path)`

Initializes the Mom object.

- `path`: The path to the mother's data.

#### `rename_channels()`

Renames the channels in the mother's data by removing the '_Mom' suffix.

#### `read_data()`

Renames channels, sets the montage, subsets channels, and obtains epochs for the mother's data.

&nbsp;

# Connectivity Measures 

## **connectivity_measures.py**

This script defines two classes, connectivityMeasure and pseudoConnectivityMeasure, which are used to compute different functional connectivity measures for EEG data. The connectivityMeasure class calculates connectivity measures for a given dyad, while the pseudoConnectivityMeasure class calculates connectivity measures with shuffling of the Hilbert transform.

### Dependencies
- mne
- numpy (as np)
- scipy.signal (as sig)
- hypyp module (importing analyses)

### Classes
#### :triangular_ruler:connectivityMeasure
- Description: Computes a functional connectivity measure for one dyad.
- Methods:
    - __init__(self, baby_epochs, mom_epochs, mode): Initializes the connectivityMeasure object.
    - calculate_sync(self): Calculates the specified connectivity measure for different frequency bands.

#### :straight_ruler: pseudoConnectivityMeasure
- Description: Computes a connectivity measure for one dyad with shuffling of the Hilbert transform.
- Methods:
    - __init__(self, baby_epochs, mom_epochs, mode): Initializes the pseudoConnectivityMeasure object.
    - calculate_surrogate_sync(self): Calculates the specified connectivity measure with shuffling of the Hilbert transform.

&nbsp;

# Validation 

## **validation_script.py**

This script processes EEG data to validate synchrony using different connectivity measures. It calculates and validates the functional connectivity measures (PLV, PLI, and wPLI) between infant and mother EEG data at different stages and frequency bands.

### Dependencies
- load_data module (importing DataLoader, Infant, and Mom classes)
- connectivity_measures module (importing connectivityMeasure and pseudoConnectivityMeasure classes)
- os
- numpy (as np)
- json
- re

### Functions
#### :white_check_mark: validate_synchrony(dataPath, mode)
- Description: Validates synchrony between infant and mother EEG data using different connectivity measures.
- Parameters:
    - dataPath (str): Path to the folder containing data for all participants.
    - mode (str): The connectivity measure mode ('plv', 'pli', or 'wpli').
    - Returns: None

### Note
- Make sure to adjust the `dataPath` variable to the correct path of the folder containing the preprocessed data.

&nbsp;

# Statistical Analysis

## **statistical_analysis.py**
This script performs statistical analysis and visualization of EEG synchrony data along with questionnaire data related to postpartum depression and anxiety.

### Dependencies
- json
- numpy (as np)
- scipy.stats (as stats)
- pandas (as pd)
- OrderedSet from ordered_set
- csv
- pingouin (as pg)
- shapiro from scipy.stats
- matplotlib.pyplot (as plt)
- os
- pyreadstat
- seaborn (as sns)
- LinearRegression from sklearn.linear_model
- f_regression from sklearn.feature_selection
- MinMaxScaler from sklearn.preprocessing
- variance_inflation_factor from statsmodels.stats.outliers_influence
- add_constant from statsmodels.tools.tools
- scikit_posthocs (as sp)

### Functions
#### :open_file_folder: load_data(file_path)
- Description: Loads neural synchrony data from a JSON file.
- Parameters:
    - file_path (str): Path to the JSON file containing synchrony data.
    - Returns: Tuple containing participant indices and a list of numpy arrays for different SFP stages.
      
#### :floppy_disk: create_csv(data_array, participant_indices, filename, folder=None)
- Description: Creates a CSV file from the given data array.
- Parameters:
    - data_array (list): List of numpy arrays containing synchrony data for different stages.
    - participant_indices (numpy.array): Participant indices.
    - filename (str): Name of the output CSV file.
    - folder (str, optional): Folder path to save the CSV file.
    - Returns: None
#### :bar_chart: descriptive_analysis(data_array, folder=None)
- Description: Performs descriptive analysis on the data and generates a LaTeX table.
- Parameters:
    - data_array (list): List of numpy arrays containing synchrony data for different stages.
    - folder (str, optional): Folder path to save the LaTeX table.
    - Returns: DataFrame containing descriptives.
#### :arrow_right_hook: prepare_data_frame(data_array)
- Description: Prepares a DataFrame from the given data array.
- Parameters:
    - data_array (list): List of numpy arrays containing synchrony data for different stages.
    - Returns: Melted DataFrame with columns "id", "Condition", and "Synchrony".
#### :heavy_check_mark: test_normality(df_melted)
- Description: Performs the Shapiro-Wilk normality test on the data.
- Parameters:
    - df_melted (DataFrame): Melted DataFrame with columns "id", "Condition", and "Synchrony".
    - Returns: None
#### :chart_with_upwards_trend: fisher_z_transform(data_array)
- Description: Performs Fisher z-transform on the data and tests normality.
- Parameters:
    - data_array (list): List of numpy arrays containing synchrony data for different stages.
    - Returns: None
#### :bar_chart: visualize_results(df_melted, frequency, method, x_col_label, y_col_label, title, x_tick_labels, save_filename, save_folder=None)
- Description: Creates and saves a boxplot visualization of results.
- Parameters:
    - df_melted (DataFrame): Melted DataFrame with columns "id", "Condition", and "Synchrony".
    - frequency (str): Frequency band (e.g., "Theta").
    - method (str): Connectivity measure (e.g., "PLI").
    - x_col_label (str): Label for the x-axis column.
    - y_col_label (str): Label for the y-axis column.
    - title (str): Title for the plot.
    - x_tick_labels (list): Labels for x-axis ticks.
    - save_filename (str): Name of the file to save the plot.
    - save_folder (str, optional): Folder path to save the plot.
    - Returns: None
#### :bar_chart: linear_regression_analysis(questionnaire_data_path, synchrony_data_path, frequency='Theta', connectivity_measure='PLI')
- Description: Performs linear regression analysis and scatterplot visualization.
- Parameters:
    - questionnaire_data_path (str): Path to the questionnaire data file.
    - synchrony_data_path (str): Path to the synchrony data file.
    - frequency (str, optional): Frequency band (default: "Theta").
    - connectivity_measure (str, optional): Connectivity measure (default: "PLI").
    - Returns: None

## ** topographical_analysis.py**

This script performs tomographical analysis on EEG data, including averaging epochs and visualizing connectivity.

### Dependencies
- DataLoader, Infant, Mom from load_data
- os
- mne
- numpy (as np)
- json
- re
- pickle
- pandas (as pd)
- seaborn (as sns)
- matplotlib.pyplot (as plt)
- viz from hypyp

### Functions
#### :chart_with_upwards_trend: average_epochs(dataPath)
- Description: Averages epochs for infant and mother EEG data across different stages.
- Parameters:
    - dataPath (str): Path to the directory containing EEG data.
    - Returns: A tuple containing averaged mother epochs data and averaged infant epochs data.
      
#### :bar_chart: visualize_connectivity(results_path, mom_epochs_path, baby_epochs_path, ch_names, sfreq, threshold)
- Description: Visualizes connectivity using heatmaps and topographic maps.
- Parameters:
    - results_path (str): Path to the JSON file containing connectivity results.
    - mom_epochs_path (str): Path to the pickled file containing averaged mother epochs data.
    - baby_epochs_path (str): Path to the pickled file containing averaged infant epochs data.
    - ch_names (list): List of channel names.
    - sfreq (int): Sampling frequency of the EEG data.
    - threshold (float): Threshold for topographic map visualization.
    - Returns: None
