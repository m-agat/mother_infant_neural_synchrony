# The Impact of Smartphone Distraction on Mother-Infant Neural Synchrony During Social Interactions
## Agata Mosińska - Thesis 2023
&nbsp;
# :briefcase: Data Loading

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

# :nerd_face: PLV Calculation

## **plv.py**

This script defines two classes, `PLV` and `pseudoPLV`, which are used to compute phase locking value (PLV) for EEG data. The `PLV` class calculates PLV for a given dyad, while the `pseudoPLV` class calculates PLV with shuffling of the Hilbert transform.

### Dependencies

- os
- json
- mne
- numpy (as np)
- scipy.signal (as sig)
- load_data module (importing `DataLoader`, `Infant`, and `Mom` classes)
- hypyp module (importing `analyses` and `utils`)

### Classes

#### :triangular_ruler:`PLV`

- Description: Computes PLV for one dyad.
- Methods:
    - `__init__(self, baby_epochs, mom_epochs)`: Initializes the `PLV` object.
    - `get_plv(self)`: Calculates single-frequency PLV.
    - `get_plv_alpha(self)`: Calculates cross-frequency alpha PLV.
    - `get_plv_theta(self)`: Calculates cross-frequency theta PLV.

#### :straight_ruler: `pseudoPLV`

- Description: Computes PLV for one dyad with shuffling of the Hilbert transform.
- Methods:
    - `__init__(self, baby_epochs, mom_epochs)`: Initializes the `pseudoPLV` object.
    - `get_plv(self)`: Calculates single-frequency PLV with shuffling of the Hilbert transform.
    - `get_plv_alpha(self)`: Calculates cross-frequency alpha PLV with shuffling of the Hilbert transform.
    - `get_plv_theta(self)`: Calculates cross-frequency theta PLV with shuffling of the Hilbert transform.

&nbsp;

# :bar_chart: Validation & Statistical Analysis

## **validation_script.py**

This code performs a Phase Locking Value (PLV) analysis on preprocessed data of infant-mother pairs. It calculates PLV values for theta and alpha frequency bands and compares them to surrogate PLV values. The results are stored in JSON files.

### Usage

1. Import the required modules and classes: `load_data`, `Infant`, `Mom`, `PLV`, `pseudoPLV`, `os`, `mne`, `numpy`, `scipy.signal`, `json`, and `re`.

2. Set the `dataPath` variable to the path of the folder containing the preprocessed data for all participants.

3. Create empty dictionaries `plv_results_theta` and `plv_results_alpha` to store the PLV results for theta and alpha bands, respectively.

4. Iterate over each participant in the `dataPath` folder.

5. For each participant, retrieve the participant's path and extract the participant index.

6. Create empty dictionaries `stage_dict_theta` and `stage_dict_alpha` to store the PLV results for each stage in the theta and alpha bands, respectively.

7. Iterate over each stage folder for the current participant.

8. For each stage, retrieve the stage's path and extract the stage number.

9. Load the data for the infant and mother using the `DataLoader`, `Infant`, and `Mom` classes.

10. Drop channels from the infant and mother data that are not present in both datasets.

11. Calculate the PLV values for the theta and alpha bands using the `PLV` class.

12. Initialize placeholders for counting how many times the real PLV values exceed surrogate PLV values.

13. Perform a loop to generate surrogate PLV values and compare them to the real PLV values.

14. Within each iteration of the surrogate loop, compare the theta and alpha PLV values to the surrogate values.

15. Update the count matrices for theta and alpha based on the comparison results.

16. Calculate a threshold based on the number of surrogates.

17. Apply the threshold to create masks for theta and alpha PLV values.

18. Replace values in the PLV matrices with NaN where the masks are False.

19. Store the PLV results for the current stage in the corresponding stage dictionaries.

20. Store the stage dictionaries for theta and alpha PLV results in the participant's dictionaries.

21. Serialize the theta PLV results dictionary into a JSON file named "results_theta_plv.json".

22. Serialize the alpha PLV results dictionary into a JSON file named "results_alpha_plv.json".

### Dependencies

- The code relies on the `load_data` module, which provides the `DataLoader`, `Infant`, and `Mom` classes.
- Other dependencies include `mne`, `numpy`, `scipy.signal`, `json`, and `re`.

### Results

- The PLV results for theta and alpha frequency bands are stored in separate JSON files: "results_theta_plv.json" and "results_alpha_plv.json", respectively.
- The JSON files contain nested dictionaries with the following structure:
  - Participant index (e.g., "participant_1"):
    - Stage number (e.g., "stage_1"):
      - PLV values for theta band (list of matrices)
      - PLV values for alpha band (list of matrices)

### Note

- Make sure to adjust the `dataPath` variable to the correct path of the folder containing the preprocessed data.

&nbsp;

## **statistical_analysis.ipynb**
This notebook performs a statistical analysis on the modified Still Face Paradigm conditions:

1. The notebook imports necessary libraries for statistical analysis, including `json`, `numpy`, `matplotlib.pyplot`, `scipy.stats`, `hypyp.stats`, `seaborn`, `statannot`, `scikit_posthocs`, and `pandas`.

2. The notebook loads data from a JSON file named 'results_alpha_plv.json' and extracts specific data for different conditions of the Still Face Paradigm. The extracted data is stored in separate lists for each condition.

3. Descriptive statistics are calculated for each condition, including mean, standard deviation, minimum, and maximum values. The results are stored in a pandas DataFrame named `descriptives_df` and exported to a LaTeX file named 'synchrony_descriptives_alpha.tex'.

4. The normality of the data distribution is tested using the Shapiro-Wilk test. The test is performed on the synchronized data values.

5. The Fisher z-transform is applied to the synchronized data values.

6. The Friedman test is conducted to analyze the main effect and determine if there are significant differences between the population means of the conditions.

7. Post hoc analysis is performed using the Wilcoxon signed-rank test to compare the conditions pairwise. The results are displayed in a boxplot with significance annotations.

8. The process is repeated for the theta band (3-5 Hz) analysis using a different JSON file named 'results_theta_plv.json'.

9. Descriptive statistics are calculated for the theta band data, and the results are stored in a pandas DataFrame named `descriptives_df` and exported to a LaTeX file named 'synchrony_descriptives_theta.tex'.
