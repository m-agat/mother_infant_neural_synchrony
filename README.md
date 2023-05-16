# Thesis

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
