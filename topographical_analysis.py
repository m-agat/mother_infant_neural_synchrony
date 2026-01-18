from load_data import DataLoader, Infant, Mom
import os
import mne
import numpy as np
import json
import re
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from hypyp import viz
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import multitest


def average_epochs(dataPath):

    # Loop over each stage (subfolder) within the participant's path
    mother_epochs = {}
    baby_epochs = {}

    for participant in sorted(os.listdir(dataPath)[:2]):
        participantPath = os.path.join(dataPath, participant)
        participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]

        stage_dict_moms = {}
        stage_dict_infants = {}

        for sfp_stage in sorted(os.listdir(participantPath)):
            sfp_stagePath = os.path.join(participantPath, sfp_stage)
            stage = re.findall('-[0-5]-', str(sfp_stagePath))[0]

            dyad = DataLoader(sfp_stagePath)
            dyad.read_data()

            baby = Infant(dyad.infant_path)
            baby.read_data()

            mom = Mom(dyad.mother_path)
            mom.read_data()

            # Drop channels that are not present in both infant and mother data
            for chan_baby in baby.epochs.info['ch_names']:
                if chan_baby not in mom.epochs.info['ch_names']:
                    baby.epochs.drop_channels(chan_baby)
            for chan_mom in mom.epochs.info['ch_names']:
                if chan_mom not in baby.epochs.info['ch_names']:
                    mom.epochs.drop_channels(chan_mom)

            stage_dict_moms[stage] = mom.epochs
            stage_dict_infants[stage] = baby.epochs

        mother_epochs[participant_idx] = stage_dict_moms
        baby_epochs[participant_idx] = stage_dict_infants

    def pad_arrays(arrays):
        max_length = np.max([np.array(subarray.get_data()).shape[0]
                            for subarray in arrays])
        padded_arrays = [np.pad(np.array(subarray.get_data()), ((0, max_length - np.array(
            subarray.get_data().shape[0])), (0, 0), (0, 0)), mode='constant') for subarray in arrays]
        return np.array(padded_arrays)

    fp1_mom = []
    sf1_mom = []
    fp2_mom = []
    sf2_mom = []
    ru_mom = []

    all_stages_mom = [fp1_mom, sf1_mom,
                      fp2_mom, sf2_mom, ru_mom]
    for part, stages in mother_epochs.items():
        if part == '623' or part == '802':
            continue
        else:
            for stage, data in stages.items():
                stage = int(stage[1]) - 1
                all_stages_mom[stage].append(data)

    fp1_mom = pad_arrays(fp1_mom)
    sf1_mom = pad_arrays(sf1_mom)
    fp2_mom = pad_arrays(fp2_mom)
    sf2_mom = pad_arrays(sf2_mom)
    ru_mom = pad_arrays(ru_mom)

    mom_epochs_data = [np.nanmean(np.array(fp1_mom), axis=0), np.nanmean(np.array(sf1_mom), axis=0),
                       np.nanmean(np.array(fp2_mom), axis=0), np.nanmean(
                           np.array(sf2_mom), axis=0),
                       np.nanmean(np.array(ru_mom), axis=0)]

    fp1_baby = []
    sf1_baby = []
    fp2_baby = []
    sf2_baby = []
    ru_baby = []

    all_stages_baby = [fp1_baby, sf1_baby,
                       fp2_baby, sf2_baby, ru_baby]

    for part, stages in baby_epochs.items():
        if part == '623' or part == '802':
            continue
        else:
            for stage, data in stages.items():
                stage = int(stage[1]) - 1
                all_stages_baby[stage].append(data)

    fp1_baby = pad_arrays(fp1_baby)
    sf1_baby = pad_arrays(sf1_baby)
    fp2_baby = pad_arrays(fp2_baby)
    sf2_baby = pad_arrays(sf2_baby)
    ru_baby = pad_arrays(ru_baby)

    baby_epochs_data = [np.nanmean(np.array(fp1_baby), axis=0), np.nanmean(np.array(sf1_baby), axis=0),
                        np.nanmean(np.array(fp2_baby), axis=0), np.nanmean(
                            np.array(sf2_baby), axis=0),
                        np.nanmean(np.array(ru_baby), axis=0)]

    with open(f"moms_epochs_averaged.pkl", "wb") as results_file:
        pickle.dump(mom_epochs_data, results_file)

    with open(f"baby_epochs_averaged.pkl", "wb") as results_file:
        pickle.dump(baby_epochs_data, results_file)

    return mom_epochs_data, baby_epochs_data


def visualize_connectivity(results_path, mom_epochs_path, baby_epochs_path, ch_names, sfreq, threshold):
    # Load data from JSON file
    with open(results_path) as f:
        results_theta = json.load(f)

    # Initialize empty lists for different stages
    fp1 = []
    sf1 = []
    fp2 = []
    sf2 = []
    ru = []

    # Populate the stage-specific lists
    for part, stages in results_theta.items():
        if part == '623' or part == '802':
            continue
        else:
            for stage, data in stages.items():
                stage = int(stage[1]) - 1
                if stage == 0:
                    fp1.append(data)
                if stage == 1:
                    sf1.append(data)
                if stage == 2:
                    fp2.append(data)
                if stage == 3:
                    if np.array(data).shape != (12, 12, 24):  # one array has shape (6, 12, 24)
                        continue
                    else:
                        sf2.append(data)
                if stage == 4:
                    ru.append(data)

    # Convert lists to numpy arrays
    fp1 = np.array(fp1)
    sf1 = np.array(sf1)
    fp2 = np.array(fp2)
    sf2 = np.array(sf2)
    ru = np.array(ru)

    # Combine sf1 and sf2
    sf_combined = np.concatenate((sf1, sf2), axis=0)

    # Combine fp1, fp2, and ru
    fp_combined = np.concatenate((fp1, fp2, ru), axis=0)

    # Create heatmaps for each stage
    stages = [fp1, sf1, fp2, sf2, ru, sf_combined, fp_combined]
    stage_labels = ['Free Play 1', 'Still Face 1', 'Free Play 2', 'Still Face 2',
                    'Reunion', "Still Face Stages Combined", "Free Play Stages Combined"]

    for stage_data, stage_label in zip(stages, stage_labels):
        mean_connectivity = np.nanmean(stage_data, axis=(0, 3))

        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_connectivity, annot=True, fmt=".2f",
                    cmap="coolwarm", xticklabels=ch_names, yticklabels=ch_names)
        plt.title(f"Mean Connectivity Between Electrodes - {stage_label}")
        plt.xlabel("Infant Electrode")
        plt.ylabel("Mother Electrode")
        plt.show()

    with open(mom_epochs_path, "rb") as pickle_file:
        mom_epochs = pickle.load(pickle_file)

    with open(baby_epochs_path, "rb") as pickle_file:
        baby_epochs = pickle.load(pickle_file)

    stages = [fp1, sf1, fp2, sf2, ru, fp_combined, sf_combined]
    stage_labels = ['Free Play 1', 'Still Face 1', 'Free Play 2',
                    'Still Face 2', 'Reunion', 'Free Play Combined', 'Still Face Combined']

    for i, stage_label in enumerate(stage_labels):
        mean_connectivity = np.nanmean(stages[i], axis=(0, 3))

        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        info.set_montage('biosemi64')

        print(stage_label)

        if stage_label == 'Free Play Combined':
            epochs_mom = mne.EpochsArray(mom_epochs[0], info)
            epochs_baby = mne.EpochsArray(baby_epochs[0], info)

            viz.viz_2D_topomap_inter(
                epochs_mom, epochs_baby, mean_connectivity, threshold=threshold, lab=True)

        elif stage_label == "Still Face Combined":
            epochs_mom = mne.EpochsArray(mom_epochs[1], info)
            epochs_baby = mne.EpochsArray(baby_epochs[1], info)

            viz.viz_2D_topomap_inter(
                epochs_mom, epochs_baby, mean_connectivity, threshold=threshold, lab=True)

        else:
            epochs_mom = mne.EpochsArray(mom_epochs[i], info)
            epochs_baby = mne.EpochsArray(baby_epochs[i], info)

            viz.viz_2D_topomap_inter(
                epochs_mom, epochs_baby, mean_connectivity, threshold=threshold, lab=True)


def visualize_difference(results_path, mom_epochs_path, baby_epochs_path, ch_names, sfreq, alpha):
    # Load data from JSON file
    with open(results_path) as f:
        results_theta = json.load(f)

    # Initialize empty lists for different stages
    fp1 = []
    sf1 = []
    fp2 = []
    sf2 = []
    ru = []

    # Populate the stage-specific lists
    for part, stages in results_theta.items():
        if part == '623' or part == '802':
            continue
        else:
            for stage, data in stages.items():
                stage = int(stage[1]) - 1
                if stage == 0:
                    fp1.append(data)
                if stage == 1:
                    sf1.append(data)
                if stage == 2:
                    fp2.append(data)
                if stage == 3:
                    if np.array(data).shape != (12, 12, 24):  # one array has shape (6, 12, 24)
                        continue
                    else:
                        sf2.append(data)
                if stage == 4:
                    ru.append(data)

    # Convert lists to numpy arrays
    fp1 = np.array(fp1)
    sf1 = np.array(sf1)
    fp2 = np.array(fp2)
    sf2 = np.array(sf2)
    ru = np.array(ru)

    '''
    Only last 4 phases
    '''

    # Combine sf1 and sf2
    sf_combined = np.concatenate((sf1, sf2), axis=0)
    mean_sf_combined = np.nanmean(sf_combined, axis=3)

    # Combine fp1, fp2, and ru
    fp_combined = np.concatenate((fp2, ru), axis=0)
    mean_fp_combined = np.nanmean(fp_combined, axis=3)

    min_samples = min(mean_fp_combined.shape[0], mean_sf_combined.shape[0])

    with open(mom_epochs_path, "rb") as pickle_file:
        mom_epochs = pickle.load(pickle_file)

    with open(baby_epochs_path, "rb") as pickle_file:
        baby_epochs = pickle.load(pickle_file)

    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    info.set_montage('biosemi64')

    epochs_mom = mne.EpochsArray(mom_epochs[0], info)
    epochs_baby = mne.EpochsArray(baby_epochs[0], info)

    t_stat, p_value = stats.ttest_rel(
        mean_sf_combined[:min_samples, :, :], mean_fp_combined[:min_samples, :, :], alternative='less', nan_policy='omit')

    reject, corrected_p_values, _, _ = multitest.multipletests(
        p_value.flatten(), alpha=alpha, method='fdr_bh')
    print(corrected_p_values)
    print('Significant differences for the last 4 phases')
    print(corrected_p_values[corrected_p_values < alpha])
    p_values_inverted = 1 - p_value

    viz.viz_2D_topomap_inter(
        epochs_mom, epochs_baby, p_values_inverted, threshold=1-alpha, lab=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(p_value, annot=True, fmt=".2f",
                cmap="coolwarm", xticklabels=ch_names, yticklabels=ch_names)
    plt.title(
        f"Significant Differences in Connectivity between Electrodes between SF and FP Conditions")
    plt.xlabel("Infant Electrode")
    plt.ylabel("Mother Electrode")
    plt.show()

    '''
    Mean values per electrode 
    '''

    # Combine sf1 and sf2
    sf_combined = np.concatenate((sf1, sf2), axis=0)
    mean_sf_combined = np.nanmean(sf_combined, axis=(0, 3))

    # Combine fp1, fp2, and ru
    fp_combined = np.concatenate((fp1, fp2, ru), axis=0)
    mean_fp_combined = np.nanmean(fp_combined, axis=(0, 3))

    min_samples = min(mean_fp_combined.shape[0], mean_sf_combined.shape[0])

    epochs_mom = mne.EpochsArray(mom_epochs[0], info)
    epochs_baby = mne.EpochsArray(baby_epochs[0], info)

    p_values = np.empty((len(ch_names), len(ch_names)))
    for el1 in range(len(ch_names)):
        for el2 in range(len(ch_names)):
            t_stat, p_value = stats.wilcoxon(
                mean_sf_combined[el1, el2], mean_fp_combined[el1, el2], alternative='less', nan_policy='omit')
            p_values[el1, el2] = p_value

    t_stat, p_value = stats.ttest_rel(
        mean_sf_combined, mean_fp_combined, alternative='less', nan_policy='omit')

    alpha = 0.05
    reject, corrected_p_values, _, _ = multitest.multipletests(
        p_value.flatten(), alpha=alpha, method='fdr_bh')
    print(corrected_p_values)
    corrected_p_values = corrected_p_values.reshape(p_value.shape)
    print('Significant differences for the mean values of the electrodes across participants')
    print(corrected_p_values[corrected_p_values < alpha])


def regional_differences(results_path, mom_epochs_path, baby_epochs_path, ch_names, sfreq, alpha):
    # Load data from JSON file
    with open(results_path) as f:
        results_theta = json.load(f)

    # Initialize empty lists for different stages
    fp1 = []
    sf1 = []
    fp2 = []
    sf2 = []
    ru = []

    # Populate the stage-specific lists
    for part, stages in results_theta.items():
        if part == '623' or part == '802':
            continue
        else:
            for stage, data in stages.items():
                stage = int(stage[1]) - 1
                if stage == 0:
                    fp1.append(data)
                if stage == 1:
                    sf1.append(data)
                if stage == 2:
                    fp2.append(data)
                if stage == 3:
                    if np.array(data).shape != (12, 12, 24):  # one array has shape (6, 12, 24)
                        continue
                    else:
                        sf2.append(data)
                if stage == 4:
                    ru.append(data)

    # Convert lists to numpy arrays
    fp1 = np.array(fp1)
    sf1 = np.array(sf1)
    fp2 = np.array(fp2)
    sf2 = np.array(sf2)
    ru = np.array(ru)

    # Topographical groupings
    frontal_region = [2, 3]
    central_region = [4, 5]
    left_temporal = [6]
    right_temporal = [7]
    # temporal_region = [6, 7]
    occipital_temporal_region = [10, 11]

    regions =  {'frontal': frontal_region, 
                'central': central_region, 
                'left temporal': left_temporal,
                'right temporal': right_temporal, 
                'occ. temporal': occipital_temporal_region
                }

    # Combine sf1 and sf2
    sf_combined = np.concatenate((sf1, sf2), axis=0)
    mean_sf_combined = np.nanmean(sf_combined, axis=3)

    # Combine fp1, fp2, and ru
    fp_combined = np.concatenate((fp1, fp2, ru), axis=0)
    mean_fp_combined = np.nanmean(fp_combined, axis=3)

    min_samples = min(mean_fp_combined.shape[0], mean_sf_combined.shape[0])

    # p_values = np.empty((len(ch_names), len(ch_names)))
    p_values = []
    for region_name_baby, region_baby in regions.items():
        # print('baby region', region_baby)
        for region_name_mom, region_mom in regions.items():
            # connectivity between baby region and mom region in sf vs in fp
            t_stat, p_value = stats.ttest_rel(
                np.nanmean(mean_sf_combined[:min_samples, region_baby[0]:region_baby[-1]+1, region_mom[0]:region_mom[-1]+1], axis=(1,2)), 
                np.nanmean(mean_fp_combined[:min_samples, region_baby[0]:region_baby[-1]+1, region_mom[0]:region_mom[-1]+1], axis=(1,2)), 
                alternative='less', nan_policy='omit')
            reject, corrected_p_value, _, _ = multitest.multipletests(
                p_value, alpha=alpha, method='fdr_bh')
            if corrected_p_value < alpha:
                print('After FDR correction')
                print('baby region: ', region_name_baby)
                print('mom region: ', region_name_mom)
                print('test statistic: ', t_stat)
                print('corrected p-value: ', corrected_p_value)
                p_values.append(corrected_p_value)


regional_differences(results_path=f"{os.getcwd()}/results/validated_results_theta_wpli.json",
                     mom_epochs_path=f"{os.getcwd()}/results/moms_epochs_averaged.pkl",
                     baby_epochs_path=f"{os.getcwd()}/results/baby_epochs_averaged.pkl",
                     ch_names=['F3', 'F4', 'F7', 'F8', 'C3', 'C4',
                               'T7', 'T8', 'P3', 'P4', 'P7', 'P8'],
                     sfreq=250,
                     alpha=0.05)

local_path = f"{os.getcwd()}/dyad_data/preprocessed_data/"
# gpu_path = "/home/u692590/thesis/dyad_data/preprocessed_data"

# mom_epochs_data, baby_epochs_data = average_epochs(dataPath=local_path)

# print(np.array(mom_epochs_data).shape)
