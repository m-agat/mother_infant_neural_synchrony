from load_data import DataLoader, Infant, Mom
from connectivity_measures import connectivityMeasure, pseudoConnectivityMeasure
import os
import numpy as np
import json
import re
from mpi4py import MPI


def validate_synchrony(dataPath, mode):
    connectivity_results_theta_all = {} # non-validated results for theta frequency band 
    connectivity_results_alpha_all = {} # non-validated results for alpha frequency band 
    connectivity_results_theta = {}  # validated results for theta frequency band 
    connectivity_results_alpha = {}  # validated results for alpha frequency band

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Calculate workload distribution for each MPI process
    num_participants = len(os.listdir(dataPath))
    chunk_size = num_participants // size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    if rank == size - 1:
        end_idx = num_participants

    # Loop over each stage (subfolder) within the participant's path
    for idx in range(start_idx, end_idx):
        participant = sorted(os.listdir(dataPath))[idx]
        participantPath = os.path.join(dataPath, participant)
        participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]

        stage_dict_theta_all = {} # non-vaildated results for theta frequency at each stage
        stage_dict_alpha_all = {} #  non-validated results for alpha frequency at each stage
        stage_dict_theta = {} # vaildated results for theta frequency at each stage
        stage_dict_alpha = {} #  validated results for alpha frequency at each stage

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

            # Compute synchrony
            connectivity = connectivityMeasure(baby.epochs, mom.epochs, mode)
            connectivity.calculate_sync() 
            connectivity_theta = connectivity.theta_baby.copy() # n_segments x n_channels x n_channels 
            connectivity_alpha = connectivity.alpha_baby.copy()

            # Array to store counts of real sync > surr sync for theta frequenc
            compared_theta = np.zeros_like(connectivity_theta)
            compared_alpha = np.zeros_like(connectivity_alpha)

            # Calculate surrogate connectivity measures
            n_surrogates = 200
            for surr in range(n_surrogates):
                # compute surrogate connectivity measures 
                surr_connectivity = pseudoConnectivityMeasure(baby.epochs, mom.epochs, mode)
                surr_connectivity.calculate_surrogate_sync()

                # theta comparison
                surr_theta = surr_connectivity.theta_baby
                mask_theta = connectivity_theta > surr_theta
                compared_theta[mask_theta] += 1

                # alpha comparison
                surr_alpha = surr_connectivity.alpha_baby
                mask_alpha = connectivity_alpha > surr_alpha
                compared_alpha[mask_alpha] += 1

            threshold = np.full_like(connectivity_theta, 0.95*n_surrogates) # Calculate the threshold for significant PLV

            # Create a mask for significant PLV in each frequency band
            mask_t = compared_theta > threshold
            mask_a = compared_alpha > threshold

            # Set non-significant theta PLV values to NaN
            connectivity_theta = np.where(mask_t, connectivity_theta, np.nan)
            connectivity_alpha = np.where(mask_a, connectivity_alpha, np.nan)

            # Convert PLV arrays to lists and store in stage_dict_theta/alpha
            stage_dict_theta_all[stage] = connectivity.theta_baby.tolist()
            stage_dict_alpha_all[stage] = connectivity.alpha_baby.tolist()
            stage_dict_theta[stage] = connectivity_theta.tolist()
            stage_dict_alpha[stage] = connectivity_alpha.tolist()

        connectivity_results_theta_all[participant_idx] = stage_dict_theta_all 
        connectivity_results_alpha_all[participant_idx] = stage_dict_alpha_all
        connectivity_results_theta[participant_idx] = stage_dict_theta 
        connectivity_results_alpha[participant_idx] = stage_dict_alpha
    
    all_results_theta_all = comm.gather(connectivity_results_theta_all, root=0)
    all_results_alpha_all = comm.gather(connectivity_results_alpha_all, root=0)
    all_results_theta = comm.gather(connectivity_results_theta, root=0)
    all_results_alpha = comm.gather(connectivity_results_alpha, root=0)

    if rank == 0:
        combined_results_theta_all = {k: v for d in all_results_theta_all for k, v in d.items()}
        combined_results_alpha_all = {k: v for d in all_results_alpha_all for k, v in d.items()}
        combined_results_theta = {k: v for d in all_results_theta for k, v in d.items()}
        combined_results_alpha = {k: v for d in all_results_alpha for k, v in d.items()}

        with open(f"nonvalidated_results_theta_{mode}.json", "w") as results_file:
            json.dump(combined_results_theta_all, results_file)
        
        with open(f"nonvalidated_results_alpha_{mode}.json", "w") as results_file:
            json.dump(combined_results_alpha_all, results_file)

        with open(f"validated_results_theta_{mode}.json", "w") as results_file:
            json.dump(combined_results_theta, results_file)
        
        with open(f"validated_results_alpha_{mode}.json", "w") as results_file:
            json.dump(combined_results_alpha, results_file)

    
# path to the folder with all participants
# local_path = f"{os.getcwd()}/dyad_data/preprocessed_data/"
external_disk_path = "/media/agata/My Passport/dyad_data/preprocessed_data/"

validate_synchrony(dataPath=external_disk_path, mode="plv")
# validate_synchrony(dataPath=external_disk_path, mode="pli")
# validate_synchrony(dataPath=external_disk_path, mode="wpli")
