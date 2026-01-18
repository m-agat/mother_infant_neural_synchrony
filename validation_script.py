import cupy as cp  # Import CuPy for GPU-accelerated operations
from load_data import DataLoader, Infant, Mom
from connectivity_measures import connectivityMeasure, pseudoConnectivityMeasure
import os
import json
import re

def validate_synchrony(dataPath, mode, n_surrogates=200, batch_size=50):
    connectivity_results_theta_all = {}  # Non-validated results for theta frequency band 
    connectivity_results_alpha_all = {}  # Non-validated results for alpha frequency band 
    connectivity_results_theta = {}  # Validated results for theta frequency band 
    connectivity_results_alpha = {}  # Validated results for alpha frequency band

    # Loop over each stage (subfolder) within the participant's path
    for participant in sorted(os.listdir(dataPath)):
        participantPath = os.path.join(dataPath, participant)
        participant_idx = re.findall(r'\d+', str(participantPath)[14:])[0]

        stage_dict_theta_all = {}  # Non-validated results for theta frequency at each stage
        stage_dict_alpha_all = {}  # Non-validated results for alpha frequency at each stage
        stage_dict_theta = {}  # Validated results for theta frequency at each stage
        stage_dict_alpha = {}  # Validated results for alpha frequency at each stage

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
            common_channels = list(set(baby.epochs.info['ch_names']).intersection(set(mom.epochs.info['ch_names'])))
            baby.epochs.pick_channels(common_channels)
            mom.epochs.pick_channels(common_channels)

            # Compute synchrony
            connectivity = connectivityMeasure(baby.epochs, mom.epochs, mode)
            connectivity.calculate_sync() 
            connectivity_theta = cp.array(connectivity.theta_baby)  # Move to GPU with CuPy
            connectivity_alpha = cp.array(connectivity.alpha_baby)

            # Pre-allocate arrays for compared results
            compared_theta = cp.zeros_like(connectivity_theta)
            compared_alpha = cp.zeros_like(connectivity_alpha)

            # Initialize arrays to store batch surrogate results
            surr_connectivity_theta_batch = cp.empty((batch_size,) + connectivity_theta.shape)
            surr_connectivity_alpha_batch = cp.empty((batch_size,) + connectivity_alpha.shape)

            # Create a stream for asynchronous execution
            stream = cp.cuda.Stream()

            # Process surrogates in batches
            for batch_start in range(0, n_surrogates, batch_size):
                current_batch_size = min(batch_size, n_surrogates - batch_start)

                # Compute surrogate connectivity measures in batches asynchronously
                with stream:
                    for i in range(current_batch_size):
                        surr_connectivity = pseudoConnectivityMeasure(baby.epochs, mom.epochs, mode)
                        surr_connectivity.calculate_surrogate_sync()

                        # Store the results in the batch array
                        surr_connectivity_theta_batch[i] = cp.array(surr_connectivity.theta_baby)
                        surr_connectivity_alpha_batch[i] = cp.array(surr_connectivity.alpha_baby)

                # Synchronize the stream to ensure all computations are done
                stream.synchronize()

                # Compare the batch results with the real connectivity values
                for i in range(current_batch_size):
                    compared_theta += connectivity_theta > surr_connectivity_theta_batch[i]
                    compared_alpha += connectivity_alpha > surr_connectivity_alpha_batch[i]

            # Calculate the threshold for significant PLV
            threshold = cp.full_like(connectivity_theta, 0.95 * n_surrogates)

            # Create a mask for significant PLV in each frequency band
            mask_t = compared_theta > threshold
            mask_a = compared_alpha > threshold

            # Set non-significant theta PLV values to NaN
            connectivity_theta = cp.where(mask_t, connectivity_theta, cp.nan)
            connectivity_alpha = cp.where(mask_a, connectivity_alpha, cp.nan)

            # Move back to CPU and store in stage_dict_theta/alpha
            stage_dict_theta_all[stage] = cp.asnumpy(connectivity.theta_baby).tolist()  # Convert once at the end
            stage_dict_alpha_all[stage] = cp.asnumpy(connectivity.alpha_baby).tolist()
            stage_dict_theta[stage] = cp.asnumpy(connectivity_theta).tolist()
            stage_dict_alpha[stage] = cp.asnumpy(connectivity_alpha).tolist()

        # Store results for each participant
        connectivity_results_theta_all[participant_idx] = stage_dict_theta_all 
        connectivity_results_alpha_all[participant_idx] = stage_dict_alpha_all
        connectivity_results_theta[participant_idx] = stage_dict_theta 
        connectivity_results_alpha[participant_idx] = stage_dict_alpha

    # Save results to JSON
    with open(f"nonvalidated_results_theta_{mode}.json", "w") as results_file:
        json.dump(connectivity_results_theta_all, results_file)

    with open(f"nonvalidated_results_alpha_{mode}.json", "w") as results_file:
        json.dump(connectivity_results_alpha_all, results_file)

    with open(f"validated_results_theta_{mode}.json", "w") as results_file:
        json.dump(connectivity_results_theta, results_file)

    with open(f"validated_results_alpha_{mode}.json", "w") as results_file:
        json.dump(connectivity_results_alpha, results_file)

# Path to the folder with all participants
local_path = f"{os.getcwd()}/dyad_data/preprocessed_data/"

# Example usage
validate_synchrony(dataPath=local_path, mode="wpli")
