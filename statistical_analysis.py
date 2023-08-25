import json
import numpy as np
from scipy import stats
import pandas as pd
from ordered_set import OrderedSet
import csv
import pingouin as pg
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import os
import pyreadstat
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import scikit_posthocs as sp


def load_data(file_path):
    '''
    Loading neural synchrony data. 

    Parameters: 
    - file_path: path to the file  

    Returns: 
    - data_array: a list containing numpy arrays of different SFP stages
    '''

    f = open(file_path)
    results = json.load(f)

    fp1 = []
    sf1 = []
    fp2 = []
    sf2 = []
    ru = []

    all_stages = [fp1, sf1, fp2, sf2, ru]

    participant_indices = []
    for part, stages in results.items():
        if part == '623' or part == '802':
            continue
        else:
            # Process each stage's data for the current participant
            for stage, data in stages.items():
                stage = int(stage[1]) - 1  # Convert stage number to 0-index
                all_stages[stage].append(np.nanmean(data))
            participant_indices.append(part)

    # Convert participant indices to a numpy array and remove duplicates
    participant_indices = np.array(OrderedSet(participant_indices))

    # Replace 'nan' values with 0 in sf2 list
    sf2 = [0 if str(x) == 'nan' else x for x in sf2]

    data_array = [np.array(fp1), np.array(
        sf1), np.array(fp2), np.array(sf2), np.array(ru)]

    return participant_indices, data_array


def create_csv(data_array, participant_indices, filename, folder=None):
    data_to_save = np.column_stack(
        (participant_indices, data_array[0], data_array[1], data_array[2], data_array[3], data_array[4]))

    # Define the column names
    column_names = ["participant", "fp1", "sf1", "fp2", "sf2", "reunion"]

    if folder:
        os.makedirs(folder, exist_ok=True)
        full_path = os.path.join(folder, filename)
    else:
        full_path = filename

    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        writer.writerows(data_to_save)


def descriptive_analysis(data_array, folder=None):
    descriptives = {'Mean': [],
                    'SD': [],
                    'Min': [],
                    'Max': []}

    for condition in data_array:
        descriptives['Mean'].append(round(np.mean(condition), 4))
        descriptives['SD'].append(round(np.std(condition), 4))
        descriptives['Min'].append(round(np.min(condition), 4))
        descriptives['Max'].append(round(np.max(condition), 4))

    descriptives_df = pd.DataFrame(descriptives)
    descriptives_df.insert(0, 'Condition', ['Free Play 1', 'Still Face 1', 'Free Play 2',
                                            'Still Face 2', 'Reunion'])

    if folder:
        descriptives_tex_path = os.path.join(
            folder, 'synchrony_descriptives_alpha.tex')
    else:
        descriptives_tex_path = 'synchrony_descriptives_alpha.tex'

    descriptives_df.to_latex(descriptives_tex_path, index=False)

    return descriptives_df


def prepare_data_frame(data_array):
    '''
    Prepare a DataFrame from the given data array.

    Parameters:
    - data_array: a list of numpy arrays containing different data stages

    Returns:
    - df_melted: a DataFrame with melted data
    '''
    df = pd.DataFrame({
        "FreePlay1": data_array[0],
        "StillFace1": data_array[1],
        "FreePlay2": data_array[2],
        "StillFace2": data_array[3],
        'Reunion': data_array[4]
    })

    # Create a multi-indexed dataframe
    df_melted = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[
                        'FreePlay1', 'StillFace1', 'FreePlay2', 'StillFace2', 'Reunion'])
    df_melted.columns = ['id', 'Condition', 'Synchrony']

    return df_melted


def test_normality(df_melted):
    '''
    Perform the Shapiro-Wilk normality test on data.

    Parameters:
    - df_melted: a DataFrame with melted data

    Prints:
    - Test results indicating whether the data is normally distributed
    '''

    stat, p = shapiro(df_melted["Synchrony"])

    print("Test statistic:", stat)
    if p > 0.05:
        print("p = ", p, "-> data is normally distributed")
    else:
        print("p-value = ", p, "-> data is non-normally distributed")


def fisher_z_transform(data_array):
    # Perform the Fisher z-transform
    z_fp1_alpha = np.arctanh(data_array[0])
    z_sf1_alpha = np.arctanh(data_array[1])
    z_fp2_alpha = np.arctanh(data_array[2])
    z_sf2_alpha = np.arctanh(data_array[3])
    z_ru_alpha = np.arctanh(data_array[4])

    # Load data
    df = pd.DataFrame({"FreePlay1": z_fp1_alpha,
                       "StillFace1": z_sf1_alpha,
                       "FreePlay2": z_fp2_alpha,
                       "StillFace2": z_sf2_alpha,
                       'Reunion': z_ru_alpha
                       })

    # Create a multi-indexed dataframe
    df_melted_z = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[
                          'FreePlay1', 'StillFace1', 'FreePlay2', 'StillFace2', 'Reunion'])
    df_melted_z.columns = ['id', 'Condition', 'Synchrony']

    stat, p = shapiro(df_melted_z["Synchrony"])

    print("Test statistic:", stat)
    if p > 0.05:
        print("p = ", p, "-> data is normally distributed")
    else:
        print("p-value = ", p, "-> data is non-normally distributed")


def statistical_testing(df_melted):
    '''
    Perform statistical testing on the melted DataFrame.

    Parameters:
    - df_melted: a DataFrame with melted data

    Prints:
    - Main effect results from the Friedman test
    - Post-hoc test results and computed z-values
    '''

    print('Main effect')

    print(pg.friedman(data=df_melted, dv='Synchrony',
                      within='Condition', subject='id', method='chisq'))

    def get_z_values(data, groups):
        '''
        Calculate and return the z-values for pairwise Wilcoxon tests.

        Parameters:
        - data: DataFrame containing data
        - groups: list of groups for pairwise testing

        Returns:
        - z_values: 2D array of calculated z-values
        '''

        z_values = np.zeros((len(groups), len(groups)))

        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if j >= i:
                    continue
                x = data[data['Condition'] == group1]['Synchrony']
                y = data[data['Condition'] == group2]['Synchrony']
                result = stats.wilcoxon(x, y)
                z_values[i, j] = result.statistic
                z_values[j, i] = -result.statistic

        return z_values

    #  Perform the post-hoc test and adjust p-values using the FDR method
    posthoc_result = sp.posthoc_wilcoxon(
        a=df_melted, val_col="Synchrony", group_col="Condition", p_adjust='fdr_tsbky')

    # Compute z-values
    z_values = get_z_values(df_melted, posthoc_result.index)

    print("Post-hoc test result: \n")
    print(posthoc_result)
    print('\n')
    print("Z-values:\n")
    print(z_values)


def visualize_results(df_melted, frequency, method, x_col_label, y_col_label, title, x_tick_labels, save_filename, save_folder=None):
    ax = sns.boxplot(data=df_melted, x=x_col_label, y=y_col_label, width=0.6)
    ax.set_title(f'Global {method} ' + title + f' in the {frequency} band')
    ax.set_xticklabels(x_tick_labels)
    if save_folder and save_filename:
        # Create the folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        # ax.figure.savefig(save_filename)


def linear_regression_analysis(questionnaire_data_path, synchrony_data_path, frequency='Theta', connectivity_measure='PLI'):
    questionnaire_data, meta = pyreadstat.read_sav(questionnaire_data_path)

    ids = ['T029', 'T035', 'T036', 'T049', 'T050', 'T056', 'T122', 'T137', 'T188', 'T212', 'T262', 'T287', 'T290', 'T351', 'T389', 'T435', 'T475', 'T477', 'T484', 'T487',
           'T520', 'T551', 'T563', 'T576', 'T596', 'T600', 'T623', 'T637', 'T655', 'T682', 'T684', 'T775', 'T802', 'T815', 'T882', 'T892', 'T899', 'T916', 'T917', 'T997']

    # IDs of participants from the qualtrics survey
    questionnaire_data_ids = list(questionnaire_data['ID'])
    excluded_participants = ['T475', 'T563',
                             'T623', 'T802', 'T899', 'T916', 'T917']

    # list of participant IDs to exclude because of missing data
    exclude_ids = [
        i for i in questionnaire_data_ids if i not in ids or i in excluded_participants]

    # exclude rows with the specified IDs
    questionnaire_data = questionnaire_data[~questionnaire_data['ID'].isin(
        exclude_ids)]

    questionnaire_data.to_csv('BrainsInSync_Questionnaire_Data.csv')

    synchrony = pd.read_csv(synchrony_data_path)
    still_face = synchrony[['sf1', 'sf2']].mean(axis=1)
    still_face = still_face.reset_index(drop=True)
    free_play = synchrony[['fp1', 'fp2', 'reunion']].mean(axis=1)
    free_play = free_play.reset_index(drop=True)

    difference = still_face - free_play

    postpartum_depression = np.array(questionnaire_data['EDS_mean'])
    postpartum_anxiety = np.array(questionnaire_data['PSAS_mean'])
    anxiety = np.array(questionnaire_data['PBQ_ConfidenceAnx_Mean'])

    X = np.column_stack((postpartum_depression, postpartum_anxiety, anxiety))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled)
    X_df = add_constant(X_df)
    vif = [variance_inflation_factor(X_df, i) for i in range(X_df.shape[1])]
    print(vif)
    scaled_difference = pd.DataFrame(
        scaler.fit_transform(np.array(difference).reshape(-1, 1)))

    sns.scatterplot(x=X_df[0],
                    y=scaled_difference[0]).set(title=f'{frequency} {connectivity_measure}: Questionnaire Data vs. Neural Synchrony')
    sns.scatterplot(x=X_df[1],
                    y=scaled_difference[0])
    sns.scatterplot(x=X_df[2],
                    y=scaled_difference[0])

    plt.xlabel('Scaled Questionnaire Data')
    plt.ylabel('Scaled Neural Synchrony')
    plt.legend(labels=['pospartum_depression',
               'postpartum_anxiety', 'anxiety'])

    postpartum_depression = postpartum_depression.reshape(-1, 1)
    anxiety = anxiety.reshape(-1, 1)
    postpartum_anxiety = postpartum_anxiety.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_scaled, scaled_difference)
    p_values = f_regression(X_df, scaled_difference)[1]
    print(p_values)
