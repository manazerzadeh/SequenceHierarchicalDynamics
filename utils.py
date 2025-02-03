import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict

path = "./sh4/"

path_misc = "./sh4_miscs/"

total_sub_num = 16

seq_length  = 11

encoding = 'ISO-8859-1'

subjs_ids = {
    'BH', 'BI2', 'CBC', 'CD', 'CES', 'CLD', 'CM', 'CMM', 'CNB', 'CNN', 'CS', 'CWT', 'CZE', 'DH',
    'IL', 'KA', 'KF', 'KQ', 'MAK', 'MDT', 'MDX', 'MEW', 'MGC', 'MHK', 'MIB', 'MIE', 'MIL', 'MNI',
    'MOH', 'MOT', 'MTS', 'MTS2', 'MUU', 'MZT', 'QQ', 'TE', 'TI', 'UC', 'UO', 'ZQ', 'LI'
}

seq_dict = {
    'G1': {
        'aligned': [1, 2, 4], 
        'unaligned': [3, 6, 7],
        'control': [5]
    } ,
    'G2': {
        'aligned': [3, 6, 7],
        'unaligned': [1, 2, 4],
        'control': [5]
    }

}


def filter_dat_files():
    import os

    # List all files in the path directory and subdirectories
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    # all_files = os.listdir(path)

    # Filter files that contain any of the subjs_ids and have a .dat extension
    filtered_files = [file for file in all_files if file.endswith('.dat') and any(subj_id in file for subj_id in subjs_ids)]
    return filtered_files


def filter_preferred_files():

    filtered_files = filter_dat_files()

    # Create a dictionary to hold lists of files for each subject
    subject_files = defaultdict(list)

    # Iterate over the filtered files and group them by subject ID
    for file in filtered_files:
        for subj_id in subjs_ids:
            if '_' + subj_id + '.dat' in file:
                subject_files[subj_id].append(file)
                # break

    # Convert defaultdict to a regular dictionary for easier handling
    subject_files = dict(subject_files)



    # Filter preferred files for each subject
    preferred_files = []
    for subj_id, files in subject_files.items():
        data_files = [file for file in files if '_gsdata_' not in file]
        if data_files:
            preferred_files.append(data_files[0])
        else:
            preferred_files.append(files[0])

    # Print the preferred files for each subject
    for subj_id, file in zip(subject_files.keys(), preferred_files):
        print(f"Preferred file for subject {subj_id}: {file}")
    
    return subject_files, preferred_files




# def read_dat_file(path: str):
#     data = pd.read_csv(path, delimiter='\t')
#     return data


def read_dat_files_subjs_list():
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    data = []
    headers = []
    subject_files, preferred_files = filter_preferred_files()
    for subj_id, file in zip(subject_files.keys(), preferred_files):
        if subj_id == 'LI':
            dataframe = pd.read_csv(file, sep = '\t', encoding=encoding, header = None)
            dataframe = dataframe.iloc[:, :-1]
            dataframe.columns = headers
            dataframe['SubNum'] = 'IL'

        else:
            dataframe = pd.read_csv(file, sep = '\t', encoding=encoding, usecols=lambda column: not column.startswith("Unnamed"))
            headers = dataframe.columns
            dataframe['SubNum'] = subj_id
            
        data.append(dataframe)

    return data



def seq_type_mapping(row):
    if row['seqNumb'] in seq_dict[row['group']]['aligned']:
        return 'aligned'
    elif row['seqNumb'] in seq_dict[row['group']]['unaligned']:
        return 'unaligned'
    elif row['seqNumb'] in seq_dict[row['group']]['control']:
        return 'control'


def remove_no_go_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes no-go trials
    """

    return subj[subj['announce'] == 0]


def select_training_trials(subjs: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the training trials
    """

    return subjs[subjs['trialType'] == 2]



def add_IPI(subj: pd.DataFrame):
    """
    Adds interpress intervals to a subject's dataframe
    """

    for i in range(seq_length-1):
        col1 = 'pressTime'+str(i)
        col2 = 'pressTime'+str(i+1)
        new_col = 'IPI'+str(i+1)
        subj[new_col] = subj[col2] - subj[col1]

    subj['IPI0'] = subj['pressTime0']




def finger_melt_IPIs(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """

    
    subj_melted = pd.melt(subj,
                    id_vars=['BN', 'TN', 'SubNum', 'points', 'isError', 
                    'cueS', 'cueC', 'cueP', 'FT', 'seqNumb'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number', 
                    value_name='IPI_Value')
    

    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)


    

    
    return subj_melted



def finger_melt_responses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'points', 'isError', 
                    'cueS', 'cueC', 'cueP', 'FT', 'seqNumb'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('resp')],
                    var_name='Response_Number', 
                    value_name='Response_Value')
    
    subj_melted['N'] = (subj_melted['Response_Number'].str.extract('(\d+)').astype('int64') + 1)



    return subj_melted


def finger_melt(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs(subj)
    melt_responses = finger_melt_responses(subj)

    # print(melt_IPIs.shape, melt_responses.shape)
    
    # # Check duplicates in melt_IPIs
    # duplicates_IPIs = melt_IPIs.duplicated(subset=['BN', 'TN', 'SubNum', 'points', 'isError', 'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'N'], keep=False)
    # print(f"Duplicates in melt_IPIs: {melt_IPIs[duplicates_IPIs].shape[0]}")
    # print(melt_IPIs[duplicates_IPIs])

    # # Check duplicates in melt_responses
    # duplicates_responses = melt_responses.duplicated(subset=['BN', 'TN', 'SubNum', 'points', 'isError', 'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'N'], keep=False)
    # print(f"Duplicates in melt_responses: {melt_responses[duplicates_responses].shape[0]}")

    
    merged_df = melt_IPIs.merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'points',
     'isError', 'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'N'] )

    # print(merged_df.shape)

    # return melt_IPIs[duplicates_IPIs], melt_responses[duplicates_responses], merged_df
    return merged_df






def finger_melt_IPIs_treatment(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """

    
    subj_melted = pd.melt(subj,
                    id_vars=['BN', 'TN', 'SubNum', 'points', 'isError', 
                    'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'group', 'seq_type'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number', 
                    value_name='IPI_Value')
    

    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)


    

    
    return subj_melted



def finger_melt_responses_treatment(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'points', 'isError', 
                    'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'group', 'seq_type'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('resp')],
                    var_name='Response_Number', 
                    value_name='Response_Value')
    
    subj_melted['N'] = (subj_melted['Response_Number'].str.extract('(\d+)').astype('int64') + 1)



    return subj_melted


def finger_melt_treatment(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs_treatment(subj)
    melt_responses = finger_melt_responses_treatment(subj)

    # print(melt_IPIs.shape, melt_responses.shape)
    
    # # Check duplicates in melt_IPIs
    # duplicates_IPIs = melt_IPIs.duplicated(subset=['BN', 'TN', 'SubNum', 'points', 'isError', 'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'N'], keep=False)
    # print(f"Duplicates in melt_IPIs: {melt_IPIs[duplicates_IPIs].shape[0]}")
    # print(melt_IPIs[duplicates_IPIs])

    # # Check duplicates in melt_responses
    # duplicates_responses = melt_responses.duplicated(subset=['BN', 'TN', 'SubNum', 'points', 'isError', 'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'N'], keep=False)
    # print(f"Duplicates in melt_responses: {melt_responses[duplicates_responses].shape[0]}")

    
    merged_df = melt_IPIs.merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'points',
     'isError', 'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'group', 'seq_type', 'N'] )

    # print(merged_df.shape)

    # return melt_IPIs[duplicates_IPIs], melt_responses[duplicates_responses], merged_df
    return merged_df



def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes error trials from the dat file of a subject
    """

    return subj[(subj['isError'] == 0)]


def finger_melt_Forces(subjs_force: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each Finger Force in the whole experiment adding two columns, "Force_Number" determining the order of Force
    and "Force_Value" determining the time of Force
    """

    
    subj_force_melted = pd.melt(subjs_force, 
                    id_vars=['state', 'timeReal', 'time','BN', 'TN', 'SubNum', 'points', 
    'isError', 'cueS', 'cueC', 'cueP', 'FT', 'seqNumb', 'N', 'IPI0', 'MT'], 
                    value_vars =  [_ for _ in subjs_force.columns if _.startswith('force')],
                    var_name='Force_Number', 
                    value_name='Force_Value')
    
    return subj_force_melted



def cut_force(subjs_force: pd.DataFrame, side_padding) -> pd.DataFrame:
    """
    Cuts the force data to the same length as the IPI data
    """
    subjs_force = subjs_force[(subjs_force['IPI0'] <= subjs_force['time'] + side_padding) & (subjs_force['time'] <= subjs_force['IPI0'] + subjs_force['MT'] + side_padding)]
    return subjs_force



def cut_force_left(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[(subjs_force['IPI0'] >= subjs_force['time'])]
    return subjs_force


def cut_force_right(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[(subjs_force['IPI0'] + subjs_force['MT'] <= subjs_force['time'])]
    return subjs_force