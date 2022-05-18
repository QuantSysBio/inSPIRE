""" Helpful functions used across the module.
"""
import pickle

import pandas as pd

from inspire.constants import (
    CHARGE_KEY,
    PEPTIDE_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
)

def get_ox_flag(mods_df):
    """ Function to get the flag for oxidation of methionine from the PTMs DataFrame.

    Parameters
    ----------
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs found in the data.

    Returns
    -------
    ox_flag : int
        The flag for oxidation of methionine.
    """
    try:
        ox_flag = mods_df[mods_df[PTM_NAME_KEY] == 'Oxidation (M)'].index[0] + 1
    except IndexError:
        ox_flag = -1

    return ox_flag

def get_mokapot_weights(output_folder, caravan_step):
    """ Function to get the weights of mokapot models.

    Parameters
    ----------
    output_folder : str
        The folder where all inSPIRE output is written.
    caravan_step : str
        Indicator of which step fo inSPIRE the model was trained for.

    Results
    -------
    weights_df : pd.DataFrame
        A DataFrame of the weights for different features.
    """
    for fold_idx in range(1, 4):
        with open(
            f'{output_folder}/{caravan_step}.mokapot.model_fold-{fold_idx}.pkl',
            'rb',
        ) as model_file:
            model = pickle.load(model_file)
        weights = model.estimator.coef_[0]
        if fold_idx == 1:
            n_feats = len(weights)
            feat_weights = {f'feat{idx}': [] for idx in range(1, n_feats+1)}
        for feat_idx, name in enumerate(feat_weights):
            feat_weights[name].append(round(weights[feat_idx], 2))

    return pd.DataFrame(feat_weights)

def modify_sequence_for_skyline(df_row, mod_weights):
    """ Helper function to modify the peptide sequence so that it is compatitable
        with Skyline input.

    Parameters
    ----------
    df_row : pd.Series
        A row of the DataFrame.
    mod_weights : dict
        A dictionary mapping ptm IDs to molecular weights.

    Returns
    -------
    modified_sequence : str
        The modified sequence in the correct format for Skyline.
    """
    if not isinstance(df_row[PTM_SEQ_KEY], str):
        return df_row[PEPTIDE_KEY]
    modified_sequence = ''
    for idx, entry in enumerate(df_row[PEPTIDE_KEY]):
        mod = df_row[PTM_SEQ_KEY][idx+2]
        if mod != '0':
            mod_wt = mod_weights[int(mod)]
            if mod_wt > 0:
                modified_sequence += f'{entry}[+{round(mod_wt, 1)}]'
            else:
                modified_sequence += f'{entry}[{round(mod_wt, 1)}]'
        else:
            modified_sequence += entry
    return modified_sequence

def read_distiller_log(distiller_log):
    """ Function to read the source file names from a distiller log.

    Parameters
    ----------
    distiller_log : str
        The location of distiller output giving original source names.
    """
    source_files = []
    with open(distiller_log, 'r', encoding='UTF-8') as distiller_output:
        line = distiller_output.readline()
        while line:
            if line.startswith('Raw file '):
                source = line.split('/')[-1].split('\\')[-1].split('.raw')[0]
                source_files.append(source)
            line = distiller_output.readline()
    return source_files

def remove_source_suffixes(source):
    """ Helper function to remove raw, mzML or mgf suffixes from source name.

    Parameters
    ----------
    source : str
        The name of a source file.

    Returns
    -------
    source : str
        The updated name with a suffix removed.
    """
    if source.endswith('.mzML'):
        return source[:-5]
    if source.endswith('.raw') or source.endswith('.mgf'):
        return source[:-4]
    return source

def filter_for_prosit(search_df):
    """ Function to filter sequences not suitable for Prosit input.

    Parameters
    ----------
    search_df : pd.DataFrame
        A DataFrame of search results from an ms search engine.

    Returns
    -------
    search_df : pd.DataFrame
        The input DataFrame with sequences not suitable for Prosit input removed.
    """
    search_df = search_df.dropna(subset=[PEPTIDE_KEY, CHARGE_KEY])
    search_df_filter = search_df[PEPTIDE_KEY].apply(
        lambda x : isinstance(x, str) and 'U' not in x and len(x) > 6 and len(x) < 31
    )
    search_df = search_df[search_df_filter]
    search_df = search_df[search_df[CHARGE_KEY] < 7]
    search_df.reset_index(drop=True, inplace=True)
    return search_df

def convert_prosit_ion(ion_code):
    """ Function to convert msp ion name to its code for ion type
        and its fragment index.

    Parameters
    ----------
    ion_code : str
        The ion code as found in Prosit msp output.

    Returns
    -------
    ion_type_loss : str
        A string containing the ion type, its charge and any loss.
    fragment_idx : int
        The index of the fragment.
    """
    data = ion_code.split('-')
    if len(data) > 1:
        loss = data[1]
    else:
        loss = ''
    data2 = data[0].split('^')
    if len(data2) > 1:
        charge = data2[1]
    else:
        charge = 1
    ion_letter = data2[0][0]
    fragment_idx = int(data2[0][1:])
    return f'{ion_letter}^{charge}-{loss}', fragment_idx-1

def construct_ion_code(ion_type_loss, fragment_idx):
    """ Function to calculate the ion code used in Prosit output based on
        the ion type, its loss, and fragment index.

    Parameters
    ----------
    ion_type_loss : str
        A string containing the ion type, its charge and any loss.
    fragment_idx : int
        The index of the fragment.

    Returns
    -------
    ion_code : str
        The ion code as found in Prosit msp output.
    """
    ion_data = ion_type_loss.split('^')
    ion_charge_and_loss = ion_data[1].split('-')

    ion_letter = ion_data[0]
    ion_charge = ion_charge_and_loss[0]
    ion_loss = ion_charge_and_loss[1]
    ion_code = f'{ion_letter}{fragment_idx+1}'
    if ion_charge != '1':
        ion_code += f'^{ion_charge}'
    if ion_loss != '':
        ion_code += f'-{ion_loss}'

    return ion_code
