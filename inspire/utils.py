""" Helpful functions used across the module.
"""
import pickle
import re

import pandas as pd

from inspire.constants import (
    CHARGE_KEY,
    KNOWN_PTM_LOC,
    KNOWN_PTM_WEIGHTS,
    PEPTIDE_KEY,
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    SCAN_KEY,
    SOURCE_KEY,
    SPECTRAL_ANGLE_KEY,
)
from inspire.input.mgf import process_mgf_file
from inspire.input.mzml import process_mzml_file

def fetch_collision_energy(output_folder):
    """ Function to fetch the calibrated collision energy setting.
    """
    results_df = pd.read_csv(
        f'{output_folder}/collisionEnergyStats.csv'
    )
    optimal_collision_energy = results_df['collisionEnergy'].iloc[
        results_df[SPECTRAL_ANGLE_KEY].idxmax()
    ]

    return optimal_collision_energy

def permute_ptms(peptide, ptm_seq, uniform_length=False):
    """ Function to generate all possible permutations on the PTMs of a peptide
        sequence due to adjacent amino acid swaps.

    Parameters
    ----------
    peptide : str
        The original peptide.
    ptm_seq : str or nan
        The PTM sequence of the peptide.
    uniform_length : bool (default=False)
        Flag indicating whether to return a list of length 29 (max possible permutations)
        or to only return the permutations possible.

    Returns
    -------
    permed_ptms : list of str or None
        A list of the possible PTM permutations.
    """
    if not isinstance(ptm_seq, str):
        return [ptm_seq]*29

    permed_ptms = []
    for idx in range(len(ptm_seq)-5):
        if peptide[idx] != peptide[idx+1]:
            permed_ptms.append(
                ptm_seq[:idx+2] + ptm_seq[idx+3] + ptm_seq[idx+2] + ptm_seq[idx+4:]
            )
        elif uniform_length:
            permed_ptms.append(None)
    if uniform_length:
        permed_ptms += [None]*(29-len(permed_ptms))
    return permed_ptms

def permute_seq(peptide, uniform_length=False):
    """ Function to generate all possible permutations on a peptide
        sequence due to adjacent amino acid swaps.

    Parameters
    ----------
    peptide : str
        The original peptide.
    uniform_length : bool (default=False)
        Flag indicating whether to return a list of length 29 (max possible permutations)
        or to only return the permutations possible.

    Returns
    -------
    permed_peps : list of str or None
        A list of the possible peptide permutations.
    """
    permed_peps = []
    for idx in range(len(peptide)-1):
        if peptide[idx] != peptide[idx+1]:
            permed_peps.append(
                peptide[:idx] + peptide[idx+1] + peptide[idx] + peptide[idx+2:]
            )
        elif uniform_length:
            permed_peps.append(None)
    if uniform_length:
        permed_peps += [None]*(29-len(permed_peps))
    return permed_peps

def check_bad_mods(ptm_str, bad_mods):
    """ Function to check for the presence of ptms other than oxidation of methionine
        or carbamidomehtylation of Cysteine.
    """
    if not isinstance(ptm_str, str):
        return False

    for mod_id in ptm_str:
        if mod_id in bad_mods:
            return True
    return False

def get_cam_flag(mods_df):
    """ Function to get the flag for oxidation of methionine from the PTMs DataFrame.
    Parameters
    ----------
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs found in the data.
    Returns
    -------
    carb_flag : int
        The flag for carbamidomtheylation of cysteine.
    """
    try:
        carb_flag = int(mods_df[
            (mods_df[PTM_NAME_KEY] == 'Carbamidomethylation') |
            (mods_df[PTM_NAME_KEY] == 'Carbamidomethyl (C)')
        ][PTM_ID_KEY].iloc[0])
    except IndexError:
        carb_flag = -2

    return carb_flag

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

def get_mokapot_weights(output_folder, inspire_step):
    """ Function to get the weights of mokapot models.

    Parameters
    ----------
    output_folder : str
        The folder where all inSPIRE output is written.
    inspire_step : str
        Indicator of which step fo inSPIRE the model was trained for.

    Results
    -------
    weights_df : pd.DataFrame
        A DataFrame of the weights for different features.
    """
    for fold_idx in range(1, 4):
        with open(
            f'{output_folder}/{inspire_step}.mokapot.model_fold-{fold_idx}.pkl',
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

def _modify_ptm_seq(pep_seq, ptm_seq, ptm_locs, ptm_id):
    """ Helper function to add a fixed PTM to the ptm_seq column of MS search
        result DataFrame.

    Parameters
    ----------
    pep_seq : str
        The peptide sequence.
    ptm_seq : str
        The ptm sequence.
    ptm_locs : str
        The location(s) where the ptm should be added.
    ptm_id : str
        The id of the ptm.

    Returns
    -------
    ptm_seq : str
        The modified ptm sequence.
    """
    if ptm_locs == 'N-term':
        if not isinstance(ptm_seq, str):
            return ptm_id + '.' + ''.join(['0']*len(pep_seq)) + '.0'
        return ptm_id + ptm_seq[1:]

    for loc in ptm_locs:
        edit_points = [m.start() for m in re.finditer(loc, pep_seq)]
        if edit_points and not isinstance(ptm_seq, str):
            ptm_seq = '0.' + ''.join(['0']*len(pep_seq)) + '.0'
        for e_point in edit_points:
            ptm_seq = ptm_seq[:e_point+2] + ptm_id + ptm_seq[e_point+3:]

    return ptm_seq

def add_fixed_modifications(search_df, mods_df, fixed_modifications):
    """ Function to add fixed modifications to the search data.

    Parameters
    ----------
    search_df : pd.DataFrame
        The DataFrame of MS search results from either MaxQuant or PEAKS.
    mods_df : pd.DataFrame
        The DataFrame of variable modifications in the search results.
    fixed_modifications : list
        A list of the fixed modifications found in the DataFrame.

    Returns
    -------
    mods_df : pd.DataFrame
        A DataFrame of all modifications found (both variable and fixed).
    """
    ptm_ids = []
    ptm_names = []
    ptm_weights = []
    for idx, modification in enumerate(fixed_modifications):
        if modification in mods_df[PTM_NAME_KEY]:
            continue

        ptm_weight = KNOWN_PTM_WEIGHTS.get(modification)
        ptm_idx = mods_df.shape[0]+idx+1
        if ptm_weight is None:
            raise ValueError(f'Unsupported fixed modification {modification}.')

        ptm_ids.append(ptm_idx)
        ptm_names.append(modification)
        ptm_weights.append(ptm_weight)
        str_ptm_idx = str(ptm_idx)
        search_df[PTM_SEQ_KEY] = search_df[[PEPTIDE_KEY, PTM_SEQ_KEY]].apply(
            lambda x, mod=modification, ptm_id=str_ptm_idx : _modify_ptm_seq(
                x[PEPTIDE_KEY],
                x[PTM_SEQ_KEY],
                KNOWN_PTM_LOC[mod],
                ptm_id
            ),
            axis=1
        )

    fixed_ptm_df = pd.DataFrame({
        PTM_ID_KEY: ptm_ids,
        PTM_NAME_KEY: ptm_names,
        PTM_WEIGHT_KEY: ptm_weights,
        PTM_IS_VAR_KEY: [False]*len(fixed_modifications)
    })
    mods_df = pd.concat([
        mods_df[[PTM_ID_KEY, PTM_NAME_KEY, PTM_WEIGHT_KEY, PTM_IS_VAR_KEY]],
        fixed_ptm_df]
    )

    return search_df, mods_df

def convert_mod_seq_to_ptm_seq(mod_seq):
    """ Function to convert the modified prosit sequence to the PTM seq used elsewhere.

    Parameters
    ----------
    mod_seq : str
        The input sequence for Prosit.

    Returns
    -------
    ptm_seq : str
        The string of digits used to represent any PTMs present.
    """
    ptm_seq = '0.'
    while mod_seq:
        if len(mod_seq) == 1 or mod_seq[1] != '[':
            ptm_seq += '0'
            mod_seq = mod_seq[1:]
        elif mod_seq[1] == '[' and mod_seq[0] == 'C':
            ptm_seq += '2'
            mod_seq = mod_seq[8:]
        else:
            ptm_seq += '1'
            mod_seq = mod_seq[8:]

    ptm_seq += '.0'
    return ptm_seq

def fetch_scan_data(input_df, config, with_charge):
    """ Function to fetch the experimental scan data.

    Parameters
    ----------
    input_df : pd.DataFrame
        The DataFrame of PSMs whose spectra we wish to plot.
    config : inspire.config.Config
        The Config object used to run the inSPIRE experiment.

    Returns
    -------
    total_scan_df : pd.DataFrame
        A DataFrame of the necessary scan data.
    """
    source_files = input_df[SOURCE_KEY].unique().tolist()
    scan_dfs = []
    if config.combined_scans_file is not None:
        scan_ids = input_df[SCAN_KEY].tolist()
        mgf_filename = f'{config.scans_folder}/{config.combined_scans_file}'
        scan_dfs = [process_mgf_file(
            mgf_filename,
            set(scan_ids),
            config.scan_title_format,
            config.source_files,
            combined_source_file=True,
            with_charge=with_charge,
        )]
    else:
        for scan_file in source_files:
            scan_ids = input_df[input_df[SOURCE_KEY] == scan_file][SCAN_KEY].tolist()
            if config.scans_format == 'mzML':
                scan_df = process_mzml_file(
                    f'{config.scans_folder}/{scan_file}.{config.scans_format}',
                    set(scan_ids),
                    with_charge=with_charge,
                )
            else:
                mgf_filename = f'{config.scans_folder}/{scan_file}.{config.scans_format}'
                scan_df = process_mgf_file(
                    mgf_filename,
                    set(scan_ids),
                    config.scan_title_format,
                    config.source_files,
                    combined_source_file=False,
                    with_charge=with_charge,
                )
            scan_dfs.append(scan_df)
    total_scan_df = pd.concat(scan_dfs)
    return total_scan_df
