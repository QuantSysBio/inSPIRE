""" Functions for reading in MSFragger search results.
"""
import os
import multiprocessing
from pathlib import Path

import pandas as pd
from pyteomics import pepxml

from inspire.constants import (
    ACCESSION_KEY,
    CHARGE_KEY,
    DELTA_SCORE_KEY,
    ENGINE_SCORE_KEY,
    KNOWN_PTM_WEIGHTS,
    LABEL_KEY,
    MASS_DIFF_KEY,
    PEPTIDE_KEY,
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    RT_KEY,
    SCAN_KEY,
    SEQ_LEN_KEY,
    SOURCE_KEY,
)

# Define the relevant column names from MSFragger search results.
MSF_ACCESSION_KEY = 'Proteins'
MSF_DELTA_KEY = 'delta_hyperscore'
MSF_PPM_ERR_KEY = 'abs_ppm'
MSF_PEPTIDE_KEY = 'Peptide'
MSF_MASS_KEY = 'ExpMass'
MSF_LABEL_KEY = 'Label'
MSF_PEP_LEN_KEY = 'peptide_length'
MSF_SCORE_KEY = 'hyperscore'
MSF_RT_KEY = 'retentiontime'
MSF_LOG10E_KEY = 'log10_evalue'
MSF_NTT_KEY = 'ntt'
MSF_NMC_KEY = 'nmc'

MSF_IGNORE_COLS = [
    MSF_LOG10E_KEY,
    MSF_NMC_KEY,
    MSF_NTT_KEY,
]

MSF_MAPPING = {
    '15.9949M': 'Oxidation (M)',
    '57.9214C': 'Carbamidomethyl (C)',
}
MSF_MAPPING_PEP_XML = {
    '43': 'Acetyl (N-term)',
    '115': 'Deamidated (N)',
    '129': 'Deamidated (Q)',
    '147': 'Oxidation (M)',
    '160': 'Carbamidomethyl (C)',
}
MOD_NAMES = [
    'Carbamidomethyl (C)',
    'Oxidation (M)',
    'Acetyl (N-term)',
    'Deamidated (N)',
    'Deamidated (Q)',
]

MSF_MAPPING_REV = {
    item: key for key, item in MSF_MAPPING.items()
}
MSF_MAPPING_PEP_XML_REV = {
    item: key for key, item in MSF_MAPPING.items()
}

def _extract_psm_id_data(df_row):
    """ Function to extract data from the PSM ID column to standard inSPIRE format.
    """
    spec_id_data = df_row['SpecId'].split('.')
    df_row[SCAN_KEY] = int(spec_id_data[1])
    if CHARGE_KEY not in df_row:
        df_row[CHARGE_KEY] = int(spec_id_data[-1].split('_')[0])
    return df_row

def _get_msf_mods_pepxml(msf_df, fixed_modifications):
    """ Function to get the modifications based on the MS fragger pepXML columns.

    Parameters
    ----------
    msf_df : pd.DataFrame
        MSFragger results read in from pepXML format.
    fixed_modifications : list of str or None
        A list of the fixed modification used in the exeriment.

    Returns
    -------
    mod_df : pd.DataFrame
        Small DataFrame of the modifications found in the data.
    msf_name_to_id : dict
        A dictionary mapping names of modifications in MSFragger to inSPIRE IDs.
    """
    if 'modifications' not in msf_df.columns:
        return pd.DataFrame(), {}
    msf_mods = sorted(
        msf_df['modifications'].explode().apply(
            lambda x : round(x['mass']) if isinstance(x, dict) else None
        ).dropna().unique().tolist()
    )
    msf_mods = [str(int(x)) for x in msf_mods]

    mod_names = [MSF_MAPPING_PEP_XML[msf_mod] for msf_mod in msf_mods]
    mod_weights = [KNOWN_PTM_WEIGHTS[mod] for mod in mod_names]

    mod_df = pd.DataFrame({
        PTM_ID_KEY: [MOD_NAMES.index(mod_name)+1 for mod_name in mod_names],
        PTM_NAME_KEY: mod_names,
        PTM_WEIGHT_KEY: mod_weights,
        PTM_IS_VAR_KEY: [
            fixed_modifications is None or not nm in fixed_modifications for nm in mod_names
        ],
        'msfMod': msf_mods,
    })
    mod_df = mod_df.reset_index(drop=True)

    msf_name_to_id = dict(zip(
        mod_df['msfMod'].tolist(),
        [str(x) for x in mod_df[PTM_ID_KEY].tolist()],
    ))
    return mod_df, msf_name_to_id

def separate_msf_ptms(df_row, ms_frag_mappings, mod_pep_key):
    """ Function to separate MSFragger reported PTMs.

    Parameters
    ----------
    df_row : pd.Series
        Row of MSFragger results DataFrame
    ms_frag_mappings : dict
        Dictionary mapping PTM weights to their inSPIRE ID.

    Returns
    -------
    df_row : pd.Series
        The input row with peptide sequence and modifications separated.
    """
    if mod_pep_key == MSF_PEPTIDE_KEY:
        peptide = '.'.join(df_row[mod_pep_key].split('.')[1:-1])
    else:
        peptide = df_row[mod_pep_key]

    n_term_mod = '0'
    if peptide.startswith('n'):
        end_mod_str = peptide.index(']') + 1
        n_term_mod = ms_frag_mappings[peptide[2:end_mod_str-1]]
        peptide = peptide[end_mod_str:]


    if '[' in peptide or n_term_mod != '0':
        pep_seq = ''
        ptm_seq = ''
        while peptide:
            assert peptide[0] != '['
            if len(peptide) == 1:
                pep_seq += peptide[0]
                ptm_seq += '0'
                break

            if peptide[1] != '[':
                pep_seq += peptide[0]
                ptm_seq += '0'
                peptide = peptide[1:]
            else:
                pep_seq += peptide[0]
                peptide = peptide[1:]

                end_mod_str = peptide.index(']') + 1
                ptm_seq += ms_frag_mappings[peptide[1:end_mod_str-1]]
                peptide = peptide[end_mod_str:]

        df_row[PEPTIDE_KEY] = pep_seq
        df_row[PTM_SEQ_KEY] = f'{n_term_mod}.{ptm_seq}.0'
    else:
        df_row[PEPTIDE_KEY] = peptide
        df_row[PTM_SEQ_KEY] = None
    return df_row

def flatten_protein_data(df_row):
    """ Function to extract protein accession data from pepXML dictionary.

    Parameters
    ----------
    df_row : pd.Series
        The pepXML data for a single hit.

    Returns
    -------
    df_row : pd.Series
        The input data updated to parse protein accession data.
    """
    label = -1
    proteins = []
    for protein in df_row['proteins']:
        if not protein['protein'].startswith('rev'):
            label =1
        proteins.append(protein['protein'])
    df_row[LABEL_KEY] = label
    df_row[ACCESSION_KEY] = ','.join(proteins)
    return df_row

def read_single_ms_fragger_data(df_loc, fixed_modifications, file_index):
    """ Function to read in PEAKS DB search results from a single file.

    Parameters
    ----------
    df_loc : str
        A location of PEAKS DB search results.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    source_name = Path(df_loc).name
    if source_name.endswith('_uncalibrated.pepXML'): 
        source_name = source_name[:-20]
    elif source_name.endswith('_calibrated.pepXML'):
        source_name = source_name[:-18]
    elif source_name.endswith('.pepXML'):
        source_name = source_name[:-7]

    if "'" in source_name:
        with open(df_loc, 'r', encoding='UTF-8') as file :
            filedata = file.read()

        filedata = filedata.replace(source_name, f'temp_{file_index}')

        with open(df_loc.replace(source_name, f'temp_{file_index}'), 'w', encoding='UTF-8') as file:
            file.write(filedata)

        read_loc = df_loc.replace(source_name, f'temp_{file_index}')
    else:
        read_loc = df_loc
    all_psms = []

    with pepxml.read(read_loc) as psms:
        for psm in psms:
            if len(psm['search_hit']) > 1:
                rank_2_score = psm['search_hit'][1]['search_score']['hyperscore']
            else:
                rank_2_score = 0

            for pep_psm in psm['search_hit']:
                pep_psm[CHARGE_KEY] = int(psm['assumed_charge'])
                pep_psm['SpecId'] = psm['spectrum']
                pep_psm[MSF_RT_KEY] = psm['retention_time_sec']
                pep_psm['hyperscore'] = pep_psm['search_score']['hyperscore']
                pep_psm[MSF_DELTA_KEY] = pep_psm['hyperscore'] - rank_2_score
                del pep_psm['search_score']
                all_psms.append(pep_psm)
    msf_df = pd.DataFrame(all_psms)

    if "'" in source_name:
        os.remove(read_loc)

    if not msf_df.shape[0]:
        return None, None

    msf_df[SOURCE_KEY] = source_name
    msf_df = msf_df.apply(_extract_psm_id_data, axis=1)

    # Separate PTMs.
    var_mod_df, msf_name_to_id = _get_msf_mods_pepxml(msf_df, fixed_modifications)
    msf_df = msf_df.apply(
        lambda df_row : separate_msf_ptms(df_row, msf_name_to_id, 'modified_peptide'), axis=1
    )
    msf_df = msf_df.apply(
        flatten_protein_data, axis=1
    )
    msf_df[MSF_PEP_LEN_KEY] = msf_df[PEPTIDE_KEY].apply(len)
    msf_df['avgResidueMass'] = msf_df['calc_neutral_pep_mass']/msf_df[MSF_PEP_LEN_KEY]
    ms1_err_key = 'massdiff'

    # Rename to match inSPIRE naming scheme.
    msf_df = msf_df.rename(columns={
        MSF_ACCESSION_KEY: ACCESSION_KEY,
        MSF_LABEL_KEY: LABEL_KEY,
        MSF_PEP_LEN_KEY: SEQ_LEN_KEY,
        ms1_err_key: MASS_DIFF_KEY,
        MSF_RT_KEY: RT_KEY,
        MSF_SCORE_KEY: ENGINE_SCORE_KEY,
        MSF_DELTA_KEY: DELTA_SCORE_KEY,
    })
    msf_df[MASS_DIFF_KEY] = msf_df[MASS_DIFF_KEY].apply(abs)

    # Filter for Prosit and add feature columns not present.
    msf_df['fromChimera'] = 0
    msf_df['missedCleavages'] = 0


    return msf_df, var_mod_df

def read_ms_fragger_data(ms_fragger_data, fixed_modifications, n_cores):
    """ Function to read in PEAKS DB search results from one or more files.

    Parameters
    ----------
    peaks_data : str or list of str
        A single location of PEAKS DB search results or a list of locations.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    if isinstance(ms_fragger_data, list):
        func_args = [
            (
                msf_file, fixed_modifications, file_idx
            ) for file_idx, msf_file in enumerate(
                ms_fragger_data
            )
        ]

        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(read_single_ms_fragger_data, func_args)
        mods_dfs = [res[1] for res in results if res is not None]
        if mods_dfs:
            mods_df = pd.concat(mods_dfs).drop_duplicates(subset=['Name'])
            mods_df = mods_df.reset_index(drop=True)
            mods_df[PTM_IS_VAR_KEY] = mods_df[PTM_IS_VAR_KEY].astype(bool)
            mods_df[PTM_ID_KEY] = mods_df[PTM_ID_KEY].astype(int)
        else:
            mods_df = pd.DataFrame({
                PTM_ID_KEY: [],
                PTM_IS_VAR_KEY:[],
                PTM_NAME_KEY: [],
                PTM_WEIGHT_KEY: []
            })
        

        # Combine DataFrames and validate that same PTMs are present.
        hits_df = pd.concat([res[0] for res in results if res is not None])

        return hits_df, mods_df

    return read_single_ms_fragger_data(ms_fragger_data, fixed_modifications)
