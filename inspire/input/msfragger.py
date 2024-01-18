""" Functions for reading in MSFragger search results.
"""
import multiprocessing
import os
from pathlib import Path

import polars as pl
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
from inspire.utils import filter_for_prosit

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

ID_NUMBERS = {
    'unknown': 6,
    'Deamidation (N)': 5,
    'Deamidation (Q)': 5,
    'Phospho (S)': 4,
    'Phospho (T)': 4,
    'Phospho (Y)': 4,
    'Acetyl (N-term)': 3,
    'Oxidation (M)': 2,
    'Carbamidomethyl (C)': 1,
}

MSF_MAPPING_PEP_XML = {
    '115': 'Deamidation (N)',
    '129': 'Deamidation (Q)',
    '167': 'Phospho (S)',
    '181': 'Phospho (T)',
    '243': 'Phospho (Y)',
    '43': 'Acetyl (N-term)',
    '147': 'Oxidation (M)',
    '160': 'Carbamidomethyl (C)',
}

MSF_MAPPING_PEP_XML_REV = {
    item: key for key, item in MSF_MAPPING_PEP_XML.items()
}

def _extract_psm_id_data(spec_id):
    """ Function to extract data from the PSM ID column to standard inSPIRE format.

    Parameters
    ----------
    spec_id : str
        The spectrum ID from MSFragger pepXML output.

    Returns
    -------
    results : dict
        Dictionary of the scan, source and reported charge of the peptide.
    """
    results = {}
    spec_id_data = spec_id.split('.')
    results[SOURCE_KEY] = spec_id_data[0]
    results[SCAN_KEY] = int(spec_id_data[1])
    results[f'alt_{CHARGE_KEY}'] = int(spec_id_data[-1].split('_')[0])
    return results

def _extract_mod_weight(code):
    """ Function to extract the weight of a modification.

    Parameters
    ----------
    code : dict
        Dictionary describing a modification.

    Returns
    -------
    mass : int or None
    """
    if code['mass'] is not None:
        return round(code['mass'])
    return None


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
            _extract_mod_weight,
            skip_nulls=False,
        ).drop_nulls().unique().to_list()
    )
    msf_mods = [str(int(x)) for x in msf_mods if int(x) != -1]

    mod_names = [MSF_MAPPING_PEP_XML.get(msf_mod, 'unknown') for msf_mod in msf_mods]
    mod_weights = [KNOWN_PTM_WEIGHTS.get(mod, 100) for mod in mod_names]

    mod_df = pd.DataFrame({
        PTM_NAME_KEY: mod_names,
        PTM_WEIGHT_KEY: mod_weights,
        PTM_IS_VAR_KEY: [
            fixed_modifications is None or not nm in fixed_modifications for nm in mod_names
        ],
        'msfMod': msf_mods,
    })

    mod_df = mod_df.sort_values(by=PTM_NAME_KEY)
    mod_df = mod_df.reset_index(drop=True)
    mod_df[PTM_ID_KEY] = mod_df[PTM_NAME_KEY].apply(ID_NUMBERS.get)
    msf_name_to_id = dict(zip(
        mod_df['msfMod'].tolist(),
        [str(x) for x in mod_df[PTM_ID_KEY].tolist()],
    ))
    return mod_df, msf_name_to_id

def _separate_msf_ptms(mod_pep, ms_frag_mappings):
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
    peptide = mod_pep
    start_mod = '0'
    if peptide[0:5] == 'n[43]':
        start_mod = '3'
        peptide = peptide[5:]
    if '[' in peptide:
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

        return f'{start_mod}.{ptm_seq}.0'
    if start_mod == '3':
        return f'{start_mod}.{"0"*len(peptide)}.0'
    return None

def flatten_protein_data(proteins):
    """ Function to extract protein accession data from pepXML dictionary.

    Parameters
    ----------
    proteins : pd.Series
        The pepXML protein data for a single hit.

    Returns
    -------
    df_row : pd.Series
        The input data updated to parse protein accession data.
    """
    label = -1
    results = {}
    new_proteins = []
    for protein in proteins:
        if not protein['protein'].startswith('rev'):
            label = 1
        new_proteins.append(protein['protein'])
    results[LABEL_KEY] = label
    results[ACCESSION_KEY] = ','.join(new_proteins)
    return results

def get_read_path(df_loc, file_idx):
    """ Function to remove invalid characters from a source.

    Parameters
    ----------
    df_loc : str
        Location of the file.
    file_idx : int
        The index of the input file.

    Returns
    -------
    read_loc : str
        The final location that should be read from.
    remove_read_loc : bool
        Flag indicating if a temporary file was generated which should be removed.
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
        filedata = filedata.replace(source_name, f'temp_{file_idx}')
        with open(df_loc.replace(source_name, f'temp_{file_idx}'), 'w', encoding='UTF-8') as file:
            file.write(filedata)
        return df_loc.replace(source_name, f'temp_{file_idx}'), True
    return df_loc, False

def read_single_ms_fragger_data(df_loc, fixed_modifications, file_idx):
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
    read_loc, remove_read_loc  = get_read_path(df_loc, file_idx)

    all_psms = []
    with pepxml.read(read_loc) as psms:
        for psm in psms:
            psms_to_add = []
            peps_added = []

            for pep_psm in psm['search_hit']:
                if pep_psm['peptide'] not in peps_added:
                    pep_psm[CHARGE_KEY] = int(psm['assumed_charge'])
                    pep_psm['SpecId'] = psm['spectrum']
                    pep_psm[MSF_RT_KEY] = psm['retention_time_sec']
                    pep_psm['hyperscore'] = pep_psm['search_score']['hyperscore']
                    del pep_psm['search_score']
                    peps_added.append(pep_psm['peptide'])
                    psms_to_add.append(pep_psm)

            if len(psms_to_add) > 1:
                rank_2_score = psms_to_add[1]['hyperscore']
            else:
                rank_2_score = 0

            for pep_psm in psms_to_add:
                pep_psm[MSF_DELTA_KEY] = pep_psm['hyperscore'] - rank_2_score
                all_psms.append(pep_psm)

    msf_df = pl.DataFrame(all_psms)

    msf_df = msf_df.with_columns(
        pl.col('SpecId').apply(
            _extract_psm_id_data
        ).alias('results')
    ).unnest('results')

    if CHARGE_KEY not in msf_df.columns:
        msf_df = msf_df.rename({f'alt_{CHARGE_KEY}': CHARGE_KEY})

    # Separate PTMs.
    var_mod_df, msf_name_to_id = _get_msf_mods_pepxml(msf_df, fixed_modifications)

    msf_df = msf_df.with_columns(
        pl.col('modified_peptide').apply(
            lambda mod_pep : _separate_msf_ptms(mod_pep, msf_name_to_id)
        ).alias(PTM_SEQ_KEY)
    )

    msf_df = msf_df.with_columns(
        pl.col('proteins').apply(flatten_protein_data).alias('results')
    )
    msf_df = msf_df.drop('proteins')
    msf_df = msf_df.unnest('results')

    msf_df= msf_df.with_columns(
        pl.col(PEPTIDE_KEY).apply(len).alias(MSF_PEP_LEN_KEY),
    )
    msf_df= msf_df.with_columns(
        (
            pl.col('calc_neutral_pep_mass')/pl.col(MSF_PEP_LEN_KEY)
        ).alias('avgResidueMass'),
    )

    # Rename to match inSPIRE naming scheme.
    msf_df = msf_df.rename({
        MSF_PEP_LEN_KEY: SEQ_LEN_KEY,
        'massdiff': MASS_DIFF_KEY,
        MSF_RT_KEY: RT_KEY,
        MSF_SCORE_KEY: ENGINE_SCORE_KEY,
        MSF_DELTA_KEY: DELTA_SCORE_KEY,
    })

    msf_df = msf_df.with_columns(
        pl.lit(0).alias('fromChimera'),
        pl.lit(0).alias('missedCleavages'),
        pl.col(MASS_DIFF_KEY).apply(abs).alias(MASS_DIFF_KEY),
    )

    # Filter for Prosit and add feature columns not present.
    msf_df = filter_for_prosit(msf_df)

    msf_df = msf_df.select(
        'source',
        'scan',
        'peptide',
        'Label',
        'proteins',
        'ptm_seq',
        'sequenceLength',
        'missedCleavages',
        'charge',
        'massDiff',
        'retentionTime',
        'engineScore',
        'deltaScore',
        'fromChimera',
        'avgResidueMass',
    )

    if remove_read_loc:
        os.remove(read_loc)

    return msf_df, var_mod_df

def read_ms_fragger_data(ms_fragger_data, fixed_modifications, n_cores, reduce_results):
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
            (msf_file, fixed_modifications, file_idx) for file_idx, msf_file in enumerate(
                ms_fragger_data
            )
        ]

        with multiprocessing.get_context('spawn').Pool(processes=n_cores) as pool:
            results = pool.starmap(read_single_ms_fragger_data, func_args)

        mods_dfs = [res[1] for res in results]

        # Combine DataFrames and validate that same PTMs are present.
        hits_df = pl.concat([res[0] for res in results])

        if reduce_results:
            hits_df = hits_df.sort(by=[ENGINE_SCORE_KEY, LABEL_KEY], descending=True)
            hits_df = hits_df.unique(subset=[SOURCE_KEY, SCAN_KEY])

        return hits_df, mods_dfs[0]

    hits_df, mods_df = read_single_ms_fragger_data(ms_fragger_data, fixed_modifications, 0)
    if reduce_results:
        hits_df = hits_df.sort(by=[ENGINE_SCORE_KEY, LABEL_KEY], descending=True)
        hits_df = hits_df.unique(subset=[SOURCE_KEY, SCAN_KEY])

    return hits_df, mods_df
