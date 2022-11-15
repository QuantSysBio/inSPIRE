""" Functions for reading in Mascot search results.
"""
import re

import pandas as pd

from inspire.constants import (
    ACCESSION_KEY,
    CHARGE_KEY,
    DELTA_SCORE_KEY,
    ENGINE_SCORE_KEY,
    LABEL_KEY,
    MASS_DIFF_KEY,
    PEPTIDE_KEY,
    PTM_ID_KEY,
    PTM_IS_VAR_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    RT_KEY,
    SCAN_KEY,
    SEQ_LEN_KEY,
    SOURCE_KEY,
)
from inspire.utils import filter_for_prosit

# Define separators within Mascot output and the names for relevant columns.
MASCOT_HEADER_MARKER = 'Header'
MASCOT_FILENAME_MARKER = 'Peak list data path'
MASCOT_HITS_START_MARKER = 'prot_hit_num'
MASCOT_NO_FIXED_MODS_MARKER = '""'
MASCOT_QUERIES_START_MARKER = 'Queries'
MASCOT_VAR_MODS_MARKER = 'Variable modifications'
MASCOT_FIXED_MODS_MARKER = 'Fixed modifications'
MASCOT_SEARCH_PARAMS_MARKER = 'Search Parameters'
MASCOT_SCAN_FILE_NAME_LINE = 'Peak list data path'

MASCOT_CHARGE_KEY = 'pep_exp_z'
MASCOT_MISS_KEY = 'pep_miss'
MASCOT_DECOY_KEY = 'prot_desc'
MASCOT_ENGINE_SCORE_KEY = 'pep_score'
MASCOT_PEPTIDE_KEY = 'pep_seq'
MASCOT_PEP_QUERY_KEY = 'pep_query'
MASCOT_SCAN_TITLE_KEY = 'pep_scan_title'
MASCOT_PTM_SEQ_KEY = 'pep_var_mod_pos'
MASCOT_ACCESSION_KEY = 'prot_acc'
MASCOT_MASS_KEY = 'pep_exp_mr'
MASCOT_PRED_MASS_KEY = 'pep_calc_mr'
REQUIRED_MASCOT_COLUMNS = [
    MASCOT_PEP_QUERY_KEY,
    MASCOT_ACCESSION_KEY,
    MASCOT_CHARGE_KEY,
    MASCOT_DECOY_KEY,
    MASCOT_ENGINE_SCORE_KEY,
    MASCOT_MASS_KEY,
    MASCOT_MISS_KEY,
    MASCOT_PEPTIDE_KEY,
    MASCOT_PRED_MASS_KEY,
    MASCOT_PTM_SEQ_KEY,
    MASCOT_SCAN_TITLE_KEY,
]
MASCOT_QUERIES_TITLE_KEY = 'StringTitle'
MASCOT_QUERIES_RT_KEY = 'Retention time range'
MASCOT_QUERIES_INTENSITY_KEY = 'intensity'

def _get_mascot_file_metadata(csv_file):
    """ Function to extract metadata from a csv file of Mascot results.

    Parameters
    ----------
    csv_file : str
        The input address for a mascot search results file.

    Returns
    -------
    header_line : int
        The line number of the Header line.
    output_filename : str
        The name to be used for the output file of processing to ensure
        it matches with the processed Peaks file.
    hits_line : int
        The line number where the data on peptide matches starts.
    queries_line : int
        The line number where the query data starts (including Retention
        Time data).
    """
    with open(csv_file, 'r', encoding='UTF-8') as open_file:
        line_idx = 0
        line = open_file.readline()
        header_line = -1
        hits_line = -1
        queries_line = None
        fixed_mods_line = None
        var_mods_line = -1
        search_params_line = -1
        scan_filename = ''

        while line:
            if header_line==-1 and MASCOT_HEADER_MARKER in line:
                header_line = line_idx
            if (
                fixed_mods_line is None and
                MASCOT_FIXED_MODS_MARKER in line and
                MASCOT_NO_FIXED_MODS_MARKER not in line
            ):
                fixed_mods_line = line_idx
            if var_mods_line == -1 and MASCOT_VAR_MODS_MARKER in line:
                var_mods_line = line_idx
            if MASCOT_SCAN_FILE_NAME_LINE in line:
                scan_filename = line.split(',')[-1].strip('\n').strip('"')
            if search_params_line == -1 and MASCOT_SEARCH_PARAMS_MARKER in line:
                search_params_line = line_idx
            if hits_line==-1 and MASCOT_HITS_START_MARKER in line:
                hits_line = line_idx
            if queries_line is None and MASCOT_QUERIES_START_MARKER in line:
                queries_line = line_idx
                break
            line = open_file.readline()
            line_idx += 1

    mod_ranges = {}
    mod_ranges['variable'] = range(var_mods_line+2, search_params_line-1)
    if fixed_mods_line is not None:
        mod_ranges['fixed'] = range(fixed_mods_line+2, var_mods_line-1)

    return header_line, hits_line, queries_line, mod_ranges, scan_filename

def _skip_logic(idx, start_idx=0, end_idx=None):
    """ Simple function for identifying which lines to skip when reading a csv.

    Parameters
    ----------
    idx : int
        A line number in a file.
    start_idx : int
        The line number at which csv data starts.
    end_idx: int
        The line number at which csv data ends.

    Returns
    ------
    skip_line : bool
        A flag on whether to skip the line or not.
    """
    if end_idx is not None:
        if start_idx <= idx < end_idx:
            return False
    else:
        if idx >= start_idx:
            return False
    return True

def separate_scan_and_source(df_row, scan_title_format, source_list=None, source_filename=None):
    """ Function to separate source file and scan number (as well as retention time if
        Distiller format used) from mascots scan_title_format column.

    Parameters
    ----------
    df_row : pd.Series
        A row of the mascot search results DataFrame.
    scan_title_format : str or None
        The format of the scan title (see README for options).
    source_list : str or None
        A list of the source files if Mascot Distiller was used.
    source_filename : str or None
        The name of the source file used in the search if available.

    Returns
    -------
    df_row : pd.Series
        The input row updated with new columns.
    """
    scan_title = df_row[MASCOT_SCAN_TITLE_KEY]
    if scan_title_format is None:
        df_row[SCAN_KEY] = int(scan_title.split('=')[-1].strip('~'))
        if source_filename is None:
            source = scan_title.split('File:')[-1]
            if source.startswith('~'):
                source = source.split('~')[1]
            else:
                source = source.split(', ')[0]
        else:
            source = source_filename

        if source.endswith('.raw') or source.endswith('.mgf'):
            source = source[:-4]
        df_row[SOURCE_KEY] = source

    elif scan_title_format == 'mascotDistiller':
        scan_rt_details = scan_title.split(' Scan ')[-1].split(' (rt')
        df_row[SCAN_KEY] = int(scan_rt_details[0])
        source_idx = int(scan_title.split(' from file [')[-1].strip(']'))
        source_idx = int(scan_title.split(' from file [')[-1].strip(']'))
        df_row[SOURCE_KEY] = source_list[source_idx]
        df_row[RT_KEY] = float(scan_rt_details[-1][1:].split(')')[0])
    elif scan_title_format == 'distiller':
        split_name = scan_title.split(' (rt=')
        df_row[SCAN_KEY] = int(split_name[0].split(' Scan ')[-1])
        df_row[RT_KEY] = float(split_name[-1].split(')')[0])
    else:
        raise ValueError('Oops')
    return df_row

def add_rt_data(hits_df, csv_filename, queries_line, scan_file):
    """ Function to add retention time to the main mascot results DataFrame

    Parameters
    ----------
    hits_df : pd.DataFrame
        The main search results DataFrame from Mascot.
    csv_filename : str
        The path to the file containing Mascot search results.
    queries_line : int
        The line number where mascot queries begin.

    Returns
    -------
    hits_df : pd.DataFrame
        The input DataFrame with a retentionTime column added.
    """
    if queries_line is None:
        hits_df[RT_KEY] = 0
        return hits_df
    queries_df = pd.read_csv(
        csv_filename,
        skiprows=lambda idx : _skip_logic(idx, queries_line+2),
        usecols=['query_number', MASCOT_QUERIES_RT_KEY, MASCOT_QUERIES_INTENSITY_KEY]
    )

    queries_df = queries_df.rename(columns={
        'query_number': MASCOT_PEP_QUERY_KEY,
        MASCOT_QUERIES_RT_KEY: RT_KEY,
        MASCOT_QUERIES_INTENSITY_KEY: 'ms1Intensity',
    })
    queries_df[MASCOT_PEP_QUERY_KEY] = queries_df[MASCOT_PEP_QUERY_KEY].apply(
        lambda x : scan_file + str(x)
    )

    hits_df = pd.merge(
        hits_df,
        queries_df,
        how='inner',
        on=MASCOT_PEP_QUERY_KEY,
    )

    return hits_df

def _read_mascot_dfs(
        csv_filename,
        hits_line,
        queries_line,
        mods_range,
        scan_title_format,
        scan_file,
    ):
    hits_df = pd.read_csv(
        csv_filename,
        skiprows=lambda idx : _skip_logic(idx, hits_line, queries_line),
        usecols=REQUIRED_MASCOT_COLUMNS
    )

    mods_df = pd.read_csv(
        csv_filename,
        skiprows=lambda idx : idx not in mods_range['variable'],
        usecols=range(4),
    )
    mods_df[PTM_IS_VAR_KEY] = True

    # Rename to match inSPIRE naming scheme.
    hits_df = hits_df.rename(
        columns={
            MASCOT_ACCESSION_KEY: ACCESSION_KEY,
            MASCOT_CHARGE_KEY: CHARGE_KEY,
            MASCOT_ENGINE_SCORE_KEY: ENGINE_SCORE_KEY,
            MASCOT_PTM_SEQ_KEY: PTM_SEQ_KEY,
            MASCOT_PEPTIDE_KEY: PEPTIDE_KEY,
            MASCOT_MISS_KEY: 'missedCleavages',
        }
    )

    # Filter for Prosit and add feature columns not present.
    hits_df = filter_for_prosit(hits_df)
    hits_df[MASS_DIFF_KEY] = hits_df[MASCOT_MASS_KEY] - hits_df[MASCOT_PRED_MASS_KEY]
    hits_df[SEQ_LEN_KEY] = hits_df[PEPTIDE_KEY].apply(len)
    hits_df['avgResidueMass'] = hits_df[MASCOT_MASS_KEY]/hits_df[SEQ_LEN_KEY]
    hits_df.drop([MASCOT_MASS_KEY, MASCOT_PRED_MASS_KEY], axis=1, inplace=True)

    hits_df[MASCOT_PEP_QUERY_KEY] = hits_df[MASCOT_PEP_QUERY_KEY].apply(
        lambda x : scan_file + str(x)
    )

    if scan_title_format not in ('mascotDistiller', 'distiller'):
        # Mascot keeps retention time data in a separate table,
        # except if distiller is used in which case it is within the
        # pep_scan_title_column.
        hits_df = add_rt_data(
            hits_df,
            csv_filename,
            queries_line,
            scan_file,
        )


    hits_df[LABEL_KEY] = hits_df[MASCOT_DECOY_KEY].apply(
        lambda x : -1 if 'Reversed' in x or 'random' in x or 'Random' in x else 1
    )
    hits_df.drop([MASCOT_DECOY_KEY], axis=1, inplace=True)

    # Add fixed mods
    if 'fixed' in mods_range:
        fixed_mods_df = pd.read_csv(
            csv_filename,
            skiprows=lambda idx : idx not in mods_range['fixed'],
            usecols=range(4)
        )
        fixed_mods_df[PTM_ID_KEY] = fixed_mods_df[PTM_ID_KEY] + mods_df.shape[0]
        fixed_ptm_dict = {}
        for mod_name, mod_id in zip(
                fixed_mods_df[PTM_NAME_KEY].tolist(),
                fixed_mods_df[PTM_ID_KEY].tolist()
            ):
            modified_residues = mod_name.split('(')[-1].split(')')[0]
            fixed_ptm_dict[modified_residues] = mod_id

        hits_df[PTM_SEQ_KEY] = hits_df[[PEPTIDE_KEY, PTM_SEQ_KEY]].apply(
            lambda x : _add_fixed_mod(x[PEPTIDE_KEY], x[PTM_SEQ_KEY], fixed_ptm_dict),
            axis=1
        )

        fixed_mods_df[PTM_IS_VAR_KEY] = False

        mods_df = pd.concat([mods_df, fixed_mods_df])


    return hits_df, mods_df

def _add_fixed_mod(peptide, ptm_seq, fixed_ptm_dict):
    """ Function to add fixed modifications to Mascot search results in the same style as
        variable modifications.

    Parameters
    ----------
    peptide : str
        The peptide sequence.
    ptm_seq : str
        The sequence of variable modifications.
    fixed_ptm_dict : dict
        A dictionary mapping fixed ptm residues to their ID.

    Returns
    -------
    ptm_seq : str
        The input ptm_seq updated with any relevant fixed modifications.
    """
    for residues in fixed_ptm_dict:
        if residues == 'N-term':
            if not isinstance(ptm_seq, str):
                ptm_seq = '0.' + ''.join(['0']*len(peptide)) + '.0'
            ptm_seq = fixed_ptm_dict[residues] + ptm_seq[1:]
        elif residues == 'C-term':
            if not isinstance(ptm_seq, str):
                ptm_seq = '0.' + ''.join(['0']*len(peptide)) + '.0'
            ptm_seq = ptm_seq[:-1] + fixed_ptm_dict[residues]
        else:
            for res in residues:
                ptm_postns = [m.start() for m in re.finditer(res, peptide)]
                if ptm_postns and not isinstance(ptm_seq, str):
                    ptm_seq = '0.' + ''.join(['0']*len(peptide)) + '.0'
                for pos in ptm_postns:
                    ptm_seq = ptm_seq[:pos+2] + str(fixed_ptm_dict[residues]) + ptm_seq[pos+3:]

    return ptm_seq

def read_single_mascot_data(input_filename, scan_title_format, variable_mods):
    """ Function to read in mascot search results from a single file.

    Parameters
    ----------
    input_filename : str
        A location of mascot search results.
    scan_title_format : str
        The format of mascot's pep_scan_title column.
    source_list : list of str or None
        A list of the source files used in the mascot search.
    variable_mods : pd.DataFrame
        The previously discovered variable modifications.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    # Original File contains both peptide hits and queries, search file to find separators.
    header_line, hits_line, queries_line, mod_ranges, scan_file = _get_mascot_file_metadata(
        input_filename
    )

    if header_line == -1:
        raise ValueError('No header line found in Mascot search results.')

    # Read separate files as pandas dataframes.
    hits_df, new_variable_mods = _read_mascot_dfs(
        input_filename,
        hits_line,
        queries_line,
        mod_ranges,
        scan_title_format,
        scan_file,
    )

    if scan_title_format == 'distiller':
        hits_df[SOURCE_KEY] = scan_file.strip('.raw').strip('.mgf').strip('.temp')

    if variable_mods is None or new_variable_mods.equals(variable_mods):
        return hits_df, new_variable_mods

    new_variable_mods = new_variable_mods.rename(
        columns={
            PTM_ID_KEY: 'newId'
        }
    )[[PTM_NAME_KEY, 'newId']]
    combo_df = pd.merge(
        variable_mods,
        new_variable_mods,
        on=[PTM_NAME_KEY],
        how='outer',
    )
    unmatched_df = combo_df[combo_df['newId'] != combo_df[PTM_ID_KEY]]
    replacements = {str(k): str(v) for k, v in zip(
        unmatched_df['newId'].tolist(), unmatched_df[PTM_ID_KEY].tolist()
    )}
    hits_df[PTM_SEQ_KEY] = hits_df[PTM_SEQ_KEY].apply(
        lambda x : _replace_ptm_id(x, replacements)
    )

    return hits_df, variable_mods

def _replace_ptm_id(ptm_seq, replacements):
    """ Function to replace certain ptm ids in the ptm seq.

    Parameters
    ----------
    ptm_seq : str or None
        The PTM sequence for the PSM.
    replacements : dict
        The PTM IDs to be updated.

    Returns
    -------
    new_ptm_seq : str or None
        The updated PTM sequence.
    """
    if not isinstance(ptm_seq, str):
        return ptm_seq
    new_ptm_seq = ''
    for char in ptm_seq:
        if char in replacements:
            new_ptm_seq += replacements[char]
        else:
            new_ptm_seq += char
    return new_ptm_seq

def read_mascot_data(mascot_data, scan_title_format, source_list, reduce, source_filename):
    """ Function to read in mascot search results from one or more files.

    Parameters
    ----------
    mascot_data : str or list of str
        A single location of mascot search results or a list of locations.
    scan_title_format : str
        The format of mascot's pep_scan_title column.
    source_list : list of str or None
        A list of the source files used in the mascot search.
    source_filename : str or None
        The name of the source file used in the search if available.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_dfs : pd.DataFrame
        A small DataFrame detailing the ptms found in the data.
    """
    if isinstance(mascot_data, list):
        hits_dfs = []
        variable_mods_dfs = []
        variable_mods = None
        for input_filename in mascot_data:
            hits_df, variable_mods = read_single_mascot_data(
                input_filename,
                scan_title_format,
                variable_mods,
            )
            hits_dfs.append(hits_df)
            variable_mods_dfs.append(variable_mods)
        hits_df = pd.concat(hits_dfs)
        for i in range(len(variable_mods_dfs)-1):
            assert variable_mods_dfs[i].equals(variable_mods_dfs[i+1])
    else:
        hits_df, variable_mods = read_single_mascot_data(
            mascot_data,
            scan_title_format,
            None,
        )

    hits_df = mascot_reduce_to_max(hits_df, reduce)

    hits_df = hits_df.apply(
        lambda x : separate_scan_and_source(x, scan_title_format, source_list, source_filename),
        axis=1
    )
    hits_df = hits_df.drop_duplicates(
        subset=[SOURCE_KEY, SCAN_KEY, PEPTIDE_KEY]
    )

    if reduce:
        hits_df['scanCounts'] = hits_df.groupby(
            [SOURCE_KEY, SCAN_KEY]
        )[ENGINE_SCORE_KEY].transform('count')
        hits_df['fromChimera'] = hits_df['scanCounts'].apply(lambda x : 1 if x > 1 else 0)
        hits_df = hits_df.drop('scanCounts', axis=1)
    else:
        hits_df['fromChimera'] = 0


    hits_df = hits_df.drop(MASCOT_SCAN_TITLE_KEY, axis=1)

    return hits_df, variable_mods

def mascot_reduce_to_max(main_df, reduce):
    """ Function to filter down to the best scoring PSM per spectrum.

    Parameters
    ----------
    main_df : pd.DataFrame
        A DataFrame of search results.

    Returns
    -------
    main_df : pd.DataFrame
        The search DataFrame with results filtered per PSM.
    """
    main_df = main_df.sort_values(by=[LABEL_KEY, ENGINE_SCORE_KEY], ascending=False)
    if reduce:
        main_df['ilSub'] = main_df[PEPTIDE_KEY].apply(
            lambda x : x.replace('I', 'L')
        )
        main_df = main_df.drop_duplicates(subset=[MASCOT_PEP_QUERY_KEY, 'ilSub'], keep='first')
        main_df = main_df.drop('ilSub', axis=1)

    main_df['max_pep_score'] = main_df.groupby(
        [MASCOT_PEP_QUERY_KEY]
    )[ENGINE_SCORE_KEY].transform(max)

    main_df['pep_score_2'] = main_df.groupby(
        [MASCOT_PEP_QUERY_KEY]
    )[ENGINE_SCORE_KEY].transform(
        lambda x : ([0.0] + sorted(list(x)))[-2]
    )

    if reduce:
        main_df = main_df[main_df[ENGINE_SCORE_KEY] == main_df['max_pep_score']]
        main_df = main_df.drop_duplicates(subset=[MASCOT_PEP_QUERY_KEY])

    main_df[DELTA_SCORE_KEY] = main_df[
        [ENGINE_SCORE_KEY, 'max_pep_score', 'pep_score_2']
    ].apply(
        lambda x : x[ENGINE_SCORE_KEY] - x['pep_score_2'],
        axis=1
    )

    main_df = main_df.drop(['max_pep_score', 'pep_score_2'], axis=1)

    return main_df
