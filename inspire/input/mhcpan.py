""" Functions for reading NetMHCpan output.
"""
import os

import pandas as pd
import polars as pl

MHC_PAN_COL_NAMES = [
    'Pos',
    'MHC',
    'Peptide',
    'Core',
    'Of',
    'Gp',
    'Gl',
    'Ip',
    'Il',
    'Icore',
    'Identity',
    'Score_EL',
    '%Rank_EL',
    'Score_BA',
    '%Rank_BA',
    'Aff(nM)',
    'BindLevel',
]


def _get_mhc_pan_metadata(mhc_pan_file):
    """ Function to find where to read from the mhc pan output.

    Parameters
    ----------
    mhc_pan_file : str
        Path to an output of NetMHCpan predicted binding affinity.

    Returns
    -------
    start_line : int
        The line on which tabular data starts.
    end_line : int
        The line on which tabular data ends.
    """
    with open(mhc_pan_file, 'r', encoding='UTF-8') as open_file:
        line_idx = 0
        dash_lines = []

        while len(dash_lines) < 3 and (line := open_file.readline()):
            if '--------' in line:
                dash_lines.append(line_idx)
            line_idx += 1

        if len(dash_lines) < 3:
            return -1, -1
        start_line = dash_lines[1] + 1
        end_line = dash_lines[2] - 1
    return start_line, end_line

def _clean_mhc_pan_output(mhc_pan_file):
    """ Function to deal with inconsistent white spacing in the mhcpan output.

    Parameters
    ----------
    mhc_pan_file : str
        Path to an output of NetMHCpan predicted binding affinity.
    """
    with open(mhc_pan_file, 'r', encoding='UTF-8') as file :
        filedata = file.read()

    filedata = filedata.replace('<= SB', '<=SB')
    filedata = filedata.replace('<= WB', '<=WB')

    with open(mhc_pan_file, 'w', encoding='UTF-8') as file:
        file.write(filedata)

def read_single_mhcpan_file(mhc_pan_file):
    """ Function to read in from a single NetMHCpan output file.

    Parameters
    ----------
    mhc_pan_file : str
        Path to an output of NetMHCpan predicted binding affinities.

    Returns
    -------
    mhc_pan_df : pd.DataFrame
        Tabular data from the file read into a DataFrame.
    """
    _clean_mhc_pan_output(mhc_pan_file)
    start, end = _get_mhc_pan_metadata(mhc_pan_file)
    try:
        mhc_pan_df = pd.read_csv(
            mhc_pan_file,
            delim_whitespace=True,
            skiprows=lambda idx, sl=start, el=end : idx < sl or idx > el,
            header=None,
            names=MHC_PAN_COL_NAMES
        )
    except pd.errors.ParserError:
        mhc_pan_df = pd.read_csv(
            mhc_pan_file,
            delim_whitespace=True,
            skiprows=lambda idx, sl=start, el=end : idx < sl or idx > el,
            header=None,
            names=[col for col in MHC_PAN_COL_NAMES if col != 'BindLevel']
        )

    mhc_pan_df['BindLevel'] = mhc_pan_df['%Rank_BA'].apply(
        lambda x : '<=SB' if x < 0.5 else ('<=WB' if x < 2.0 else '')
    )
    return mhc_pan_df

def read_mhcpan_output(mhc_pan_dir, per_allele=False):
    """ Function read raw mhcpan output files.

    Parameters
    ----------
    mhc_pan_dir : str
        A directory containing NetMHCpan output files.
    per_allele : bool (default=False)
        A flag indicating whether the data should be separated per allele.

    Returns
    -------
    mhc_pan_combined_df : pl.DataFrame
        DataFrame of NetMHCpan predicted binding affinities.
    """
    mhc_pan_files = [
        f'{mhc_pan_dir}/{x}' for x in os.listdir(mhc_pan_dir) if x.startswith('output')
    ]
    mhc_dfs = {
        x.split('_')[-3] + x.split('_')[-1]: [] for x in mhc_pan_files
    }
    for mhc_pan_file in mhc_pan_files:
        input_code = mhc_pan_file.split('_')[-3] + mhc_pan_file.split('_')[-1]
        mhc_pan_df = read_single_mhcpan_file(mhc_pan_file)

        if per_allele:
            allele = mhc_pan_file.split('_')[-2].split('.txt')[0]
            mhc_pan_df = mhc_pan_df.rename(
                columns={
                    'Aff(nM)': f'{allele}_BindingAffinity',
                    '%Rank_BA': f'{allele}_%Rank_BA',
                    'BindLevel': f'{allele}_BindLevel',
                }
            )
            mhc_pan_df = mhc_pan_df[[
                'Peptide',
                f'{allele}_BindingAffinity',
                f'{allele}_%Rank_BA',
                f'{allele}_BindLevel',
            ]]
        mhc_dfs[input_code].append(mhc_pan_df)

    if per_allele:
        all_dfs = []
        for input_code in mhc_dfs:
            per_input_dfs = mhc_dfs[input_code]
            mhc_pan_combined_df = per_input_dfs[0].drop_duplicates(subset=['Peptide'])
            for mhc_df in per_input_dfs[1:]:
                mhc_pan_combined_df = pd.merge(
                    mhc_pan_combined_df,
                    mhc_df.drop_duplicates(subset=['Peptide']),
                    how='inner',
                    on='Peptide',
                )
            all_dfs.append(mhc_pan_combined_df)
    else:
        all_dfs = []
        for length in mhc_dfs:
            all_dfs.extend(mhc_dfs[length])

    mhc_pan_combined_df = pd.concat(all_dfs)
    if not per_allele:
        # If different HLAs have been predicted find the minimum prediction.
        mhc_pan_combined_df['minAffinity'] = mhc_pan_combined_df.groupby(
            ['Peptide']
        )['Aff(nM)'].transform(min)
        mhc_pan_combined_df = mhc_pan_combined_df[
            mhc_pan_combined_df['Aff(nM)'] == mhc_pan_combined_df['minAffinity']
        ]
        mhc_pan_combined_df = mhc_pan_combined_df.drop_duplicates('Peptide')

    return pl.from_pandas(mhc_pan_combined_df)
