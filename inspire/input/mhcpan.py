""" Functions for reading NetMHCpan output.
"""
import os

import pandas as pd

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


def get_mhc_pan_metadata(mhc_pan_file):
    """ Function to find where to read from the mhc pan output.
    """
    with open(mhc_pan_file, 'r', encoding='UTF-8') as open_file:
        line_idx = 0
        line = open_file.readline()
        dash_lines = []

        while len(dash_lines) < 3 and line:
            if '--------' in line:
                dash_lines.append(line_idx)
            line = open_file.readline()
            line_idx += 1
        if len(dash_lines) < 3:
            return -1, -1
        start_line = dash_lines[1] + 1
        end_line = dash_lines[2] - 1
    return start_line, end_line

def clean_mhc_pan_output(mhc_pan_file):
    """ Function to deal with inconsistent white spacing in the mhcpan output.
    """
    with open(mhc_pan_file, 'r', encoding='UTF-8') as file :
        filedata = file.read()

    filedata = filedata.replace('<= SB', '<=SB')
    filedata = filedata.replace('<= WB', '<=WB')

    with open(mhc_pan_file, 'w', encoding='UTF-8') as file:
        file.write(filedata)

def read_mhcpan_output(mhc_pan_dir):
    """ Function read raw mhcpan output files.
    """
    mhc_pan_files = [
        f'{mhc_pan_dir}/{x}' for x in os.listdir(mhc_pan_dir) if x.startswith('output')
    ]
    mhc_dfs = []
    for mhc_pan_file in mhc_pan_files:
        clean_mhc_pan_output(mhc_pan_file)
        start, end = get_mhc_pan_metadata(mhc_pan_file)
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
        mhc_dfs.append(mhc_pan_df)
    mhc_pan_combined_df = pd.concat(mhc_dfs)

    # If different HLAs have been predicted find the minimum prediction.
    mhc_pan_combined_df['minAffinity'] = mhc_pan_combined_df.groupby(
        ['Peptide']
    )['Aff(nM)'].transform(min)
    mhc_pan_combined_df = mhc_pan_combined_df[
        mhc_pan_combined_df['Aff(nM)'] == mhc_pan_combined_df['minAffinity']
    ]
    mhc_pan_combined_df = mhc_pan_combined_df.drop_duplicates('Peptide')

    return mhc_pan_combined_df
