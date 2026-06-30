""" Functions for reading in scans results in mgf format.
"""
import re

import numpy as np
import pandas as pd
from pyteomics import mgf

from inspire.constants import (
    CHARGE_KEY,
    INTENSITIES_KEY,
    MZS_KEY,
    SCAN_KEY,
    SOURCE_KEY,
)

def process_mgf_file(mgf_filename, scan_ids, scan_file_format, source_list, with_charge=False):
    """ Function to process an mgf file to find matches with scan IDs.

    Parameters
    ----------
    mgf_filename : str
        The mgf file from which we are reading.
    scan_ids : set of int
        A set of the scan IDs we require.
    scan_file_format : str
        The format of the file used.
    source_list : list of str
        A list of source names.

    Returns
    -------
    scans_df : pd.DataFrame
        A DataFrame of scan results.
    """
    matched_scan_ids = []
    matched_intensities = []
    matched_mzs = []
    sources = []
    filename = mgf_filename.split('/')[-1]
    if with_charge:
        charge_list = []

    with mgf.read(mgf_filename) as reader:
        for spectrum in reader:
            if scan_file_format is None or scan_file_format == 'distiller':
                if 'scans' in spectrum['params']:
                    scan_id = int(spectrum['params']['scans'])
                else:
                    regex_match = re.match(
                        r'(\d+)(.*?)',
                        spectrum['params']['title'].split('scan=')[-1]
                    )
                    scan_id = int(regex_match.group(1))
                source = filename[:-4]
            else:
                scan_id = int(spectrum['params']['title'].split(' Scan ')[-1].split(' (rt')[0])
                source = source_list[
                    int(spectrum['params']['title'].split(' from file [')[-1].strip(']'))
                ]

            if scan_ids is None or scan_id in scan_ids:
                sources.append(source)
                matched_scan_ids.append(scan_id)
                matched_intensities.append(np.array(list(spectrum['intensity array'])))
                matched_mzs.append(np.array(list(spectrum['m/z array'])))
                if with_charge:
                    charge_list.append(int(spectrum['params']['charge'][0]))

    mgf_df = pd.DataFrame(
        {
            SOURCE_KEY: pd.Series(sources),
            SCAN_KEY: pd.Series(matched_scan_ids),
            INTENSITIES_KEY: pd.Series(matched_intensities),
            MZS_KEY: pd.Series(matched_mzs)
        }
    )
    if with_charge:
        mgf_df[CHARGE_KEY] = pd.Series(charge_list)

    mgf_df = mgf_df.drop_duplicates(subset=[SOURCE_KEY, SCAN_KEY])

    return mgf_df
