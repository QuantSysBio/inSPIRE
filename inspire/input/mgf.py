""" Functions for reading in scans results in mgf format.
"""
import re

import numpy as np
import polars as pl
from pyteomics import mgf

from inspire.constants import (
    CHARGE_KEY,
    INTENSITIES_KEY,
    MZS_KEY,
    RT_KEY,
    SCAN_KEY,
    SOURCE_KEY,
)

def process_mgf_file(
        mgf_filename,
        scan_ids,
        scan_file_format,
        source_list,
        combined_source_file=False,
        with_charge=False,
        with_retention_time=False,
        with_ms1=False,
    ):
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
    if with_retention_time:
        rt_list = []
    if with_ms1:
        ms1_intes = []

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
                if combined_source_file:
                    source = spectrum['params']['title'].split('File:"')[-1].split('.raw"')[0]
                else:
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
                if with_retention_time:
                    rt_list.append(float(spectrum['params']['rtinseconds']))
                if with_ms1:
                    try:
                        ms1_intes.append(float(spectrum['params']['pepmass'][1]))
                    except Exception: # pylint: disable=bare-except
                        ms1_intes.append(-1.0)

    mgf_data = {
        SOURCE_KEY: pl.Series(sources),
        SCAN_KEY: pl.Series(matched_scan_ids),
        INTENSITIES_KEY: pl.Series(matched_intensities),
        MZS_KEY: pl.Series(matched_mzs)
    }

    if with_charge:
        mgf_data[CHARGE_KEY] = pl.Series(charge_list)
    if with_retention_time:
        mgf_data[RT_KEY] = pl.Series(rt_list)
    if with_ms1:
        mgf_data['ms1Intensity'] = pl.Series(ms1_intes)

    mgf_df = pl.DataFrame(mgf_data)
    mgf_df = mgf_df.unique(subset=[SOURCE_KEY, SCAN_KEY])

    return mgf_df
