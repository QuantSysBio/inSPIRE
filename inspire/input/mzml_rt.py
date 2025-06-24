""" Functions for loading experimental spectra from mzml files.
"""
import numpy as np
import polars as pl
import pandas as pd
from pyteomics import mzml

from inspire.constants import (
    INTENSITIES_KEY,
    MZS_KEY,
    RT_KEY,
    SCAN_KEY,
    SOURCE_KEY,
)

def spread_rt(scan_mz, scan_rt, ms1_df):
    """ Function to reestiamte retention time.
    """
    scan_ms1_df = ms1_df[
        (ms1_df['scanRT'].apply(lambda x : abs(x-scan_rt) < 60))
    ]
    total = 0
    divisor = 0
    for _, df_row in scan_ms1_df.iterrows():
        matched_mz_ind = np.argmin(
            np.abs(df_row['mzs'] - scan_mz)
        )
        if abs(df_row['mzs'][matched_mz_ind] - scan_mz) < 0.01:
            total += df_row['scanRT']*df_row[INTENSITIES_KEY][matched_mz_ind]
            divisor += df_row[INTENSITIES_KEY][matched_mz_ind]

    if divisor == 0:
        return scan_rt

    return total/divisor

def process_mzml_file_with_rt_adjustment(mzml_filename, scan_ids, with_charge=False):
    """ Function to process an MzML file to find matches with scan IDs.

    Parameters
    ----------
    mzml_filename : str
        The mzml file from which we are reading.
    scan_ids : list of int
        A list of the scan IDs we require.

    Returns
    -------
    scans_df : pd.DataFrame
        A DataFrame of scan results.
    """

    filename = mzml_filename.split('/')[-1]

    ms1_scans = []
    ms2_scans = []

    with mzml.read(mzml_filename) as reader:
        for spectrum in reader:
            scan_id = int(spectrum['id'].split('scan=')[1])

            if spectrum['ms level'] == 1:
                new_spectrum = {
                    'scanRT': float(spectrum['scanList']['scan'][0]['scan start time'])*20,
                    SCAN_KEY: scan_id,
                    INTENSITIES_KEY: np.array(list(spectrum['intensity array'])),
                    MZS_KEY: np.array(list(spectrum['m/z array'])),
                }

                ms1_scans.append(new_spectrum)
            if scan_ids is None or scan_id in scan_ids:
                new_spectrum = {
                    SOURCE_KEY: filename[:-5],
                    SCAN_KEY: int(spectrum['id'].split('scan=')[-1]),
                    'precursor': int(
                        spectrum['precursorList']['precursor'][0]['spectrumRef'].split('scan=')[-1]
                    ),
                    'mz': float(
                        spectrum['precursorList']['precursor'][0]['isolationWindow'][
                            'isolation window target m/z'
                        ]
                    ),
                    INTENSITIES_KEY: np.array(list(spectrum['intensity array'])),
                    MZS_KEY: np.array(list(spectrum['m/z array'])),
                    'ms1Intensity': spectrum[
                        'precursorList'
                    ]['precursor'][0]['selectedIonList'][
                        'selectedIon'
                    ][0]['peak intensity'],
                }
                if with_charge:
                    new_spectrum['charge'] = (
                        int(spectrum['precursorList']['precursor'][0]['selectedIonList'][
                            'selectedIon'
                        ][0]['charge state'])
                    )

                ms2_scans.append(new_spectrum)

    scans_df =  pd.DataFrame(ms2_scans)
    ms1_df = pd.DataFrame(ms1_scans)
    scans_df = pd.merge(
        scans_df,
        ms1_df[['scan', 'scanRT']].rename(
            columns={'scan': 'precursor'}
        ),
        how='inner',
        on='precursor'
    )
    scans_df[RT_KEY] = scans_df[['mz', 'scanRT']].apply(
        lambda df_row : spread_rt(df_row['mz'], df_row['scanRT'], ms1_df), axis=1
    )

    scans_df = pl.from_pandas(scans_df)
    scans_df = scans_df.unique(subset=[SOURCE_KEY, SCAN_KEY], maintain_order=True)

    return scans_df
