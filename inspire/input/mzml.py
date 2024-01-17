""" Functions for loading experimental spectra from mzml files.
"""
import numpy as np
import polars as pl
from pyteomics import mzml

from inspire.constants import (
    CHARGE_KEY,
    INTENSITIES_KEY,
    MZS_KEY,
    RT_KEY,
    SCAN_KEY,
    SOURCE_KEY,
)


def process_mzml_file(mzml_filename, scan_ids, with_charge=False, with_retention_time=False):
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
    ion_list = []
    intensities_list = []
    scan_id_list = []
    mzml_filenames = []
    filename = mzml_filename.split('/')[-1]

    if with_charge:
        charge_list = []

    if with_retention_time:
        rt_list = []

    with mzml.read(mzml_filename) as reader:
        for spectrum in reader:
            scan_id = int(spectrum['id'].split('scan=')[1])
            if scan_ids is None or scan_id in scan_ids:
                mzml_filenames.append(filename[:-5])
                scan_id_list.append(scan_id)
                intensities_list.append(np.array(list(spectrum['intensity array'])))
                ion_list.append(np.array(list(spectrum['m/z array'])))

                if with_charge:
                    charge_list.append(
                        int(spectrum['precursorList']['precursor'][0]['selectedIonList'][
                            'selectedIon'
                        ][0]['charge state'])
                    )

                if with_retention_time:
                    rt_list.append(float(spectrum['scanList']['scan'][0]['scan start time']))

    scans_df =  pl.DataFrame(
        {
            SOURCE_KEY: pl.Series(mzml_filenames),
            SCAN_KEY: pl.Series(scan_id_list),
            INTENSITIES_KEY: pl.Series(intensities_list),
            MZS_KEY: pl.Series(ion_list)
        }
    )
    if with_charge:
        scans_df[CHARGE_KEY] = pl.Series(charge_list)
    if with_retention_time:
        scans_df[RT_KEY] = pl.Series(rt_list)

    scans_df = scans_df.unique(subset=[SOURCE_KEY, SCAN_KEY])

    return scans_df
