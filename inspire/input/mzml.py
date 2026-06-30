""" Functions for loading experimental spectra from mzml files.
"""
import numpy as np
import pandas as pd
from pyteomics import mzml

from inspire.constants import CHARGE_KEY, INTENSITIES_KEY, MZS_KEY, SCAN_KEY, SOURCE_KEY


def process_mzml_file(mzml_filename, scan_ids, with_charge=False):
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

    with mzml.read(mzml_filename) as reader:
        for spectrum in reader:
            scan_id = int(spectrum.getNativeID().split('scan=')[1])

            if scan_id in scan_ids:
                mzml_filenames.append(filename)
                scan_id_list.append(scan_id)
                intensities_list.append(np.array(list(spectrum['intensity array'])))
                ion_list.append(np.array(list(spectrum['m/z array'])))
                if with_charge:
                    charge_list.append(int(spectrum['params']['charge'][0]))

    scans_df =  pd.DataFrame(
        {
            SOURCE_KEY: pd.Series(mzml_filenames),
            SCAN_KEY: pd.Series(scan_id_list),
            INTENSITIES_KEY: pd.Series(intensities_list),
            MZS_KEY: pd.Series(ion_list)
        }
    )
    if with_charge:
        scans_df[CHARGE_KEY] = pd.Series(charge_list)

    scans_df = scans_df.drop_duplicates(subset=[SOURCE_KEY, SCAN_KEY])

    return scans_df
