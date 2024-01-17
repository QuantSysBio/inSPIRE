""" Functions for reading in or output of .ssl files.
"""
import re

import pandas as pd

from inspire.utils import remove_source_suffixes

def ssl_file_to_inspire_format(ssl_loc):
    """ Function to read in data from .ssl format

    Parameters
    ----------
    ssl_loc : str
        Location of the ssl file.

    Returns
    -------
    pd.DataFrame : pd.DataFrame
        DataFrame from ssl data read in.
    """
    id_df = pd.read_csv(ssl_loc, sep='\t')
    id_df = id_df.rename(
        columns={
            'file': 'source',
            'sequence': 'peptide',
            'modifications': 'modifiedSequence',
        },
    )
    id_df['source'] = id_df['source'].apply(remove_source_suffixes)
    id_df['peptide'] = id_df['peptide'].apply(lambda x : re.sub( r"[^a-zA-Z]+", "", x))

    return id_df
