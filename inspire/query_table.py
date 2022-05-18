""" Functions for creating a query table of inspire assignments to ground
    truth datasets.
"""
import pandas as pd

from inspire.constants import (
    ACCESSION_KEY,
    ENDC_TEXT,
    ENGINE_SCORE_KEY,
    FINAL_SCORE_KEY,
    OKCYAN_TEXT,
    PEPTIDE_KEY,
    PRED_PEPTIDE_KEY,
    SCAN_KEY,
    SOURCE_KEY,
    TRUE_ACCESSION_KEY,
    TRUE_PEPTIDE_KEY,
)

def create_query_table(config):
    """ Function for creating a query table of inspire assignments to ground
        truth datasets.

    Paramters
    ---------
    config : inspire.config.Config
        Config object for the experiment.
    """
    if config.query_table is not None:
        rescored_df = pd.read_csv(f'{config.output_folder}/finalAssignments.csv')
        ground_df = pd.read_csv(config.query_table)

        rescored_df = rescored_df[[
            SOURCE_KEY,
            SCAN_KEY,
            PEPTIDE_KEY,
            ACCESSION_KEY,
            ENGINE_SCORE_KEY,
            FINAL_SCORE_KEY,
        ]].rename(columns={PEPTIDE_KEY: PRED_PEPTIDE_KEY})

        rescored_df = rescored_df.drop(ACCESSION_KEY, axis=1)

        ground_df_cols = [
            config.qt_source_key,
            config.qt_scan_key,
            config.qt_seq_key,
            config.qt_accession_stratum_key
        ]

        ground_df = ground_df[ground_df_cols]
        ground_df = ground_df.rename(
            columns={
                config.qt_accession_stratum_key: TRUE_ACCESSION_KEY,
                config.qt_scan_key: SCAN_KEY,
                config.qt_seq_key: TRUE_PEPTIDE_KEY,
                config.qt_source_key: SOURCE_KEY,
            }
        )

        ground_df = pd.merge(
            ground_df,
            rescored_df,
            how='left',
            on=[SOURCE_KEY, SCAN_KEY]
        )

        ground_df.to_csv(
            f'{config.output_folder}/queryTable.csv',
            index=False
        )
        print(
            OKCYAN_TEXT +
            '\tQuery table created.' +
            ENDC_TEXT
        )
