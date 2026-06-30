""" Functions for calculating difference between predicted and detected
    retention times.
"""
import pandas as pd
from pyteomics import achrom
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from inspire.constants import (
    LABEL_KEY,
    PEPTIDE_KEY,
    RT_KEY,
    SPECTRAL_ANGLE_KEY,
)

def _add_achrom_rt_preds(train_df, test_df):
    """ Function to fit and predict achrom's predictor.

    Parameters
    ----------
    train_df : pd.DataFrame
        A DataFrame on which to fit the retention time predictor.
    test_df : pd.DataFrame
        A DataFrame on which to predict retention time.

    Returns
    -------
    test_df : pd.DataFrame
        The input test_df with predRT as a new column.
    """
    retention_time_model = achrom.get_RCs_vary_lcp(
        train_df[PEPTIDE_KEY],
        train_df['retentionTime'],
        lcp_range=(-1.0, 1.0),
        term_aa=False,
        rcond=None
    )

    if 'C' not in retention_time_model['aa']:
        retention_time_model['aa']['C'] = retention_time_model['aa']['P']*0.956
    if 'W' not in retention_time_model['aa']:
        retention_time_model['aa']['W'] = retention_time_model['aa']['P']*0.956

    pred_rts = test_df[PEPTIDE_KEY].apply(
        lambda x :  achrom.calculate_RT(x, retention_time_model, raise_no_mod=False)
    )
    return pred_rts

def add_delta_irt(combined_df):
    """ Function to calculate difference between predicted and observed retention
        time for each PSM.

    Parameters
    ----------
    combined_df : pd.DataFrame
        A DataFrame of PSMs.

    Returns
    -------
    combined_df : pd.DataFrame
        The DataFrame updated with a deltaRT column.
    """
    if combined_df[RT_KEY].nunique() <= 1:
        combined_df['deltaRT'] = 0
        return combined_df

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    combined_df_list = []

    if 'iRT' in combined_df.columns:
        combined_df_irt_null_count = combined_df[combined_df['iRT'].isnull()].shape[0]
    else:
        combined_df_irt_null_count = 1

    for train, test in kfold.split(combined_df):
        train_df = combined_df.iloc[train]
        test_df = combined_df.iloc[test]
        top_spec_angle_cut = train_df[SPECTRAL_ANGLE_KEY].quantile(0.9)
        train_df = train_df[
            (train_df[SPECTRAL_ANGLE_KEY] > top_spec_angle_cut) &
            (train_df[LABEL_KEY] == 1)
        ]

        if combined_df_irt_null_count > 0:
            test_df['predRT'] = _add_achrom_rt_preds(train_df, test_df)
        else:
            reg = LinearRegression().fit(
                train_df[['iRT']],
                train_df[RT_KEY]
            )
            test_df['predRT'] = reg.predict(test_df[['iRT']])

        test_df['deltaRT'] = (test_df['predRT'] - test_df['retentionTime']).abs()
        combined_df_list.append(test_df)

    return pd.concat(combined_df_list)
