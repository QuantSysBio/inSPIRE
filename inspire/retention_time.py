""" Functions for calculating difference between predicted and detected
    retention times.
"""
import os

import pandas as pd
import polars as pl
from pyteomics import achrom
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from inspire.constants import (
    ACCESSION_STRATUM_KEY,
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

def add_delta_irt(combined_df, config, scan_file):
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
    if combined_df[RT_KEY].n_unique() <= 1:
        combined_df['deltaRT'] = 0
        return combined_df

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    combined_df_list = []
    coefficents = []
    intercepts = []

    if 'iRT' in combined_df.columns:
        combined_df_irt_null_count = combined_df.filter(
            pl.col('iRT').is_null()
        ).shape[0]
    else:
        combined_df_irt_null_count = 1

    if config.rt_fit_loc is not None:
        if os.path.exists(f'{config.rt_fit_loc}/rt_fit_{scan_file}.csv'):
            rt_df = pd.read_csv(f'{config.rt_fit_loc}/rt_fit_{scan_file}.csv')
        else:
            rt_dfs = [x for x in os.listdir(config.rt_fit_loc) if x.startswith('rt_fit')]
            rt_df = pd.read_csv(f'{config.rt_fit_loc}/{rt_dfs[0]}')

        coef = rt_df['coefficents'].mean()
        intercept = rt_df['intercepts'].mean()
        combined_df = combined_df.with_columns(
            pl.col('iRT').apply(lambda x : (x*coef)+ intercept).alias('predRT')
        )
        combined_df = combined_df.with_columns(
            (pl.col('predRT') - pl.col('retentionTime')).truediv(coef).abs().alias('deltaRT')
        )
        return combined_df

    try:
        for train, test in kfold.split(combined_df):
            train_df = combined_df[train, :]
            test_df = combined_df[test, :]

            if ACCESSION_STRATUM_KEY in train_df.columns:
                train_df = train_df.filter(
                    pl.col(ACCESSION_STRATUM_KEY).eq(0)
                )

            if LABEL_KEY in train_df.columns:
                train_df = train_df.filter(
                    pl.col(LABEL_KEY).eq(1)
                )

            top_spec_angle_cut = train_df[SPECTRAL_ANGLE_KEY].quantile(0.9)
            train_df = train_df.filter(
                (pl.col(SPECTRAL_ANGLE_KEY) > top_spec_angle_cut)
            )

            if combined_df_irt_null_count > 0:
                test_df['predRT'] = _add_achrom_rt_preds(train_df, test_df)
            else:
                reg = LinearRegression().fit(
                    train_df.select('iRT').to_numpy(),
                    train_df[RT_KEY].to_numpy(),
                )
                coefficents.append(reg.coef_[0])
                intercepts.append(reg.intercept_)
                pred_rt = pl.Series(
                    reg.predict(test_df.select('iRT').to_numpy())
                )
                test_df = test_df.with_columns(
                    pred_rt.alias('predRT')
                )

            test_df = test_df.with_columns(
                (pl.col('predRT') - pl.col('retentionTime')).abs().alias('deltaRT')
            )
            combined_df_list.append(test_df)
    except ValueError:
        try:
            # Simplest way to avoid errors on tiny files.
            train_df = train_df.filter(pl.col(LABEL_KEY).eq(1))
            reg = LinearRegression().fit(
                train_df.select('iRT').to_numpy(),
                train_df[RT_KEY].to_numpy(),
            )
            coefficents.append(reg.coef_[0])
            intercepts.append(reg.intercept_)
            pred_rt = pl.Series(
                reg.predict(combined_df.select('iRT').to_numpy())
            )
            combined_df = combined_df.with_columns(
                pred_rt.alias('predRT')
            )
            combined_df = combined_df.with_columns(
                (pl.col('predRT') - pl.col('retentionTime')).abs().alias('deltaRT')
            )
        except ValueError:
            reg = LinearRegression().fit(
                combined_df.select('iRT').to_numpy(),
                combined_df[RT_KEY].to_numpy(),
            )
            coefficents.append(reg.coef_[0])
            intercepts.append(reg.intercept_)
            pred_rt = pl.Series(
                reg.predict(combined_df.select('iRT').to_numpy())
            )
            combined_df = combined_df.with_columns(
                pred_rt.alias('predRT')
            )
            combined_df = combined_df.with_columns(
                (pl.col('predRT') - pl.col('retentionTime')).abs().alias('deltaRT')
            )


    rt_df = pd.DataFrame({
        'coefficents': coefficents,
        'intercepts': intercepts,
    })
    if scan_file is not None:
        rt_df.to_csv(f'{config.output_folder}/rt_fit_{scan_file}.csv', index=False)

    return pl.concat(combined_df_list)
