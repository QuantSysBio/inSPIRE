""" Functions for writing percolator/mokapot input using Prosit and other features.
"""
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import os
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBRegressor
import yaml

from inspire.basic_features import create_basic_features
from inspire.constants import(
    ACCESSION_STRATUM_KEY,
    ACCESSION_KEY,
    BASIC_FEATURES,
    CHARGE_KEY,
    ENDC_TEXT,
    FRAG_MZ_ERR_VAR_KEY,
    FRAG_MZ_ERR_MED_KEY,
    LABEL_KEY,
    MINIMAL_FEATURE_SET,
    OKCYAN_TEXT,
    IN_ACCESSION_KEY,
    PEPTIDE_KEY,
    PSM_ID_KEY,
    PEARSON_KEY,
    PERC_SCAN_ID,
    PISCES_BA_FEATURE_SETS,
    PISCES_CASA_BA_FEATURE_SETS,
    PISCES_CASA_NOBA_FEATURE_SETS,
    PISCES_NOBA_FEATURE_SETS,
    PTM_ID_KEY,
    PTM_NAME_KEY,
    PTM_SEQ_KEY,
    PTM_WEIGHT_KEY,
    RT_KEY,
    SCAN_KEY,
    SOURCE_INDEX_KEY,
    SOURCE_KEY,
    WARNING_TEXT,
)
from inspire.input.mgf import process_mgf_file
from inspire.input.mhcpan import read_mhcpan_output
from inspire.input.msp import msp_to_df
from inspire.input.mzml import process_mzml_file
from inspire.input.mzml_rt import process_mzml_file_with_rt_adjustment
from inspire.input.search_results import generic_read_df
from inspire.prepare import create_prosit_mod_seq
from inspire.retention_time import add_delta_irt
from inspire.spectral_features import (
    SPECTRAL_FEATURES,
    DELTA_FEATURES,
    create_spectral_features,
)
from inspire.utils import (
    accession_informed_filter,
    check_bad_mods,
    fetch_collision_energy,
    fetch_proteome,
    get_nuggets_pred_df,
    get_ox_flag,
    modify_sequence_for_skyline,
    parallel_remap,
    remove_source_suffixes,
)

BATCH_SIZE = 3_000

def combine_spectral_data(
        search_df, scan_df, prosit_df, ox_flag, spectral_predictor,
        output_folder, for_calibration=False, collision_energy=None
    ):
    """ Function to combine DataFrame of PEAKS results with prosit predicted
        spectrum and the true eperimental spectrum.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        The DataFrame of PEAKS search results.
    mzml_df : pd.DataFrame
        The DataFrame of experimental observed spectra.
    prosit_df : pd.DataFrame
        The DataFrame of prosit predicted spectra.

    Returns
    -------
    final_df : pd.DataFrame
        The DataFrame with all merged Data.
    """
    if spectral_predictor == 'prosit':
        search_df = search_df.with_columns(
            pl.struct([PEPTIDE_KEY, PTM_SEQ_KEY]).map_elements(
                lambda x : create_prosit_mod_seq(x[PEPTIDE_KEY], x[PTM_SEQ_KEY], ox_flag),
                skip_nulls=False, return_dtype=pl.String,
            ).alias('modified_sequence')
        )
        prosit_df = prosit_df.unique(
            subset=['modified_sequence', CHARGE_KEY, 'collisionEnergy']
        )


    print(
        OKCYAN_TEXT +
        f'\t\t\t{search_df.shape[0]} in original search results.' +
        ENDC_TEXT
    )
    if spectral_predictor == 'prosit':
        merge_keys = ['modified_sequence', CHARGE_KEY]
        if for_calibration is False:
            if collision_energy is None:
                collision_energy = fetch_collision_energy(output_folder)

            if isinstance(collision_energy, dict):
                search_df = search_df.with_columns(
                    pl.col('source').replace(collision_energy).cast(
                        pl.Int64
                    ).alias('collisionEnergy')
                )
                merge_keys += ['collisionEnergy']

        df_with_prosit = search_df.join(
            prosit_df,
            how='inner',
            on=merge_keys
        )
    elif spectral_predictor == 'ms2pip':
        df_with_prosit = search_df.join(
            prosit_df,
            how='inner',
            on=[PEPTIDE_KEY, PTM_SEQ_KEY, CHARGE_KEY]
        )

    print(
        OKCYAN_TEXT +
        f'\t\t\t{df_with_prosit.shape[0]} after combination with predicted spectra.'
        + ENDC_TEXT
    )

    scan_df = scan_df.with_columns(
        pl.col(SOURCE_KEY).map_elements(
            remove_source_suffixes, return_dtype=pl.String,
        ).alias(SOURCE_KEY)
    )
    scan_df = scan_df.with_columns(pl.col(SOURCE_KEY).cast(str))
    df_with_prosit = df_with_prosit.with_columns(pl.col(SOURCE_KEY).cast(str))

    final_df = df_with_prosit.join(
        scan_df,
        how='inner',
        on=[SOURCE_KEY, SCAN_KEY]
    )

    print(
        OKCYAN_TEXT +
        f'\t\t\t{final_df.shape[0]} after combination with experimental spectra.' +
        ENDC_TEXT
    )

    return final_df


def get_global_features(combined_df, config):
    """ Function to get global features.
    """
    if config.rt_fit_loc is not None:
        base_dir = config.rt_fit_loc
    else:
        base_dir = config.output_folder

    if os.path.exists(f'{base_dir}/benchmark_psms.csv'): #TODO fix this pathway or delete it.
        bench_df = pl.read_csv(f'{base_dir}/benchmark_psms.csv')
        filtered_df = combined_df.join(
            bench_df, how='inner', on=[SOURCE_KEY, SCAN_KEY, PEPTIDE_KEY]
        )
        collision_energy = fetch_collision_energy(base_dir)
        if isinstance(collision_energy, dict):
            combined_df = combined_df.with_columns(
                pl.lit(filtered_df['deltaRT'].median()).alias('dRtMedian'),
                pl.col('source').replace(collision_energy).cast(
                    pl.Int64
                ).alias('collisionEnergy')
            )
        else:
            combined_df = combined_df.with_columns(
                pl.lit(filtered_df['deltaRT'].median()).alias('dRtMedian'),
                pl.lit(collision_energy).alias('collisionEnergy'),
            )

        if config.use_binding_affinity == 'asFeature':
            filtered_df = combined_df.filter(pl.col('sequenceLength') == 9)
            nugget_pred_df = get_nuggets_pred_df(config)
            filtered_df = filtered_df.join(nugget_pred_df, how='inner', on='peptide')
            combined_df = combined_df.with_columns(
                pl.lit(filtered_df['mhcpanPrediction'].median()).alias('baMedian'),
                pl.lit(filtered_df['nuggetsPrediction'].median()).alias('nuggetsMedian'),
            )

    else:
        meta_features = ['dRtMedian', 'collisionEnergy']
        if config.use_binding_affinity == 'asFeature':
            meta_features.extend(['mhcpanMedian', 'nuggetsMedian'])
        with open(f'{base_dir}/meta_data.yml', 'r', encoding='UTF-8') as stream:
            meta_dict = yaml.safe_load(stream)
        for meta_feature in meta_features:
            if isinstance(meta_dict[meta_feature], dict):
                combined_df = combined_df.with_columns(
                    pl.col('source').replace(meta_dict[meta_feature]).cast(
                        pl.Int64
                    ).alias(meta_feature)
                )
            else:
                combined_df = combined_df.with_columns(
                    pl.lit(meta_dict[meta_feature]).alias(meta_feature)
                )

    return combined_df


def filter_input_columns(combined_df, config, file_idx):
    """ Function to filter the final percolator/mokapot input DataFrame to the required
        features.

    Parameters
    ----------
    combined_df : pd.DataFrame
        The percolator/mokapot input DataFrame.
    config : inspire.config.Config
        The config object.

    Returns
    -------
    combined_df : pd.DataFrame
        The input DataFrame containing only the features required for percolator/mokapot.
    """
    psm_id_key = PSM_ID_KEY[config.rescore_method]

    if config.for_pisces:
        combined_df = combined_df.with_columns(
            pl.col('specID').map_elements(
                lambda x : x.split('_')[-1], return_dtype=pl.String,
            ).alias('modifiedSequence')
        )
        combined_df = combined_df.with_columns(
            pl.col('modifiedSequence').str.count_matches(
                '[+119.0]', literal=True
            ).alias('nCysteinylation'),
            pl.col('modifiedSequence').str.count_matches(
                '[+1.0]', literal=True
            ).alias('nDeamidation'),
            (
                pl.col('modifiedSequence').str.count_matches('[+42.0]', literal=True) +
                pl.col('modifiedSequence').str.count_matches('[+43.0]', literal=True) +
                pl.col('modifiedSequence').str.count_matches('[-17.0]', literal=True) +
                pl.col('modifiedSequence').str.count_matches('[+26.0]', literal=True)
            ).alias('nTermMods'),
            pl.col('modifiedSequence').str.count_matches(
                '[+42.0]', literal=True
            ).alias('nAcetylation'),
            (
                pl.col('modifiedSequence').str.count_matches('C', literal=True) -
                (pl.col('modifiedSequence').str.count_matches('C[+57.0]', literal=True) +
                pl.col('modifiedSequence').str.count_matches('C[+119.0]', literal=True))
            ).alias('nUnmodifiedC'),
            pl.col('peptide').str.count_matches('L').alias('nLeucine')
        )
        combined_df = get_global_features(combined_df, config)
        if config.use_binding_affinity == 'asFeature':
            if config.search_engine == 'peaksDeNovo' or config.pisces_dn_method == 'peaksDeNovo':
                pisces_extra_feats = [
                    feat for feat in PISCES_BA_FEATURE_SETS[0] if feat not in BASIC_FEATURES
                ]
            else:
                pisces_extra_feats = [
                    feat for feat in PISCES_CASA_BA_FEATURE_SETS[0] if feat not in BASIC_FEATURES
                ]
        else:
            if config.search_engine == 'peaksDeNovo' or config.pisces_dn_method == 'peaksDeNovo':
                pisces_extra_feats = [
                    feat for feat in PISCES_NOBA_FEATURE_SETS[0] if feat not in BASIC_FEATURES
                ]
            else:
                pisces_extra_feats = [
                    feat for feat in PISCES_CASA_NOBA_FEATURE_SETS[0] if feat not in BASIC_FEATURES
                ]
        use_cols = (
            [psm_id_key, LABEL_KEY, PERC_SCAN_ID] + BASIC_FEATURES + pisces_extra_feats
        )
    elif config.minimal_features:
        use_cols = [
            psm_id_key,
            LABEL_KEY,
            PERC_SCAN_ID,
        ] + BASIC_FEATURES + MINIMAL_FEATURE_SET
    else:
        use_cols = [
            psm_id_key,
            LABEL_KEY,
            PERC_SCAN_ID,
        ] + BASIC_FEATURES + SPECTRAL_FEATURES

        if config.delta_method != 'ignore':
            use_cols += DELTA_FEATURES

        use_cols += ['deltaRT']

    
    if config.use_binding_affinity == 'asFeature':
        nugget_pred_df = None
        for idx, allele in enumerate(config.alleles):
            if not os.path.exists(f'{config.output_folder}/{allele}_nuggets.csv'):
                continue
            if idx == 0:
                nugget_pred_df = pl.read_csv(f'{config.output_folder}/{allele}_nuggets.csv')
                nugget_pred_df = nugget_pred_df.rename({'ic50': f'ic50_{allele}'})
            else:
                mini_pred_df = pl.read_csv(f'{config.output_folder}/{allele}_nuggets.csv')
                mini_pred_df = mini_pred_df.rename({'ic50': f'ic50_{allele}'})
                nugget_pred_df = nugget_pred_df.join(
                    mini_pred_df, how='inner', on='peptide',
                )

        if nugget_pred_df is None:
            combined_df = combined_df.with_columns(
                pl.lit(None).alias('nuggetsPrediction')
            )
        else:
            nugget_pred_df = nugget_pred_df.with_columns(
                pl.min_horizontal(*[f'ic50_{allele}' for allele in config.alleles]).log10().alias(
                    'nuggetsPrediction'
                )
            )
            combined_df = combined_df.join(
                nugget_pred_df.select(['peptide', 'nuggetsPrediction']),
                how='left', on='peptide',
            )
            combined_df = combined_df.with_columns(
                pl.col('nuggetsPrediction').fill_null(5.0)
            )

        if not config.for_pisces:
            use_cols += ['mhcpanPrediction', 'nuggetsPrediction']


    if config.combined_scans_file is not None:
        if config.source_files is not None:
            scan_files = [remove_source_suffixes(x) for x in config.source_files]
        else:
            scan_files = sorted(combined_df[SOURCE_KEY].unique().tolist())
        combined_df = combined_df.with_columns(
            pl.col(SOURCE_KEY).map_elements(scan_files.index).alias(SOURCE_INDEX_KEY)
        )
    else:
        combined_df = combined_df.with_columns(
            pl.lit(file_idx).alias(SOURCE_INDEX_KEY)
        )

    if isinstance(config.collision_energy, list):
        use_cols += ['collisionEnergy']

    use_cols += [SOURCE_INDEX_KEY]


    if config.use_accession_stratum:
        acc_cols = [ACCESSION_STRATUM_KEY]
        for a_idx in range(len(config.accession_hierarchy)):
            stratum_name = config.accession_hierarchy[int(a_idx)]
            combined_df = combined_df.with_columns(
                pl.col(ACCESSION_STRATUM_KEY).map_elements(
                    lambda x, val=a_idx: 1 if x == val else 0
                ).alias(f'accession_{stratum_name}')
            )
            acc_cols.append(f'accession_{stratum_name}')

        use_cols += sorted(acc_cols)

    use_cols += [PEPTIDE_KEY, IN_ACCESSION_KEY[config.rescore_method]]

    if config.remap_to_proteome:
        proteome = fetch_proteome(config.proteome, with_desc=False)
        pos_df = parallel_remap(
            combined_df.filter(pl.col(LABEL_KEY).eq(1)),
            config.n_cores,
            proteome,
            IN_ACCESSION_KEY[config.rescore_method],
        )
        neg_df = parallel_remap(
            combined_df.filter(pl.col(LABEL_KEY).ne(1)),
            config.n_cores,
            proteome,
            IN_ACCESSION_KEY[config.rescore_method],
            reverse=True,
        )
        combined_df = pl.concat([pos_df, neg_df])
        if config.drop_unknown:
            combined_df = combined_df.filter(
                pl.col(IN_ACCESSION_KEY[config.rescore_method]).ne('unknown') |
                pl.col(LABEL_KEY).ne(1)
            )

        combined_df = combined_df.drop(ACCESSION_KEY)
    else:
        combined_df = combined_df.rename(
            {ACCESSION_KEY: IN_ACCESSION_KEY[config.rescore_method]}
        )

    combined_df = combined_df.sort(by='tempIndex')
    combined_df = combined_df.select(use_cols)

    return combined_df

def write_with_spectral_features(
        search_df,
        mods_df,
        config,
    ):
    """ Function to write the percolator/mokapot input DataFrame with spectral features.

    Parameters
    ----------
    search_df : pd.DataFrame
        The results from the original search DataFrame.
    mods_df : pd.DataFrame
        The DataFrame of ptms.
    config : inspire.config.Config
        The Config object.
    """
    if config.combined_scans_file is not None:
        scan_files = sorted([remove_source_suffixes(config.combined_scans_file)])
    else:
        scan_files = sorted(search_df[SOURCE_KEY].unique().to_list())


    max_scan = search_df[SCAN_KEY].max()
    for file_idx, scan_file in enumerate(scan_files):
        func_args = generate_function_arguments(
            search_df, mods_df, config, file_idx, scan_file
        )
        process_single_file(
            func_args, config, file_idx, scan_file, max_scan
        )

    print(
        OKCYAN_TEXT + '\t\t\tFull input DataFrame written to csv.' + ENDC_TEXT
    )


def add_perc_scan_id(combined_df, file_idx, max_scan):
    """ Function add a Percolator scan ID to the DataFrame that will be unique
        to scans across RAW files.

    Parameters
    ----------
    combined_df : pl.DataFrame
        DataFrame of PSMs with combined spectral information.
    config : inspire.config.Config
        Config object for the whole experiment.
    file_idx : int
        The index of the raw file being processed.
    max_scan : int
        The highest scan number across all files.

    Returns
    -------
    combined_df : pl.DataFrame
        Input DataFrame with scannr column added.
    """
    combined_df = combined_df.with_columns(
        pl.col(SCAN_KEY).map_elements(
            lambda x, f_id=file_idx : f_id * max_scan  + x,
            return_dtype=pl.Int64,
        ).alias(PERC_SCAN_ID)
    )

    return combined_df

def generate_function_arguments(
        search_df, mods_df, config, file_idx, scan_file
    ):
    """ Function to process all PSMs from a single mgf or mzML file.

    Parameters
    ----------
    search_df : pd.DataFrame
        A DataFrame of PSMs.
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs present in the data.
    prosit_df : pd.DataFrame
        A DataFrame of the predicted spectra from Prosit.
    config : inspire.config.Config
        The config file for the experiment.
    file_idx : int
        The index of the file being processed.
    scan_file : str
        The name of the file being processed.
    """
    ox_flag = get_ox_flag(mods_df)
    if config.spectral_predictor == 'prosit':
        prosit_predictions = f'{config.output_folder}/prositPredictions.msp'
        prosit_df = msp_to_df(prosit_predictions, 'prosit', None)
    elif config.spectral_predictor == 'ms2pip':
        model = config.ms2pip_model
        ms2pip_predictions = f'{config.output_folder}/ms2pipInput_{model}_predictions.msp'
        prosit_df = msp_to_df(ms2pip_predictions, 'ms2pip', mods_df)

    print(
        OKCYAN_TEXT +
        f'\t\tProcessing scan file {file_idx}, {scan_file}.' +
        ENDC_TEXT
    )
    if config.combined_scans_file is not None:
        filtered_search_df = search_df
    else:
        filtered_search_df = search_df.filter(
            pl.col(SOURCE_KEY).eq(scan_file)
        )

    scans = filtered_search_df.select(SCAN_KEY).to_series().unique()

    if RT_KEY not in filtered_search_df.columns:
        with_rt = True
    elif filtered_search_df.select(RT_KEY).n_unique() <= 1:
        filtered_search_df = filtered_search_df.drop(RT_KEY)
        with_rt = True
    else:
        with_rt = False

    if config.scans_format == 'mzML':
        scan_df = process_mzml_file(
            f'{config.scans_folder}/{scan_file}.{config.scans_format}',
            set(scans.to_list()),
            with_retention_time=with_rt,
        )
    elif config.scans_format == 'mzML_rt':
        scan_df = process_mzml_file_with_rt_adjustment(
            f'{config.scans_folder}/{scan_file}.mzML',
            set(scans.to_list()),
        )
        if RT_KEY in filtered_search_df.columns:
            filtered_search_df = filtered_search_df.drop(RT_KEY)
    else:
        scan_df = process_mgf_file(
            f'{config.scans_folder}/{scan_file}.{config.scans_format}',
            set(scans.to_list()),
            config.scan_title_format,
            config.source_files,
            combined_source_file=config.combined_scans_file is not None,
            with_retention_time=with_rt,
            with_ms1=True,
        )

    scan_df = scan_df.unique(subset=[SOURCE_KEY, SCAN_KEY])
    combined_df = combine_spectral_data(
        filtered_search_df,
        scan_df,
        prosit_df,
        ox_flag,
        config.spectral_predictor,
        config.output_folder,
        collision_energy=config.collision_energy,
    )

    print(
        OKCYAN_TEXT + '\t\t\tCombined DB Search, Spectral, and Prosit Data.' + ENDC_TEXT
    )

    if not combined_df.shape[0]:
        print(
            WARNING_TEXT +
            f'Warning. No matched scans found for source file {scan_file}' +
            ENDC_TEXT
        )
        return None

    n_batches = combined_df.shape[0]//BATCH_SIZE
    additional_psms = combined_df.shape[0]%BATCH_SIZE
    batch_values = []
    for batch_idx in range(n_batches):
        batch_values.extend([batch_idx]*BATCH_SIZE)
    batch_values.extend([n_batches]*additional_psms)

    combined_df = combined_df.with_columns(
        pl.Series(name='batch', values=batch_values)
    )
    combined_df_list = combined_df.partition_by('batch')
    func_args = []
    for comb_df in combined_df_list:
        func_args.append([comb_df, mods_df, config])

    return func_args


class ChildRegressor(XGBRegressor):
    """ Class to allow use of Prosit-delta predictor in shared memory.
    """
    def __init__(self):
        super().__init__()
        home = str(Path.home())
        self.load_model(f'{home}/inSPIRE_models/models/prosit_delta_v1.json')
        self.set_params(n_jobs=1)
        self.set_params(random_state=42)

    def best_score(self):
        return 1.0

    def coef_(self):
        return np.array([1.0])

    def feature_names_in_(self):
        return np.array([''])

    def intercept_(self):
        return np.array([1.0])

class CustomManager(BaseManager):
    # nothing
    pass

def process_single_file(
        func_args, config, file_idx, scan_file, max_scan
    ):
    """ Function to process PSMs from a single raw file in parallel.

    Parameters
    ----------
    func_args : list of tuples
        A list of the arugments to be passed to each parallel execution of
        the create_spectral_features function.
    config : inspire.config.Config
        The Config object for the whole experiment.
    file_idx : int
        The index of the file being processed.
    scan_file : str
        The name of the file being processed.
    max_scan : int
        The maximum scan value in the dataset (needed to ensure uniqueness
        in Percolator scan ID).
    """
    CustomManager.register('ChildRegressor', ChildRegressor)
    with CustomManager() as manager:
        if config.delta_method != 'ignore' and not config.minimal_features:
            model = manager.ChildRegressor()
            # Avoid unecessary parallelism when model applied to small sets.
            model.set_params(n_jobs=1)
        else:
            model = None

        for entry in func_args:
            entry.append(model)

        func_args = [tuple(arg_group) for arg_group in func_args]

        with mp.get_context('spawn').Pool(processes=config.n_cores) as pool:
            results_dfs = pool.starmap(create_spectral_features, func_args)

    select_columns = results_dfs[0].columns
    combined_df = pl.concat([x.select(select_columns) for x in results_dfs])
    replace_feats = [
        FRAG_MZ_ERR_VAR_KEY,
        FRAG_MZ_ERR_MED_KEY,
        PEARSON_KEY,
    ]
    if config.delta_method != 'ignore' and not config.minimal_features:
        replace_feats += DELTA_FEATURES

    for feat in set(replace_feats):
        if feat in select_columns:
            feat_median = combined_df.filter(
                pl.col(feat).gt(-1)
            ).select(pl.median(feat))[feat][0]

            combined_df = combined_df.with_columns(
                pl.when(pl.col(feat).eq(-1)).then(feat_median)
                    .otherwise(pl.col(feat)).alias(feat)
            )

    combined_df = combined_df.sort(by='spectralAngle', descending=True)
    if isinstance(config.collision_energy, list):
        combined_df = combined_df.unique(subset=['source', 'scan', 'peptide'])

    combined_df = combined_df.sort(by='tempIndex')
    combined_df = add_delta_irt(combined_df, config, scan_file)

    print(
        OKCYAN_TEXT + '\t\t\tCreated Spectral and Delta RT Features.' + ENDC_TEXT
    )
    combined_df = add_perc_scan_id(combined_df, file_idx, max_scan)

    combined_df = filter_input_columns(combined_df, config, file_idx)

    file_loc = _write_to_tab_file(combined_df, file_idx, config.output_folder)

    return file_loc

def _write_to_tab_file(combined_df, file_idx, output_folder):
    """ Function to write percolator input in tab format.

    Parameters
    ----------
    combined_df : pd.DataFrame
        A DataFrame of processed search results.
    file_idx : int
        The index of the scan file used.
    output_folder : str
        The folder where all inSPIRE output is written.
    """
    if file_idx == 0:
        combined_df.write_csv(
            f'{output_folder}/input_all_features.tab',
            separator='\t',
        )
    else:
        with open(f'{output_folder}/input_all_features.tab', mode='ab') as out_file:
            combined_df.write_csv(
                out_file,
                separator='\t',
                include_header=False,
            )

    return f'{output_folder}/input_all_features.tab'

def write_rescoring_features(
        search_df,
        mods_df,
        config,
    ):
    """ Function to write the percolator/mokapot input DataFrame.

    Parameters
    ----------
    search_df : pd.DataFrame
        The results from the original search DataFrame.
    mods_df : pd.DataFrame
        The DataFrame of ptms.
    config : inspire.config.Config
        The Config object.
    """
    feature_df = create_basic_features(search_df, mods_df)

    if config.use_binding_affinity == 'asFeature':
        mhc_pan_df = read_mhcpan_output(f'{config.output_folder}/mhcpan', alleles=config.alleles)
        mhc_pan_df = mhc_pan_df.rename({
            'Peptide': PEPTIDE_KEY,
            'Aff(nM)': 'mhcpanPrediction'
        })
        mhc_pan_df = mhc_pan_df.with_columns(
            pl.col('mhcpanPrediction').log10()
        )

        mhc_pan_df = mhc_pan_df.select([PEPTIDE_KEY, 'mhcpanPrediction'])
        feature_df = feature_df.join(
            mhc_pan_df,
            on=PEPTIDE_KEY,
            how='left'
        )
        feature_df.select('mhcpanPrediction')
        feature_df = feature_df.with_columns(
            pl.col('mhcpanPrediction').fill_null(5.0),
        )


    psm_id_key = PSM_ID_KEY[config.rescore_method]
    if mods_df.empty:
        mod_weights = {}
    else:
        mod_weights = dict(zip(mods_df[PTM_ID_KEY].tolist(), mods_df[PTM_WEIGHT_KEY].tolist()))

    feature_df = feature_df.with_columns(
        pl.struct([PEPTIDE_KEY, PTM_SEQ_KEY]).map_elements(
            lambda x : modify_sequence_for_skyline(x, mod_weights),
            skip_nulls=False, return_dtype=pl.String,
        ).alias('modSeq')
    )

    feature_df = feature_df.with_columns(
        pl.struct([SOURCE_KEY, SCAN_KEY, 'modSeq']).map_elements(
            lambda x : str(x[SOURCE_KEY]) + '_' + str(x[SCAN_KEY]) + '_' + str(x['modSeq']),
            skip_nulls=False, return_dtype=pl.String,
        ).alias(psm_id_key)
    )
    print(
        OKCYAN_TEXT +
        '\tBasic Features added, adding Spectral Features.' +
        ENDC_TEXT
    )

    write_with_spectral_features(
        feature_df,
        mods_df,
        config,
    )

def process_unknown_modifications(target_df, mods_df, config):
    """ Function to handle modifications which are unknown to the Prosit spectral predictor
        (as well as unmodified cysteines).

    Parameters
    ----------
    Parameters
    ----------
    search_df : pd.DataFrame
        The results from the original search DataFrame.
    mods_df : pd.DataFrame
        The DataFrame of ptms.
    config : inspire.config.Config
        The Config object.

    Returns
    -------
    target_df : pd.DataFrame
        The results from the original search DataFrame with unknown modifications filtered.
    """
    if config.drop_unknown_mods:
        unknown_mods = mods_df[
            (mods_df[PTM_NAME_KEY] != 'Oxidation (M)') &
            (mods_df[PTM_NAME_KEY] != 'Carbamidomethylation') &
            (mods_df[PTM_NAME_KEY] != 'Carbamidomethyl (C)')
        ][PTM_ID_KEY].tolist()
        unknown_mods = {str(x) for x in unknown_mods}
        target_df = target_df.with_columns(
            pl.col(PTM_SEQ_KEY).map_elements(
                lambda x : check_bad_mods(x, unknown_mods),
                skip_nulls=False, return_dtype=pl.Boolean,
            ).alias('unknownModifications')
        )
        count_before_drop = target_df.shape[0]
        if config.use_accession_stratum:
            target_df = accession_informed_filter(target_df, 'unknownModifications')
        else:
            target_df = target_df.filter(
                pl.col('unknownModifications').not_()
            ).drop('unknownModifications')

        count_after_drop = target_df.shape[0]
        filtered_psms = count_before_drop - count_after_drop
        print(
            OKCYAN_TEXT +
            f'\tFiltered {filtered_psms} PSMs due to modifications unknown to Prosit.' +
            ENDC_TEXT
        )

        if not mods_df[
                (mods_df[PTM_NAME_KEY] == 'Carbamidomethylation') |
                (mods_df[PTM_NAME_KEY] == 'Carbamidomethyl (C)')
            ].shape[0] and config.filter_c:
            count_before_drop = target_df.shape[0]
            target_df = target_df.filter(
                ~pl.col(PEPTIDE_KEY).str.contains('C')
            )
            count_after_drop = target_df.shape[0]
            filtered_psms = count_before_drop - count_after_drop
            print(
                OKCYAN_TEXT +
                f'\tFiltered {filtered_psms} PSMs due to unmodified cysteines.' +
                ENDC_TEXT
            )

    return target_df

def create_features(config):
    """ Function to create features for percolator/mokapot input.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object used throughout the pipeline.
    """
    target_df, mods_df = generic_read_df(config)
    target_df = target_df.with_row_count(name='tempIndex')

    target_df = process_unknown_modifications(target_df, mods_df, config)

    n_variable_mods = mods_df.shape[0]  # pylint: disable=no-member
    if  n_variable_mods > 9:
        raise ValueError(
            f'inSPIRE supports no more than 9 unique PTMs, found {n_variable_mods}'
        )

    print(
        OKCYAN_TEXT +
        '\tMS Search Results ready.' +
        ENDC_TEXT
    )
    write_rescoring_features(
        target_df,
        mods_df,
        config,
    )
