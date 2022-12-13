""" Functions for validating spliced PSMs.
"""
import itertools
import re

from Bio import SeqIO
import numpy as np
import pandas as pd

from inspire.constants import (
    CHARGE_KEY,
    KNOWN_PTM_WEIGHTS,
    MINIMAL_FEATURE_SET,
    PROSIT_IONS_KEY,
    PROTON,
    RESIDUE_WEIGHTS,
    SCAN_KEY,
    SOURCE_KEY,
)
from inspire.input.msp import msp_to_df
from inspire.predict_spectra import predict_spectra
from inspire.spectral_features import (
    calculate_spectral_angle,
    get_coverage,
    get_ion_masses,
    get_matches,
)
from inspire.utils import fetch_scan_data



def get_sa(
        df_row,
        mz_accuracy,
        mz_units,
    ):
    """ Function to extract the ion intensities from the true spectra which match
    """
    sequence = df_row['pcpPeptide']
    seq_len = len(sequence)
    prosit_preds = df_row[PROSIT_IONS_KEY]
    mod_seq = df_row['modified_sequence']


    if '(ox)' not in mod_seq and 'C' not in mod_seq:
        mods = None
    else:
        mods = '0.'
        while mod_seq:
            if len(mod_seq) > 1:
                if mod_seq[1] == '(':
                    mods += '1'
                    mod_seq = mod_seq[5:]
                    continue
            if mod_seq[0] != 'C':
                mods += '0'
            else:
                mods += '2'
            mod_seq = mod_seq[1:]
        mods += '.0'
    potential_ion_mzs, precursor_weight = get_ion_masses(
        sequence,
        {
            0: 0.0,
            1: KNOWN_PTM_WEIGHTS['Oxidation (M)'],
            2: KNOWN_PTM_WEIGHTS['Carbamidomethylation'],
        },
        modifications=mods
    )

    precursor_mz = (precursor_weight + (PROTON*df_row[CHARGE_KEY]))/df_row[CHARGE_KEY]

    (
        match_info, _, __,
        ___, ____
    ) = get_matches(
        potential_ion_mzs,
        prosit_preds,
        df_row['mzs'],
        precursor_mz,
        df_row['intensities'],
        mz_accuracy,
        df_row['charge'],
        mz_units,
    )

    matched_intensities = match_info['matched_intensities']
    ordered_prosit_intes = match_info['ordered_prosit_intes']
    ordered_prosit_ions = match_info['ordered_prosit_ions']

    matched_l2_norm = np.linalg.norm(matched_intensities, ord=2)

    if matched_l2_norm:
        normed_matched_intensities = matched_intensities/matched_l2_norm
    else:
        normed_matched_intensities = matched_intensities

    df_row['spectralAngle'] = calculate_spectral_angle(
        normed_matched_intensities,
        ordered_prosit_intes,
    )
    ordered_matched_ions = ordered_prosit_ions[matched_intensities > 0.0]
    df_row['matchedCoverage'], _, __ = get_coverage(
        seq_len, ordered_matched_ions, per_letter=True
    )
    return df_row



def get_cleavage_peptides(proteome_loc):
    """ Function to find all nonspliced peptides.
    """
    with open(proteome_loc, 'r', encoding='UTF-8') as fasta_file:
        prot_sequences = list(SeqIO.parse(fasta_file, 'fasta'))

    weights = []
    pcp_seqs = []
    for protein in prot_sequences:
        prot_seq = str(protein.seq)
        for pep_len in range(7, 31):
            for i in range(1 + len(prot_seq) - pep_len):
                pep = prot_seq[i:i+pep_len]
                pcp_seqs.append(pep)
                weights.append(
                    round(sum([RESIDUE_WEIGHTS[z] for z in pep]), 2)
                )

    return pd.DataFrame({
        'pcpPeptide': pd.Series(pcp_seqs),
        'mw': pd.Series(weights),
    })

def get_mod_matched_seqs(mod_seq, pcp_pep):
    """ Funciton to find all matched sequences including oxidation of methionine.
    """
    n_mod_ox = mod_seq.count('[+16.0]')
    if n_mod_ox == 0:
        return [pcp_pep]
    n_methionine = pcp_pep.count('M')
    if n_methionine < n_mod_ox:
        return []
    methionine_pos = [
        m.start() for m in re.finditer('M', pcp_pep)
    ]
    possible_oxidations = itertools.combinations(methionine_pos, n_mod_ox)

    possible_mod_seqs = []
    for possible_os in possible_oxidations:
        mod_str = pcp_pep
        for o_site in sorted(possible_os, reverse=True):
            mod_str = mod_str[:o_site+1] + '[+16.0]' + mod_str[o_site+1:]
        possible_mod_seqs.append(mod_str)
    return possible_mod_seqs

def find_competitors(config):
    """ Function to find isobaric nonspliced competitors to spliced peptides.
    """
    final_df = pd.read_csv(
        f'{config.output_folder}/pre_finalAssignments.csv'
    )

    final_df = final_df[
        (final_df['qValue'] < 0.01) &
        (final_df['accessionGroup'] == 'spliced')
    ]

    final_df['mw'] = final_df['peptide'].apply(
        lambda x : round(sum([RESIDUE_WEIGHTS[z] for z in x]), 2)
    )

    pcp_df = get_cleavage_peptides(config.proteome)
    pcp_df = pcp_df.drop_duplicates()
    pcp_df['label'] = 1


    merged_df = pd.merge(
        final_df[[
            'source',
            'scan',
            'peptide',
            'modifiedSequence',
            'charge',
            'retentionTime',
            'proteins',
            'mw'
        ]],
        pcp_df[['pcpPeptide', 'mw', 'label']],
        how='inner',
        on='mw'
    )
    if merged_df.shape[0] == 0:
        return 0
    merged_df['matched_pcps'] = merged_df.apply(
        lambda x : get_mod_matched_seqs(x['modifiedSequence'], x['pcpPeptide']),
        axis=1,
    )
    merged_df = merged_df.explode('matched_pcps')
    merged_df = merged_df[merged_df['matched_pcps'].apply(lambda x : isinstance(x, str))]
    merged_df = merged_df.sort_values('peptide')
    merged_df['modified_sequence'] = merged_df['matched_pcps'].apply(
        lambda x : x.replace('[+16.0]', '(ox)').replace('[+57.0]', '')
    )
    merged_df.to_csv(
        f'{config.output_folder}/cleavageCompetitors.csv',
        index=False
    )
    merged_df['collision_energy'] = config.collision_energy
    merged_df = merged_df.rename(columns={'charge': 'precursor_charge'})
    prosit_input_df = merged_df[[
        'modified_sequence', 'precursor_charge', 'collision_energy'
    ]].drop_duplicates()
    prosit_input_df.to_csv(
        f'{config.output_folder}/validationInput.csv', index=False
    )
    predict_spectra(config, 'validation')
    return merged_df.shape[0]

def calculate_competitor_spectral_data(competitors_df, config):
    """ Function to calculate spectral angles for isobaric nonspliced competitors.
    """
    scan_df = fetch_scan_data(competitors_df, config, with_charge=False)
    competitors_df = pd.merge(
        competitors_df,
        scan_df,
        how='inner',
        on=[SOURCE_KEY, SCAN_KEY]
    )

    prosit_df = msp_to_df(
        f'{config.output_folder}/validationPredictions.msp', 'prosit', None
    )
    prosit_df = prosit_df.drop_duplicates(subset=['modified_sequence', 'charge'])
    competitors_df['modified_sequence'] = competitors_df['matched_pcps'].apply(
        lambda x : x.replace('[+16.0]', '(ox)').replace('[+57.0]', '')
    )
    competitors_df = pd.merge(
        competitors_df,
        prosit_df,
        how='inner',
        on=['modified_sequence', CHARGE_KEY]
    )

    competitors_df['peptide'] = competitors_df['pcpPeptide']
    competitors_df = competitors_df.apply(
        lambda x : get_sa(x, config.mz_accuracy, config.mz_units),
        axis=1
    )
    return competitors_df

def validate_spliced(config):
    """ Funciton to validate spliced PSMs against isobaric nonspliced competitors.

    Parameters
    ----------
    config : inspire.config.Config
        The Config object for the experiment.
    """
    n_competitors = find_competitors(config)

    if not n_competitors:
        final_df = pd.read_csv(
            f'{config.output_folder}/pre_finalAssignments.csv'
        )

        final_df = final_df[
            (final_df['spectralAngle'] > 0.7) &
            (final_df['deltaRT'] < 150) &
            (final_df['qValue'] < 0.01)
        ]

        final_df.to_csv(
            f'{config.output_folder}/finalAssignments.csv',
            index=False,
        )
        return

    competitors_df = pd.read_csv(f'{config.output_folder}/cleavageCompetitors.csv')
    competitors_df = calculate_competitor_spectral_data(competitors_df, config)

    if config.combined_scans_file is None:
        rt_ints = {}
        rt_coefs = {}
        for raw_file in competitors_df[SOURCE_KEY].unique().tolist():
            rt_df = pd.read_csv(
                f'{config.output_folder}/rt_fit_{raw_file}.csv'
            )
            rt_ints[raw_file] = rt_df['intercepts'].mean()
            rt_coefs[raw_file] = rt_df['coefficents'].mean()

        competitors_df['deltaRT'] = competitors_df.apply(
            lambda x : abs(
                x['retentionTime'] - (rt_ints[x['source']] + (rt_coefs[x['source']]*x['iRT']))
            ),
            axis=1,
        )
    else:
        rt_df = pd.read_csv(
            f'{config.output_folder}/rt_fit_{config.combined_scans_file[:-4]}.csv'
        )
        rt_int = rt_df['intercepts'].mean()
        rt_coef = rt_df['coefficents'].mean()
        competitors_df['deltaRT'] = competitors_df.apply(
            lambda x : abs(x['retentionTime'] - (rt_int + (rt_coef*x['iRT']))),
            axis=1
        )

    competitors_df['accession_spliced'] = 0
    competitors_df['accession_nonspliced'] = 1

    weights_df = pd.read_csv(
        f'{config.output_folder}/final.percolatorSeparate.weights.csv',
        skiprows=lambda x : x not in [0, 2, 5, 8],
        sep='\t',
    )

    feature_set = MINIMAL_FEATURE_SET + ['accession_nonspliced', 'accession_spliced']
    no_rt_feats = [
        a for a in feature_set if a != 'deltaRT'
    ]
    if 'deltaRT' in weights_df.columns:
        scores = np.matmul(
            competitors_df[feature_set].values,
            weights_df[feature_set].transpose().values
        )
    else:
        scores = np.matmul(
            competitors_df[no_rt_feats].values,
            weights_df[no_rt_feats].transpose().values
        )


    bias_terms = weights_df['m0'].values
    scores += bias_terms.transpose()

    final_scores = np.average(scores, axis=1)

    competitors_df['compScore'] = final_scores
    competitors_df = competitors_df.sort_values(by='compScore', ascending=False)
    competitors_df = competitors_df.drop_duplicates(subset=[SOURCE_KEY, SCAN_KEY])

    competitors_df = competitors_df.reset_index(drop=True)
    competitors_df['group'] = competitors_df.index // 10
    competitors_df['index'] = competitors_df.index % 10

    final_df = pd.read_csv(
        f'{config.output_folder}/pre_finalAssignments.csv'
    )


    final_df = pd.merge(
        final_df,
        competitors_df[[SOURCE_KEY, SCAN_KEY, 'compScore', 'group', 'index']],
        how='left',
        on=[SOURCE_KEY, SCAN_KEY]
    )
    final_df['accession_nonspliced'] = final_df['accessionGroup'].apply(
        lambda x : 1 if x == 'nonspliced' else 0
    )
    final_df['accession_spliced'] = final_df['accessionGroup'].apply(
        lambda x : 1 if x == 'spliced' else 0
    )
    if 'deltaRT' in weights_df.columns:
        re_scores = np.matmul(
            final_df[feature_set].values,
            weights_df[feature_set].transpose().values
        )
    else:
        re_scores = np.matmul(
            final_df[no_rt_feats].values,
            weights_df[no_rt_feats].transpose().values
        )
    re_scores += bias_terms
    final_df['reScore1'] = re_scores[:,0]
    final_df['reScore2'] = re_scores[:,0]
    final_df['reScore3'] = re_scores[:,1]
    final_df['reScore'] = np.average(re_scores, axis=1)

    final_df['compScore']  = final_df['compScore'].fillna(-100_000)

    final_df = final_df[final_df.apply(
        lambda x : x['compScore'] is None or x['reScore'] > x['compScore'],
        axis=1,
    )]

    final_df = final_df.drop(['compScore'], axis=1)
    final_df = final_df.drop(
        [
            'group',
            'index',
            'accession_nonspliced',
            'accession_spliced',
            'reScore1',
            'reScore2',
            'reScore3',
            'reScore',
        ],
        axis=1,
    )

    final_df = final_df[
        (final_df['spectralAngle'] > 0.7) &
        (final_df['deltaRT'] < 150) &
        (final_df['qValue'] < 0.01)
    ]

    final_df.to_csv(
        f'{config.output_folder}/finalAssignments.csv',
        index=False,
    )
