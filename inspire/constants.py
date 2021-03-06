""" Relevant constants used through the inspire pipeline.
"""

# Constants for msp read in.
PROSIT_IONS_KEY = 'prositIons'
OXIDATION_PREFIX = 'Oxidation@M'
OXIDATION_PREFIX_LEN = len(OXIDATION_PREFIX)

# Prosit DataFrame column names.
PROSIT_SEQ_KEY = 'modified_sequence'
PROSIT_CHARGE_KEY = 'precursor_charge'
PROSIT_COLLISION_ENERGY_KEY = 'collision_energy'

HEADER_TEXT = '\033[95m'
OKBLUE_TEXT = '\033[94m'
OKCYAN_TEXT = '\033[96m'
OKGREEN_TEXT = '\033[92m'
WARNING_TEXT = '\033[93m'
FAIL_TEXT = '\033[91m'
ENDC_TEXT = '\033[0m'
BOLD_TEXT = '\033[1m'
UNDERLINE_TEXT = '\033[4m'

CHARGE_KEY = 'charge'
PEPTIDE_KEY = 'peptide'
ACCESSION_KEY = 'proteins'
INTENSITIES_KEY = 'intensities'
MZS_KEY = 'mzs'
PTM_SEQ_KEY = 'ptm_seq'
SOURCE_KEY = 'source'
SCAN_KEY = 'scan'
LABEL_KEY = 'label'
ENGINE_SCORE_KEY = 'engineScore'
SPECTRAL_ANGLE_KEY = 'spectralAngle'
MASS_DIFF_KEY = 'massDiff'
RT_KEY = 'retentionTime'
DELTA_SCORE_KEY = 'deltaScore'
SEQ_LEN_KEY = 'sequenceLength'
ACCESSION_STRATUM_KEY = 'accessionGroup'
SOURCE_INDEX_KEY = 'sourceIndex'

SPEARMAN_KEY = 'spearmanR'
PEARSON_KEY = 'pearsonR'
MAE_KEY = 'medianAbsoluteError'

PTM_NAME_KEY = 'Name'
PTM_ID_KEY = 'Identifier'
PTM_IS_VAR_KEY = 'isVar'
PTM_WEIGHT_KEY = 'Delta'

# Spectral Features:
FRAG_MZ_ERR_MED_KEY = 'medianFragmentMzError'
FRAG_MZ_ERR_VAR_KEY = 'fragmentMzErrorVariance'

MATCHED_IONS_KEY = 'nMatchedIonsDivFrags'
MATCHABLE_IONS_KEY = 'nPrositIonsDivFrags'
OBS_NOT_PRED_KEY = 'nFoundNotPredDivFrags'
NOT_ASSIGNED_KEY = 'nNotMatchableDivFrags'
LOSS_IONS_KEY = 'nLossIonsDivFrags'

MATCHED_INTE_KEY = 'meanMatchedInte'
NON_PRED_INTE_KEY = 'meanPredZeroInte'
MEAN_NOT_MATCHABLE_INTE_KEY = 'meanNotMatchableInte'
LOSS_INTE_KEY = 'meanLossInte'
NOT_MATCHED_INTE_KEY = 'meanNotMatchedInte'

PRECURSOR_INTE_KEY = 'precursorIntensity'

# Percolator column names.
PERC_SCAN_ID = 'scannr'
PSM_ID_KEY = {
    'mokapot': 'specID',
    'percolator': 'PSMId',
}
OUT_ACCESSION_KEY = {
    'mokapot': 'proteins',
    'percolator': 'proteinIds',
}
OUT_SCORE_KEY = {
    'mokapot': 'mokapot score',
    'percolator': 'score',
}
OUT_Q_KEY = {
    'mokapot': 'mokapot q-value',
    'percolator': 'q-value',
}

PREFIX_KEYS = {
    'mokapot': [PSM_ID_KEY['mokapot'], LABEL_KEY, PERC_SCAN_ID],
    'percolator': [PSM_ID_KEY['percolator'], LABEL_KEY, PERC_SCAN_ID],
}
SUFFIX_KEYS = [PEPTIDE_KEY, ACCESSION_KEY]

FINAL_SCORE_KEY = 'percolatorScore'
FINAL_Q_VALUE_KEY = 'qValue'

PRED_ACCESSION_KEY = 'predictedAccession'
PRED_PEPTIDE_KEY = 'predictedPeptide'
TRUE_ACCESSION_KEY = 'trueAccession'
TRUE_PEPTIDE_KEY = 'truePeptide'

BASIC_FEATURES = [
    MASS_DIFF_KEY,
    SEQ_LEN_KEY,
    ENGINE_SCORE_KEY,
    DELTA_SCORE_KEY,
    CHARGE_KEY,
    'nVarMods',
    'missedCleavages',
    'avgResidueMass',
    'nRepeatedResidues',
    'fracUnique',
    'fracIL',
    'fracKR',
    'fracC',
    'fromChimera',
    'retentionTime',
    'seqLenMeanDiff',
    'massMeanDiff',
    'absMassDiff',
]

PROTON = 1.007276466622
ELECTRON = 0.00054858
H = 1.007825035
C = 12.0
O = 15.99491463
N = 14.003074

N_TERMINUS = H
C_TERMINUS = O + H
CO = C + O
CHO = C + H + O
NH2 = N + (H * 2)
H2O = (H * 2) + O
NH3 = N + (H * 3)
LOSS_WEIGHTS = {
    '': 0,
    'NH3': NH3,
    'H2O': H2O
}
NEUTRAL_LOSSES = [
    NH3,
    H2O,
]

RESIDUE_WEIGHTS = {
    'A': 71.037114,
    'R': 156.101111,
    'N': 114.042927,
    'D': 115.026943,
    'C': 103.009185,# + 57.021464,
    'E': 129.042593,
    'Q': 128.058578,
    'G': 57.021464,
    'H': 137.058912,
    'I': 113.084064,
    'L': 113.084064,
    'K': 128.094963,
    'M': 131.040485,
    'F': 147.068414,
    'P': 97.052764,
    'S': 87.032028,
    'T': 101.047679,
    'W': 186.079313,
    'Y': 163.06332,
    'V': 99.068414,
}

KNOWN_PTM_WEIGHTS = {
    'Deamidated (N)': 0.984016,
    'Deamidated (NQ)': 0.984016,
    'Deamidation (NQ)': 0.984016,
    'Deamidation (N)': 0.984016,
    'Oxidation (M)': 15.994915,
    'Acetyl (N-term)': 42.010565,
    'Acetylation (N-term)': 42.010565,
    'Acetyl (Protein N-term)': 42.010565,
    'Phospho (Y)': 79.966331,
    'Phospho (ST)': 79.966331,
    'Phospho (STY)': 79.966331,
    'Phosphorylation (STY)': 79.966331,
    'Carbamidomethyl (C)': 57.021464,
    'Carbamidomethylation': 57.021464,
}

KNOWN_PTM_LOC = {
    'Deamidated (N)': 'N',
    'Deamidated (NQ)': 'NQ',
    'Deamidation (NQ)': 'NQ',
    'Deamidation (N)': 'N',
    'Oxidation (M)': 'M',
    'Acetyl (N-term)': 'N-term',
    'Phospho (Y)': 'Y',
    'Phospho (ST)': 'ST',
    'Phospho (STY)': 'STY',
    'Phosphorylation (STY)': 'STY',
    'Carbamidomethyl (C)': 'C',
    'Carbamidomethylation': 'C',
}

ION_OFFSET = {
    'a': N_TERMINUS - CHO,
    'b': N_TERMINUS - H,
    'c': N_TERMINUS + NH2,
    'x': C_TERMINUS + CO - H,
    'y': C_TERMINUS + H,
    'z': C_TERMINUS - NH2,
}

DELTA_PRO_FEATURES = [
    'cTermDist',
    'nTermDist',
    'trueSpectralAngle',
    'inteAtLoc',
    'inteLeft',
    'inteRight',
    MASS_DIFF_KEY,
    'hydroDiff',
    'pkaDiff',
    'polaDiff',
    'BLOSUM6.1',
]


# Constants used by deltapro predictor.

RESIDUE_PROPERTIES = {
    'A': {
        'polarity': 0,
        'hydrophobicity': 1.8,
        'pka': 2.35,
    },
    'C': {
        'polarity': 0,
        'hydrophobicity': 2.5,
        'pka': 1.92,
    },
    'D': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 1.99,
    },
    'E': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 2.10,
    },
    'F': {
        'polarity': 0.0,
        'hydrophobicity': -2.8,
        'pka': 2.20,
    },
    'G': {
        'polarity': 0.0,
        'hydrophobicity': -0.4,
        'pka': 2.35,
    },
    'H': {
        'polarity': 1.0,
        'hydrophobicity': -3.2,
        'pka': 1.80,
    },
    'I': {
        'polarity': 0.0,
        'hydrophobicity': 4.5,
        'pka': 2.32,
    },
    'K': {
        'polarity': 1.0,
        'hydrophobicity': -3.9,
        'pka': 2.16,
    },
    'L': {
        'polarity': 0.0,
        'hydrophobicity': 3.8,
        'pka': 2.33,
    },
    'M': {
        'polarity': 0.0,
        'hydrophobicity': 1.9,
        'pka': 2.13,
    },
    'N': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 2.14,
    },
    'P': {
        'polarity': 0.0,
        'hydrophobicity': -1.6,
        'pka': 1.95,
    },
    'Q': {
        'polarity': 1.0,
        'hydrophobicity': -3.5,
        'pka': 2.17,
    },
    'R': {
        'polarity': 1.0,
        'hydrophobicity': -4.5,
        'pka': 1.82,
    },
    'S': {
        'polarity': 1.0,
        'hydrophobicity': -0.8,
        'pka': 2.19,
    },
    'T': {
        'polarity': 1.0,
        'hydrophobicity': -0.7,
        'pka': 2.09,
    },
    'V': {
        'polarity': 0.0,
        'hydrophobicity': 4.2,
        'pka': 2.29,
    },
    'W': {
        'polarity': 0.0,
        'hydrophobicity': -0.9,
        'pka': 2.46,
    },
    'Y': {
        'polarity': 1.0,
        'hydrophobicity': -1.3,
        'pka': 2.20,
    },
}

BLOSUM6_1_VALUES = {
    'A': 0.19,
    'C': -1.05,
    'D': 0.01,
    'E': -0.08,
    'F': 0.29,
    'G': 1.19,
    'H': -0.79,
    'I': 0.28,
    'K': 0.1,
    'L': 0.34,
    'M': 0.37,
    'N': 0.83,
    'P': -2.02,
    'Q': -0.08,
    'R': 0.2,
    'S': 0.54,
    'T': 0.38,
    'V': 0.16,
    'W': 0.24,
    'Y': -0.48
}
