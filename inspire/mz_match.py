""" Script of functions for matching molecular weights to possible fragment ions.
"""
import numpy as np

from inspire.constants import (
    C_TERMINUS,
    N_TERMINUS,
    PROTON,
    RESIDUE_WEIGHTS,
    ION_OFFSET,
)

def compute_potential_mws(sequence, modifications, reverse, ptm_id_weights):
    """ Function to compute the molecular weights of potential fragments
        generated from a peptide (y & b ions, charges 1,2, or 3, and H2O
        or O2 losses).

    Parameters
    ----------
    sequence : str
        The peptide sequence for which we require molecular weights.
    modifications : str
        A string of the ptms for the sequence which will alter
        the potential mzs.
    reverse : bool
        Whether we are getting fragment mzs in the forward direction
        (eg for b ions), or backward direction (eg. for y ions).
    ptm_id_weights : dict
        Mapping of ptm ids to their molecular weights.

    Returns
    -------
    mzs : np.array of floats
        An array of all the possible mzs that coule be observed in
        the MS2 spectrum of a sequence.
    """
    sequence_length = len(sequence)
    n_fragments = sequence_length - 1
    mzs = np.empty(n_fragments)

    if (
        modifications and
        isinstance(modifications, str) and
        modifications != 'nan'
        and modifications != 'None'
    ):
        ptms_list = modifications.split(".")
        mods_list = [int(mod) for mod in ptms_list[1]]

        if reverse:
            ptm_start = int(ptms_list[2])
            ptm_end = int(ptms_list[0])
            mods_list = mods_list[::-1]
        else:
            ptm_start = int(ptms_list[0])
            ptm_end = int(ptms_list[2])
    else:
        mods_list = None
        ptm_start = 0
        ptm_end = 0

    if reverse:
        sequence = sequence[::-1]

    tracking_mw = ptm_id_weights[ptm_start]

    for idx in range(n_fragments):
        tracking_mw += RESIDUE_WEIGHTS[sequence[idx]]
        if mods_list is not None and mods_list[idx]:
            tracking_mw += ptm_id_weights[mods_list[idx]]

        mzs[idx] = tracking_mw

    tracking_mw += RESIDUE_WEIGHTS[sequence[n_fragments]]
    if mods_list is not None and mods_list[n_fragments]:
        tracking_mw += ptm_id_weights[mods_list[n_fragments]]
    if ptm_end:
        tracking_mw += ptm_id_weights[ptm_end]

    return mzs, tracking_mw


def get_ion_masses(sequence, ptm_id_weights, modifications=None):
    """ Function to the get mz's for all the ions predicted by prosit.

    Parameters
    ----------
    sequence : str
        The peptide sequence for which we require molecular weights.
    modifications : str
        A string of the ptms for the sequence which will alter
        the potential mzs.

    Returns
    -------
    all_possible_ions : dict
        A dictionary of all the mzs of all possible b and y ions
        that could be produced.
    """
    sub_seq_mass, total_residue_mass = compute_potential_mws(
        sequence=sequence,
        modifications=modifications,
        reverse=False,
        ptm_id_weights=ptm_id_weights
    )

    rev_sub_seq_mass, _ = compute_potential_mws(
        sequence=sequence,
        modifications=modifications,
        reverse=True,
        ptm_id_weights=ptm_id_weights
    )

    # b- and y-ions for each of the charge options
    prosit_ions = {}
    prosit_ions['b'] = ION_OFFSET['b'] + sub_seq_mass
    prosit_ions['y'] = ION_OFFSET['y'] + rev_sub_seq_mass

    return prosit_ions, (total_residue_mass + C_TERMINUS + N_TERMINUS)


def match_mz(base_mass, frag_z, observed_mzs, loss=0.0):
    """ Function to match a fragment m/z to the nearest experimental m/z.

    Parameters
    ----------
    base_mass : float
        The mass of the b or y ion of that fragment index.
    frag_z : int
        The charge of the fragment ion.
    observed_mzs : np.array of float
        The experimentally observed fragment mzs.
    loss : float (default=0.0)
        The neutral loss weight to be applied.

    Returns
    -------
    mz_error : float
        The mz error on the nearest observed fragment ion.
    matched_mz_ind : int
        The index of the nearest observed fragment ion.
    """
    fragment_mz = (
        (base_mass + (frag_z * PROTON)) - loss
    )/frag_z
    matched_mz_ind = np.argmin(
        np.abs(observed_mzs - fragment_mz)
    )
    return observed_mzs[matched_mz_ind] - fragment_mz, matched_mz_ind
