""" Function for creating sequence logo plots.
"""
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests

from inspire.custom_logo import CustomLogo


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_COLOUR_SCHEME = {
    'P': 'deeppink',
    'M': 'orange',
    'A': 'orange',
    'V': 'orange',
    'I': 'orange',
    'L': 'orange',
    'F': 'orange',
    'Y': 'orange',
    'W': 'orange',
    'H': 'seagreen',
    'R': 'seagreen',
    'K': 'seagreen',
    'D': 'firebrick',
    'E': 'firebrick',
    'N': 'dodgerblue',
    'Q': 'dodgerblue',
    'S': 'dodgerblue',
    'T': 'dodgerblue',
    'G': 'dodgerblue',
    'C': 'dodgerblue',
}


def create_comparison_logo_plot(
    data_frame_list, title_list, peptide_length, output_folder, out_name, data_index=None,
):
    """ Function to compare frequencies of amino acids
    """
    dim_val = len(data_frame_list)-1
    n_plots = (dim_val ** 2 + dim_val)//2
    _, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    for col_idx, stata_df1 in enumerate(data_frame_list[:-1]):
        for row_idx, strata_df2 in enumerate(data_frame_list[col_idx+1:]):
            count_df_1 = get_count_df(stata_df1, peptide_length)
            count_df_2 = get_count_df(strata_df2, peptide_length)

            js_div, entropies = jensenshannon(
                count_df_1,
                count_df_2,
                axis=1,
            )

            entropy_df = pd.DataFrame(entropies)
            entropy_df.columns = list(AMINO_ACIDS)
            entropy_df.index += 1


            entropy_df['jsDivergence'] = js_div
            entropy_df = entropy_df.apply(scale_to_js_divergence, axis=1)
            entropy_df = entropy_df.drop(['jsDivergence'], axis=1)
            p_val_df = get_fishers_exact_pvals(count_df_1, count_df_2, peptide_length)
            signif_dict = get_significance_dict(p_val_df)

            logo_plot, axes[plot_idx] = create_logo_plot(
                entropy_df, axes[plot_idx], peptide_length, signif_dict
            )

            logo_plot.ax.set_title(f'{title_list[col_idx]} vs. {title_list[col_idx+row_idx+1]}')
            max_val = np.abs(js_div).max()
            if max_val < 0.2:
                cut_off = ceil(max_val*20)/20
                logo_plot.ax.set_ylim([-cut_off, cut_off])
                y_ticks = [x for x in [
                    -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2
                ] if abs(x) <= cut_off]
                logo_plot.ax.set_yticks(y_ticks)
            else:
                logo_plot.ax.set_ylim([-ceil(max_val*5)/5, ceil(max_val*5)/5])

            if data_index is not None:
                logo_plot.ax.set_xlim([0.5, peptide_length+0.5])
                logo_plot.ax.set_xticklabels(data_index)
                logo_plot.ax.set_xlabel('Cluster Compared')
            plot_idx += 1

    axes[0].set_ylabel('JS Divergence')

    plt.tight_layout()
    plt.savefig(f'{output_folder}/{out_name}.svg', format='svg')


def get_count_df(strat_df, peptide_length):
    """ Function to get counts of amino acids at each position for peptide of a
        given length.
    """
    aa_counts = np.zeros([peptide_length, len(AMINO_ACIDS)])
    for _, df_row in strat_df.iterrows():
        for pos_idx, amino_acid in enumerate(df_row['peptide']):
            if amino_acid == 'X':
                continue
            aa_counts[pos_idx, AMINO_ACIDS.index(amino_acid)] += 1

    count_df = pd.DataFrame(
        aa_counts,
        columns=list(AMINO_ACIDS)
    )
    count_df.index += 1

    return count_df


def jensenshannon(p, q, base=None, *, axis=0, keepdims=False):
    """ Modified js divergence from scipy.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)
    m = (p + q) / 2.0
    relative_entropy_fn = np.vectorize(_element_wise_rel_entropy)
    left = relative_entropy_fn(p, m)
    right = relative_entropy_fn(q, m)

    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum
    if base is not None:
        js /= np.log(base)

    return (js / 2.0), left-right


def _element_wise_rel_entropy(x, y):
    """ Function to calculate entropy per element.
    """
    if x > 0 and y > 0:
        return x*np.log(x/y)
    if x == 0 and y >= 0:
        return 0.0
    return np.inf


def get_significance_dict(p_val_df):
    """ Helper function to get out the significantce into a dictionary.
    """
    sig_dict = {}
    for _, df_row in p_val_df.iterrows():
        sig_dict[f'{df_row["residue"]}{df_row["position"]}'] = df_row['significant']
    return sig_dict


def scale_to_js_divergence(df_row):
    """ Function to scale positive and negative entropies to JS divergence.
    """
    sum_neg = 0.0
    sum_pos = 0.0
    for a_a in AMINO_ACIDS:
        if df_row[a_a] > 0:
            sum_pos += df_row[a_a]
        elif df_row[a_a] < 0:
            sum_neg += abs(df_row[a_a])

    for a_a in AMINO_ACIDS:
        if df_row[a_a] > 0:
            df_row[a_a] = df_row['jsDivergence']*df_row[a_a]/sum_pos
        elif df_row[a_a] < 0:
            df_row[a_a] = df_row['jsDivergence']*df_row[a_a]/sum_neg
    return df_row


def create_logo_plot(amino_acid_data, axis, peptide_length, custom_alphas=None):
    """ Function to clean style on plots.
    """
    logo_plot = CustomLogo(
        amino_acid_data,
        ax=axis,
        font_name='DejaVu Sans',
        color_scheme=AA_COLOUR_SCHEME,
        flip_below=False,
        vpad=.2,
        width=.8,
        custom_alphas=custom_alphas,
    )
    logo_plot.style_xticks(anchor=1, spacing=1)
    logo_plot.ax.set_xlim([0, peptide_length+1])

    # Hide the right and top spines
    logo_plot.ax.spines.right.set_visible(False)
    logo_plot.ax.spines.top.set_visible(False)

    # Only show ticks on the left and bottom spines
    logo_plot.ax.yaxis.set_ticks_position('left')
    logo_plot.ax.xaxis.set_ticks_position('bottom')

    return logo_plot, axis

def get_fishers_exact_pvals(count_df_1, count_df_2, peptide_length):
    """ Function to get fishers exact p values.
    """
    p_val_list = []

    sums_1 = count_df_1.sum(axis=1)
    sums_2 = count_df_2.sum(axis=1)

    for a_a in AMINO_ACIDS:
        for idx in range(1,peptide_length+1):
            group_1_aa = count_df_1[a_a].iloc[idx-1]
            group_2_aa = count_df_2[a_a].iloc[idx-1]

            group_1_notaa = sums_1.iloc[idx-1] - group_1_aa
            group_2_notaa = sums_2.iloc[idx-1] - group_2_aa

            fishers_array = np.array([
                [group_1_aa, group_2_aa],
                [group_1_notaa, group_2_notaa],
            ])

            _, pvalue = fisher_exact(fishers_array)
            p_val_list.append({
                'position': idx,
                'residue': a_a,
                'count1': group_1_aa,
                'count2': group_2_aa,
                'count1_false': group_1_notaa,
                'count2_false': group_2_notaa,
                'pvalue': pvalue,
            })

    pvalue_df = pd.DataFrame(p_val_list)
    pvalue_df['adjusted_pValue'] = multipletests(pvalue_df['pvalue'], method='fdr_bh')[1]
    pvalue_df['significant'] = pvalue_df['pvalue'].apply(lambda x : 1 if x < 0.05 else 0.5)

    return pvalue_df
