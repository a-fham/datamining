"""
Bias Metrics
=============
Data-driven, statistically grounded measurements of political bias.

Provides three key metrics per outlet:
  - Partisan Lean Score (PLS): directional bias on a [-1, +1] scale
  - Bias Entropy           : how evenly/unevenly phrases are distributed
  - Phrase Exclusivity Index (PEI): % of an outlet's phrases used *only* by them

And two key metrics per phrase:
  - Log-Odds Ratio  : how much more/less likely a phrase is in Right vs Left outlet
  - Chi-Square test : whether phrase usage is statistically non-random w.r.t. bias
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency


# ─────────────────────────────────────────────────────────────────────────────
# OUTLET-LEVEL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def partisan_lean_score(count_matrix: pd.DataFrame, outlet_bias_labels: dict) -> pd.Series:
    """
    Partisan Lean Score (PLS) for each outlet.

    Strategy: Compare how much each outlet's phrase-usage profile *cosine-aligns*
    with the average "Right" profile vs the average "Left" profile.
    PLS ∈ [-1, 1]:  -1 = fully Left, +1 = fully Right, 0 = Center
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    label_map = {
        'Left': -1.0, 'Center-Left': -0.5,
        'Center': 0.0,
        'Center-Right': 0.5, 'Right': 1.0,
    }

    right_outlets  = [o for o, l in outlet_bias_labels.items() if 'Right' in l and o in count_matrix.columns]
    left_outlets   = [o for o, l in outlet_bias_labels.items() if 'Left'  in l and o in count_matrix.columns]

    if not right_outlets or not left_outlets:
        return pd.Series(dtype=float)

    # Prototype vectors: mean of all known-left / known-right outlet vectors
    left_proto  = count_matrix[left_outlets].mean(axis=1).values.reshape(1, -1)
    right_proto = count_matrix[right_outlets].mean(axis=1).values.reshape(1, -1)

    scores = {}
    for outlet in count_matrix.columns:
        vec = count_matrix[outlet].values.reshape(1, -1)
        sim_right = cosine_similarity(vec, right_proto)[0, 0]
        sim_left  = cosine_similarity(vec, left_proto)[0, 0]
        denom = sim_right + sim_left
        scores[outlet] = (sim_right - sim_left) / denom if denom > 0 else 0.0

    return pd.Series(scores, name='partisan_lean_score').sort_values()


def bias_entropy(count_matrix: pd.DataFrame) -> pd.Series:
    """
    Shannon Entropy of phrase distribution for each outlet.

    High entropy → outlet uses many phrases relatively uniformly.
    Low entropy  → outlet concentrates usage on very few phrases (more extreme).
    """
    entropies = {}
    for outlet in count_matrix.columns:
        counts = count_matrix[outlet].values.astype(float)
        counts = counts[counts > 0]
        if len(counts) == 0:
            entropies[outlet] = 0.0
            continue
        probs = counts / counts.sum()
        entropies[outlet] = float(stats.entropy(probs, base=2))

    return pd.Series(entropies, name='bias_entropy').sort_values(ascending=False)


def phrase_exclusivity_index(count_matrix: pd.DataFrame) -> pd.Series:
    """
    Phrase Exclusivity Index (PEI): fraction of an outlet's active phrases
    that are used by that outlet *only* (unique to them).

    High PEI → outlet favours niche, exclusive language.
    """
    binary = (count_matrix > 0).astype(int)
    outlet_usage_count = binary.sum(axis=1)   # how many outlets use each phrase

    pei = {}
    for outlet in binary.columns:
        active_phrases = binary[binary[outlet] == 1].index
        if len(active_phrases) == 0:
            pei[outlet] = 0.0
            continue
        exclusive = outlet_usage_count.loc[active_phrases] == 1
        pei[outlet] = round(exclusive.sum() / len(active_phrases), 4)

    return pd.Series(pei, name='phrase_exclusivity_index').sort_values(ascending=False)


def outlet_bias_report(count_matrix: pd.DataFrame, outlet_bias_labels: dict) -> pd.DataFrame:
    """
    Combine PLS, Entropy, and PEI into one report DataFrame.
    """
    pls = partisan_lean_score(count_matrix, outlet_bias_labels)
    ent = bias_entropy(count_matrix)
    pei = phrase_exclusivity_index(count_matrix)

    df = pd.DataFrame({
        'partisan_lean_score':     pls,
        'bias_entropy':            ent,
        'phrase_exclusivity_index': pei,
    }).fillna(0)

    df['known_label'] = df.index.map(outlet_bias_labels).fillna('Unknown')
    return df.sort_values('partisan_lean_score')


# ─────────────────────────────────────────────────────────────────────────────
# PHRASE-LEVEL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def log_odds_ratio(count_matrix: pd.DataFrame, outlet_bias_labels: dict,
                   smoothing: float = 0.5) -> pd.DataFrame:
    """
    Log-Odds Ratio (LOR) per phrase: how much more likely a phrase appears in
    Right-leaning vs Left-leaning outlets.

    LOR > 0  → phrase skews Right
    LOR < 0  → phrase skews Left
    LOR ≈ 0  → neutral phrase

    Uses Laplace smoothing to stabilise rare phrase estimates.
    """
    right_outlets = [o for o, l in outlet_bias_labels.items() if 'Right' in l and o in count_matrix.columns]
    left_outlets  = [o for o, l in outlet_bias_labels.items() if 'Left'  in l and o in count_matrix.columns]

    right_counts = count_matrix[right_outlets].sum(axis=1) + smoothing
    left_counts  = count_matrix[left_outlets].sum(axis=1)  + smoothing

    total_right = right_counts.sum()
    total_left  = left_counts.sum()

    right_freq = right_counts / total_right
    left_freq  = left_counts  / total_left

    lor = np.log(right_freq / left_freq)

    return pd.DataFrame({
        'log_odds_ratio': lor,
        'right_count':    count_matrix[right_outlets].sum(axis=1),
        'left_count':     count_matrix[left_outlets].sum(axis=1),
    }).sort_values('log_odds_ratio', ascending=False)


def chi_square_partisan(count_matrix: pd.DataFrame, outlet_bias_labels: dict,
                        p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Run a Chi-Square test for each phrase to see whether its distribution
    across Left vs Right outlets is statistically significant.

    Returns a DataFrame with chi2 statistic, p-value, and a 'significant' flag.
    Only phrases with total count >= 5 are tested (low-count cells violate Chi-Sq).
    """
    right_outlets = [o for o, l in outlet_bias_labels.items() if 'Right' in l and o in count_matrix.columns]
    left_outlets  = [o for o, l in outlet_bias_labels.items() if 'Left'  in l and o in count_matrix.columns]

    results = []
    for phrase, row in count_matrix.iterrows():
        r_count = int(row[right_outlets].sum())
        l_count = int(row[left_outlets].sum())
        total   = r_count + l_count
        if total < 5:
            continue

        # 2×2 contingency: [used in Right, not used in Right] × [used in Left, not used ...]
        n_right_total = int(count_matrix[right_outlets].values.sum())
        n_left_total  = int(count_matrix[left_outlets].values.sum())
        contingency = np.array([
            [r_count,                   l_count],
            [n_right_total - r_count,   n_left_total - l_count],
        ])
        chi2, p, _, _ = chi2_contingency(contingency, correction=True)
        results.append({
            'phrase':      phrase,
            'chi2':        round(chi2, 4),
            'p_value':     round(p, 6),
            'significant': p < p_threshold,
            'right_count': r_count,
            'left_count':  l_count,
        })

    df = pd.DataFrame(results).set_index('phrase')
    return df.sort_values('chi2', ascending=False)
