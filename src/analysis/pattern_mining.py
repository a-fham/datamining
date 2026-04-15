"""
Pattern Mining
==============
Core data mining algorithms:
  - Normalisation
  - PCA / t-SNE dimensionality reduction
  - K-Means & Agglomerative clustering (with automatic k selection via silhouette)
  - FP-Growth / Apriori association rules
  - Data-driven bias pattern detection using Log-Odds (replaces old hardcoded logic)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def normalize_phrase_counts(count_matrix: pd.DataFrame) -> pd.DataFrame:
    """Min-Max normalise each phrase row so outlet vectors are comparable."""
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(count_matrix.T).T
    return pd.DataFrame(normalized, index=count_matrix.index, columns=count_matrix.columns)


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

def perform_pca(count_matrix: pd.DataFrame, n_components: int = 2):
    """
    PCA on the outlet × phrase space.
    Returns (result_df, pca_model).
    """
    pca = PCA(n_components=n_components, random_state=42)
    outlet_vectors = StandardScaler().fit_transform(count_matrix.T.values)
    transformed = pca.fit_transform(outlet_vectors)
    result = pd.DataFrame(
        transformed,
        index=count_matrix.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    return result, pca


def perform_tsne(count_matrix: pd.DataFrame, perplexity: int = 10, n_iter: int = 1000) -> pd.DataFrame:
    """t-SNE non-linear dimensionality reduction on outlets."""
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    outlet_vectors = StandardScaler().fit_transform(count_matrix.T.values)
    transformed = tsne.fit_transform(outlet_vectors)
    return pd.DataFrame(transformed, index=count_matrix.columns, columns=['t-SNE1', 't-SNE2'])


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def best_k_kmeans(outlet_vectors: np.ndarray, k_range=range(2, 8)) -> int:
    """
    Automatically select the best k for K-Means using silhouette score.
    Returns the k that maximises the average silhouette coefficient.
    """
    best_k, best_score = 2, -1
    for k in k_range:
        if k >= outlet_vectors.shape[0]:
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(outlet_vectors)
        score = silhouette_score(outlet_vectors, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_outlets(count_matrix: pd.DataFrame, n_clusters: int = None,
                    method: str = 'kmeans') -> pd.Series:
    """
    Cluster outlets by their phrase-usage fingerprints.

    If n_clusters is None, the optimal k is chosen automatically via silhouette
    analysis (K-Means only).  Agglomerative defaults to 4.
    """
    outlet_vectors = StandardScaler().fit_transform(count_matrix.T.values)

    if method == 'kmeans':
        k = n_clusters if n_clusters else best_k_kmeans(outlet_vectors)
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
    else:
        k = n_clusters if n_clusters else 4
        clusterer = AgglomerativeClustering(n_clusters=k)

    labels = clusterer.fit_predict(outlet_vectors)
    return pd.Series(labels, index=count_matrix.columns, name='cluster')


def linkage_matrix(count_matrix: pd.DataFrame) -> np.ndarray:
    """Return a scipy linkage matrix for dendrogram plotting."""
    outlet_vectors = StandardScaler().fit_transform(count_matrix.T.values)
    return linkage(outlet_vectors, method='ward')


# ─────────────────────────────────────────────────────────────────────────────
# ASSOCIATION RULE MINING
# ─────────────────────────────────────────────────────────────────────────────

def outlet_jaccard_similarity(count_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full Jaccard Similarity matrix between all outlet pairs.

    Jaccard(A, B) = |phrases used by BOTH A and B| / |phrases used by A OR B|

    - Score of 1.0 = identical phrase fingerprints
    - Score of 0.0 = no shared phrases at all

    This replaces FP-Growth: it captures the same "which outlets
    cluster together by shared language" insight but runs in O(outlets²)
    time using fast vectorised operations.
    """
    binary = (count_matrix > 0).astype(int)  # phrase × outlet
    outlets = binary.columns.tolist()

    # Vectorised: dot product = intersection sizes, broadcasting = union sizes
    B = binary.values.T.astype(np.float32)           # outlet × phrase
    intersection = B @ B.T                            # outlet × outlet
    col_sums = B.sum(axis=1, keepdims=True)           # outlet totals
    union = col_sums + col_sums.T - intersection      # inclusion-exclusion
    union = np.where(union == 0, 1, union)            # avoid /0
    jaccard = intersection / union

    return pd.DataFrame(jaccard, index=outlets, columns=outlets)


def top_similar_outlet_pairs(jaccard_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Extract the top_n most similar outlet pairs from the Jaccard matrix,
    excluding self-pairs (diagonal).

    Returns a DataFrame with columns:
        ['outlet_a', 'outlet_b', 'jaccard_similarity']
    sorted descending by similarity.
    """
    outlets = jaccard_df.index.tolist()
    pairs = []
    for i, a in enumerate(outlets):
        for j, b in enumerate(outlets):
            if j <= i:       # upper triangle only, skip diagonal
                continue
            pairs.append({
                'outlet_a':           a,
                'outlet_b':           b,
                'jaccard_similarity': round(float(jaccard_df.loc[a, b]), 4),
            })

    df = pd.DataFrame(pairs).sort_values('jaccard_similarity', ascending=False)
    return df.head(top_n).reset_index(drop=True)


def mine_association_rules(count_matrix: pd.DataFrame,
                           min_support: float = 0.25,
                           method: str = 'jaccard',
                           top_n_phrases: int = 200) -> tuple:
    """
    Thin wrapper kept for API compatibility.
    Delegates to outlet_jaccard_similarity + top_similar_outlet_pairs.
    Returns (jaccard_df, pairs_df) — same calling convention as before.
    """
    jaccard_df = outlet_jaccard_similarity(count_matrix)
    pairs_df   = top_similar_outlet_pairs(jaccard_df)
    return jaccard_df, pairs_df


# ─────────────────────────────────────────────────────────────────────────────
# BIAS PATTERN DETECTION  (data-driven, replaces hardcoded outlet lists)
# ─────────────────────────────────────────────────────────────────────────────

def detect_bias_patterns(count_matrix: pd.DataFrame, outlet_bias_labels: dict,
                         top_n: int = 50) -> pd.DataFrame:
    """
    Identify the most partisan phrases using a Log-Odds Ratio approach.

    Unlike the old hardcoded version, this is entirely data-driven:
      1. Sum phrase counts across known-Right and known-Left outlets.
      2. Compute LOR with Laplace smoothing.
      3. Return the top_n most Right-skewed and top_n most Left-skewed phrases.

    Returns a sorted DataFrame with columns:
        ['log_odds_ratio', 'right_count', 'left_count', 'direction']
    """
    from src.analysis.bias_metrics import log_odds_ratio

    lor_df = log_odds_ratio(count_matrix, outlet_bias_labels)

    # Tag direction
    lor_df['direction'] = lor_df['log_odds_ratio'].apply(
        lambda x: 'Right' if x > 0 else ('Left' if x < 0 else 'Neutral')
    )

    top_right = lor_df.nlargest(top_n, 'log_odds_ratio')
    top_left  = lor_df.nsmallest(top_n, 'log_odds_ratio')

    return pd.concat([top_right, top_left]).drop_duplicates()


def keyword_distinctiveness(count_matrix: pd.DataFrame, outlet_bias_labels: dict,
                             top_n: int = 20) -> dict:
    """
    Per-topic: return the most distinctive keywords for Right vs Left outlets,
    ranked by Log-Odds Ratio magnitude.

    Returns:  { 'Right': [(phrase, lor), ...], 'Left': [(phrase, lor), ...] }
    """
    from src.analysis.bias_metrics import log_odds_ratio

    lor_df = log_odds_ratio(count_matrix, outlet_bias_labels)

    return {
        'Right': list(
            lor_df.nlargest(top_n, 'log_odds_ratio')['log_odds_ratio'].items()
        ),
        'Left': list(
            lor_df.nsmallest(top_n, 'log_odds_ratio')['log_odds_ratio'].items()
        ),
    }
