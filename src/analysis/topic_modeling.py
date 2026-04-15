"""
LDA Topic Modeling
==================
Applies Latent Dirichlet Allocation to the PhraseBias dataset
to discover hidden latent topics across the 28 political topics,
and identifies which discovered topics are Left-heavy vs Right-heavy.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def build_phrase_corpus(combined_phrases: pd.DataFrame) -> pd.Series:
    """
    Treat all phrases per named topic as one 'document'.
    Returns a Series where index = topic name, values = space-joined phrases.
    """
    return (
        combined_phrases
        .groupby('topic')['PHRASE']
        .apply(lambda phrases: ' '.join(phrases.astype(str).str.lower()))
    )


def fit_lda(
    corpus: pd.Series,
    n_components: int = 8,
    random_state: int = 42,
    max_features: int = 500,
) -> tuple:
    """
    Fit LDA on the phrase corpus.

    Returns
    -------
    (lda_model, vectorizer, doc_topic_matrix, feature_names)
      doc_topic_matrix: shape [n_topics, n_components] — topic → latent topic
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        stop_words='english',
    )
    X = vectorizer.fit_transform(corpus.values)

    lda = LatentDirichletAllocation(
        n_components=n_components,
        random_state=random_state,
        max_iter=30,
        learning_method='online',
        n_jobs=-1,
    )
    doc_topic_matrix = lda.fit_transform(X)

    return lda, vectorizer, doc_topic_matrix, vectorizer.get_feature_names_out()


def top_words_per_topic(lda, feature_names, n_top: int = 10) -> dict:
    """Return the top n words for each discovered latent topic."""
    result = {}
    for idx, component in enumerate(lda.components_):
        top_idx = component.argsort()[::-1][:n_top]
        result[f'Latent Topic {idx+1}'] = [(feature_names[i], round(float(component[i]), 4)) for i in top_idx]
    return result


def topic_bias_profile(
    doc_topic_matrix: np.ndarray,
    corpus: pd.Series,
    outlet_topic_counts: pd.DataFrame,
    outlet_bias_labels: dict,
) -> pd.DataFrame:
    """
    For each latent topic, compute the weighted average political lean
    based on outlet-level usage of the named topics.

    Returns a DataFrame with columns:
        ['latent_topic', 'dominant_named_topic', 'left_weight', 'right_weight', 'lean']
    """
    # Dominant named topic for each latent topic component
    named_topics = corpus.index.tolist()
    latent_dominant = []
    for i in range(doc_topic_matrix.shape[1]):
        best_doc = int(np.argmax(doc_topic_matrix[:, i]))
        latent_dominant.append(named_topics[best_doc])

    rows = []
    for i, dominant in enumerate(latent_dominant):
        rows.append({
            'latent_topic': f'Latent Topic {i+1}',
            'dominant_named_topic': dominant,
        })

    return pd.DataFrame(rows)


def run_lda_pipeline(combined_phrases: pd.DataFrame, outlet_bias_labels: dict,
                     n_components: int = 8) -> dict:
    """
    Full LDA pipeline. Returns a dict with all results needed for display/export.
    """
    corpus = build_phrase_corpus(combined_phrases)
    lda, vec, doc_topic_mat, feature_names = fit_lda(
        corpus, n_components=n_components
    )
    topic_words  = top_words_per_topic(lda, feature_names)
    bias_profile = topic_bias_profile(doc_topic_mat, corpus, None, outlet_bias_labels)

    return {
        'lda':             lda,
        'vectorizer':      vec,
        'doc_topic_matrix': doc_topic_mat,
        'corpus':          corpus,
        'feature_names':   feature_names,
        'topic_words':     topic_words,
        'bias_profile':    bias_profile,
        'n_components':    n_components,
    }
