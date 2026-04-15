"""
Hybrid Media Bias Detection Framework -- Main Pipeline
======================================================
Run with:
    python src/main.py

Stages:
  1.  Data loading
  2.  Bias metrics  (PLS, Entropy, PEI, Log-Odds, Chi-Square)
  3.  Dimensionality reduction (PCA + t-SNE)
  4.  Clustering (K-Means auto-k + Hierarchical)
  5.  Jaccard Similarity Mining
  6.  LDA Topic Modeling
  7.  Ensemble ML Classifier (trains on All-The-News if available)
  8.  Network analysis (co-occurrence graph + PageRank + community detection)
  9.  Visualisation (12 plots)
  10. Export CSVs
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OUTLET_BIAS_LABELS, RESULTS_DIR, PROCESSED_DATA_DIR
from src.data.loader import load_all_topics, create_phrase_outlet_matrix

from src.analysis.pattern_mining import (
    normalize_phrase_counts,
    perform_pca, perform_tsne,
    cluster_outlets, linkage_matrix,
    mine_association_rules,
    outlet_jaccard_similarity, top_similar_outlet_pairs,
    detect_bias_patterns,
    keyword_distinctiveness,
)
from src.analysis.bias_metrics import (
    partisan_lean_score, bias_entropy,
    phrase_exclusivity_index, outlet_bias_report,
    log_odds_ratio, chi_square_partisan,
)
from src.models.classifier import train_classifier, predict
from src.models.network_analysis import (
    build_outlet_network, node_metrics,
    detect_communities, get_network_summary,
)
from src.analysis.topic_modeling import run_lda_pipeline
from src.visualization.plots import (
    plot_bias_heatmap, plot_pca_results, plot_tsne_results,
    plot_cluster_dendrogram, plot_topic_distribution,
    plot_outlet_bias_comparison, plot_log_odds_chart,
    plot_partisan_lean_scores, plot_confusion_matrix,
    plot_outlet_network, plot_bias_entropy,
    plot_jaccard_heatmap,
)

os.makedirs(RESULTS_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

DIVIDER = "=" * 65

def header(text):
    print(f"\n{DIVIDER}")
    print(f"  {text}")
    print(DIVIDER)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 1 / 9 — Data Loading")

combined_phrases, combined_counts = load_all_topics()
n_topics  = combined_phrases['topic'].nunique() if not combined_phrases.empty else 0
n_phrases = len(combined_phrases)

print(f"  Topics loaded  : {n_topics}")
print(f"  Total phrases  : {n_phrases:,}")
print(f"  Outlets in data: {len([c for c in combined_counts.columns if c not in ['PHRASE','TOTAL','topic']])}")

if combined_phrases.empty:
    print("\n  [ERROR] No data loaded. Check phrasebias_data/ directory.")
    sys.exit(1)

combined_matrix = create_phrase_outlet_matrix(combined_counts)
print(f"  Phrase-outlet matrix shape: {combined_matrix.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Bias Metrics
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 2 / 9 — Bias Metrics (PLS, Entropy, PEI, LOR, Chi-Square)")

print("\n  [2a] Computing Outlet-Level Metrics...")
bias_report = outlet_bias_report(combined_matrix, OUTLET_BIAS_LABELS)
print(f"       Outlets scored: {len(bias_report)}")
print("\n  Top 5 most RIGHT-leaning outlets (data-driven PLS):")
top_right = bias_report.nlargest(5, 'partisan_lean_score')
for outlet, row in top_right.iterrows():
    print(f"       {outlet:20s}  PLS={row['partisan_lean_score']:+.4f}  label={row['known_label']}")

print("\n  Top 5 most LEFT-leaning outlets (data-driven PLS):")
top_left = bias_report.nsmallest(5, 'partisan_lean_score')
for outlet, row in top_left.iterrows():
    print(f"       {outlet:20s}  PLS={row['partisan_lean_score']:+.4f}  label={row['known_label']}")

print("\n  [2b] Computing Phrase-Level Log-Odds Ratios...")
lor_df = log_odds_ratio(combined_matrix, OUTLET_BIAS_LABELS)
print(f"       Phrases scored: {len(lor_df):,}")
print("\n  Top 5 most RIGHT-skewed phrases:")
for phrase, row in lor_df.head(5).iterrows():
    print(f"       '{phrase}'  LOR={row['log_odds_ratio']:+.3f}")
print("\n  Top 5 most LEFT-skewed phrases:")
for phrase, row in lor_df.tail(5).iterrows():
    print(f"       '{phrase}'  LOR={row['log_odds_ratio']:+.3f}")

print("\n  [2c] Running Chi-Square Significance Tests...")
chi_df = chi_square_partisan(combined_matrix, OUTLET_BIAS_LABELS)
sig_count = chi_df['significant'].sum()
print(f"       Statistically significant partisan phrases (p<0.05): {sig_count:,} / {len(chi_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Dimensionality Reduction
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 3 / 9 — Dimensionality Reduction (PCA + t-SNE)")

pca_df, pca_model = perform_pca(combined_matrix)
print(f"  PCA variance explained: {pca_model.explained_variance_ratio_.sum():.2%}")

print("  Running t-SNE... (this may take ~30 s)")
tsne_df = perform_tsne(combined_matrix)
print("  t-SNE complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Clustering
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 4 / 9 — Clustering (K-Means auto-k + Hierarchical)")

cluster_labels = cluster_outlets(combined_matrix)
best_k = cluster_labels.nunique()
print(f"  Auto-selected k = {best_k}  (via silhouette score)")
for c in sorted(cluster_labels.unique()):
    members = cluster_labels[cluster_labels == c].index.tolist()
    print(f"  Cluster {c}: {members}")

lmat = linkage_matrix(combined_matrix)
print("  Hierarchical linkage matrix computed.")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Association Rule Mining
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 5 / 9 — Outlet Similarity Mining (Jaccard Similarity)")
print("  Computing pairwise Jaccard similarity across all outlets...")

jaccard_df = outlet_jaccard_similarity(combined_matrix)
pairs_df   = top_similar_outlet_pairs(jaccard_df, top_n=20)

print(f"  Similarity matrix shape: {jaccard_df.shape}")
print(f"\n  Top 10 most similar outlet pairs (by shared phrase fingerprint):")
for _, row in pairs_df.head(10).iterrows():
    print(f"    {row['outlet_a']:20s} <-> {row['outlet_b']:20s}  Jaccard={row['jaccard_similarity']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 -- LDA Topic Modeling
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 6 / 10 -- LDA Topic Modeling")

try:
    lda_results = run_lda_pipeline(combined_phrases, OUTLET_BIAS_LABELS, n_components=8)
    print(f"  Discovered {lda_results['n_components']} latent topics across {len(lda_results['corpus'])} named topics")
    print("\n  Top words per latent topic:")
    for topic_name, words in lda_results['topic_words'].items():
        top_words_str = ', '.join([w for w, _ in words[:7]])
        print(f"    {topic_name}: {top_words_str}")
    lda_results['bias_profile'].to_csv(os.path.join(PROCESSED_DATA_DIR, 'lda_topic_profiles.csv'), index=False)
except Exception as e:
    print(f"  [WARN] LDA failed: {e}")
    lda_results = None

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 -- Ensemble ML Bias Classifier (All-The-News dataset)
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 7 / 10 -- Ensemble ML Classifier")

cm       = None
report   = None
ensemble = None

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
article_csvs = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')] if os.path.isdir(RAW_DATA_DIR) else []

if article_csvs:
    print(f"  Found article dataset: {article_csvs[0]}")
    print("  Loading articles (chunked, balanced sampling)...")
    try:
        from src.data.article_loader import load_articles
        from src.models.trainer import train, load_model
        from src.models.explainer import save_lr_for_explanation

        article_df = load_articles(max_per_label=8000)
        print(f"  Training ensemble on {len(article_df):,} articles...")
        results = train(article_df, verbose=True)
        ensemble = results['pipeline']
        cm       = results['confusion_matrix']
        report   = results['report']

        # Save the LR sub-model separately for the explainer
        # Access the first model (LR) from the ensemble's internal list
        try:
            save_lr_for_explanation(ensemble.models[0])
            print("  LR explainer saved for word-level explanations.")
        except Exception as ex:
            print(f"  [WARN] Could not save LR explainer: {ex}")

        # Demo predictions
        test_phrases = [
            "border security illegal immigrants wall deport",
            "climate change carbon emissions renewable energy policy",
            "federal reserve interest rates inflation monetary policy",
        ]
        print("\n  Demo predictions:")
        from src.models.trainer import predict_single
        for phrase in test_phrases:
            res = predict_single(phrase, ensemble)
            probs_str = '  '.join(f"{k}:{v:.2f}" for k, v in res['probabilities'].items())
            print(f"  '{phrase[:55]}'")
            print(f"       -> {res['label']}  [{probs_str}]")

    except Exception as e:
        import traceback
        print(f"  [WARN] Ensemble classifier failed: {e}")
        traceback.print_exc()
else:
    print("  [SKIP] No article CSV found in data/raw/.")
    print("  Place the All-The-News CSV there and re-run to train the full classifier.")
    # Fall back to old phrase-level classifier for demo purposes
    try:
        pipeline, train_df, report, cm = train_classifier(combined_matrix, OUTLET_BIAS_LABELS)
        print(f"  [Fallback] Phrase-level classifier trained on {len(train_df)} samples.")
    except Exception as e:
        print(f"  [WARN] Fallback classifier also failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 -- Network Analysis
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 8 / 10 -- Network Analysis (Co-occurrence Graph)")

G          = build_outlet_network(combined_matrix, min_cooccur=3)
net_stats  = get_network_summary(G)
node_df    = node_metrics(G)
communities = detect_communities(G)

print(f"  Network: {net_stats.get('nodes',0)} nodes, {net_stats.get('edges',0)} edges")
print(f"  Density         : {net_stats.get('density', 0):.4f}")
print(f"  Avg edge weight : {net_stats.get('avg_edge_weight', 0):.1f} shared phrases")
print(f"  Connected       : {net_stats.get('is_connected', False)}")
print(f"  Communities found: {len(set(communities.values()))}")
print("\n  Top 5 outlets by PageRank (most 'influential' in phrase-sharing):")
for outlet, row in node_df.head(5).iterrows():
    print(f"  {outlet:20s}  PageRank={row['pagerank']:.5f}  degree={int(row['degree'])}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — Visualisations
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 8 / 9 — Generating Visualisations (12 plots)")

def R(filename):
    return os.path.join(RESULTS_DIR, filename)

print("\n  [1/11] Phrase Usage Heatmap")
plot_bias_heatmap(combined_matrix, OUTLET_BIAS_LABELS, save_path=R('bias_heatmap.png'))

print("  [2/11] PCA Scatter Plot")
plot_pca_results(pca_df, OUTLET_BIAS_LABELS, save_path=R('pca_outlets.png'))

print("  [3/11] t-SNE Scatter Plot")
plot_tsne_results(tsne_df, OUTLET_BIAS_LABELS, save_path=R('tsne_outlets.png'))

print("  [4/11] Hierarchical Clustering Dendrogram")
plot_cluster_dendrogram(lmat, combined_matrix.columns.tolist(), save_path=R('dendrogram.png'))

print("  [5/11] Topic Distribution")
plot_topic_distribution(combined_counts, save_path=R('topic_distribution.png'))

print("  [6/11] Outlet Bias Comparison")
plot_outlet_bias_comparison(combined_matrix, OUTLET_BIAS_LABELS, save_path=R('outlet_comparison.png'))

print("  [7/11] Log-Odds Ratio Chart")
plot_log_odds_chart(lor_df, save_path=R('log_odds_ratio.png'))

print("  [8/11] Partisan Lean Score")
pls_series = partisan_lean_score(combined_matrix, OUTLET_BIAS_LABELS)
plot_partisan_lean_scores(pls_series, OUTLET_BIAS_LABELS, save_path=R('partisan_lean_scores.png'))

if cm is not None:
    print("  [9/11] Classifier Confusion Matrix")
    plot_confusion_matrix(cm, ['Left', 'Center', 'Right'], save_path=R('classifier_confusion_matrix.png'))
else:
    print("  [9/11] Skipped (classifier did not train)")

print("  [10/11] Outlet Co-occurrence Network Graph")
plot_outlet_network(G, OUTLET_BIAS_LABELS, communities=communities, save_path=R('outlet_network.png'))

print("  [11/12] Bias Entropy per Outlet")
entropy_series = bias_entropy(combined_matrix)
plot_bias_entropy(entropy_series, OUTLET_BIAS_LABELS, save_path=R('bias_entropy.png'))

print("  [12/12] Jaccard Similarity Heatmap")
plot_jaccard_heatmap(jaccard_df, OUTLET_BIAS_LABELS, save_path=R('jaccard_similarity.png'))

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9 — Export Results
# ─────────────────────────────────────────────────────────────────────────────
header("STAGE 9 / 9 — Exporting Results")

def P(filename):
    return os.path.join(PROCESSED_DATA_DIR, filename)

bias_report.to_csv(P('outlet_bias_report.csv'))
print(f"  Saved outlet_bias_report.csv  ({len(bias_report)} rows)")

lor_df.to_csv(P('log_odds_ratios.csv'))
print(f"  Saved log_odds_ratios.csv     ({len(lor_df)} rows)")

chi_df.to_csv(P('chi_square_results.csv'))
print(f"  Saved chi_square_results.csv  ({len(chi_df)} rows)")

pca_df.to_csv(P('pca_results.csv'))
print(f"  Saved pca_results.csv         ({len(pca_df)} rows)")

tsne_df.to_csv(P('tsne_results.csv'))
print(f"  Saved tsne_results.csv        ({len(tsne_df)} rows)")

node_df.to_csv(P('network_node_metrics.csv'))
print(f"  Saved network_node_metrics.csv ({len(node_df)} rows)")

combined_matrix.to_csv(P('phrase_outlet_matrix.csv'))
print(f"  Saved phrase_outlet_matrix.csv ({combined_matrix.shape})")

jaccard_df.to_csv(P('jaccard_similarity_matrix.csv'))
print(f"  Saved jaccard_similarity_matrix.csv ({jaccard_df.shape})")

pairs_df.to_csv(P('top_outlet_pairs.csv'), index=False)
print(f"  Saved top_outlet_pairs.csv          ({len(pairs_df)} pairs)")

pd.Series(communities, name='community_id').to_csv(P('network_communities.csv'))
print(f"  Saved network_communities.csv ({len(communities)} outlets)")

# ─────────────────────────────────────────────────────────────────────────────
header("COMPLETE")
print(f"\n  Visualisations : {RESULTS_DIR}")
print(f"  Data exports   : {PROCESSED_DATA_DIR}")
print()
