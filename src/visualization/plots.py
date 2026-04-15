"""
Visualization Module
====================
All plotting functions for the Hybrid Media Bias Detection Framework.
Each function saves a PNG to the results directory when save_path is provided.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#2d3143',
    'grid.linestyle':   '--',
    'font.family':      'DejaVu Sans',
})

BIAS_COLORS = {
    'Right':        '#ef4444',
    'Center-Right': '#f97316',
    'Center':       '#6b7280',
    'Center-Left':  '#60a5fa',
    'Left':         '#3b82f6',
    'Unknown':      '#9ca3af',
}


def _savefig(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Phrase Usage Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_bias_heatmap(count_matrix, outlet_labels, save_path=None, n_phrases=30):
    ordered_outlets = sorted(
        [o for o in outlet_labels if o in count_matrix.columns],
        key=lambda x: ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right'].index(outlet_labels.get(x, 'Center'))
        if outlet_labels.get(x, 'Center') in ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right'] else 2
    )

    plot_matrix = count_matrix[ordered_outlets].head(n_phrases) if ordered_outlets else count_matrix.head(n_phrases)

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        plot_matrix, cmap='magma', ax=ax,
        cbar_kws={'label': 'Phrase Count', 'shrink': 0.6},
        linewidths=0.3, linecolor='#0f1117',
    )
    ax.set_title('Phrase Usage Heatmap by News Outlet\n(ordered Left -> Right)', fontsize=14, pad=12)
    ax.set_xlabel('News Outlets', fontsize=10)
    ax.set_ylabel('Top Phrases', fontsize=10)
    plt.xticks(rotation=75, fontsize=7, ha='right')
    plt.yticks(fontsize=7)
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PCA Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_results(pca_df, outlet_labels, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 10))

    for outlet in pca_df.index:
        label = outlet_labels.get(outlet, 'Unknown')
        color = BIAS_COLORS.get(label, '#9ca3af')
        ax.scatter(pca_df.loc[outlet, 'PC1'], pca_df.loc[outlet, 'PC2'],
                   c=color, s=120, alpha=0.85, edgecolors='white', linewidths=0.4, zorder=3)
        ax.annotate(outlet, (pca_df.loc[outlet, 'PC1'], pca_df.loc[outlet, 'PC2']),
                    fontsize=7.5, alpha=0.9, xytext=(4, 4), textcoords='offset points')

    handles = [mpatches.Patch(color=BIAS_COLORS[l], label=l) for l in BIAS_COLORS if l != 'Unknown']
    ax.legend(handles=handles, loc='best', title='Political Lean', fontsize=9, framealpha=0.3)
    ax.set_xlabel('Principal Component 1', fontsize=11)
    ax.set_ylabel('Principal Component 2', fontsize=11)
    ax.set_title('PCA of News Outlets by Phrase-Usage Fingerprint', fontsize=13)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. t-SNE Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne_results(tsne_df, outlet_labels, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 10))

    for outlet in tsne_df.index:
        label = outlet_labels.get(outlet, 'Unknown')
        color = BIAS_COLORS.get(label, '#9ca3af')
        ax.scatter(tsne_df.loc[outlet, 't-SNE1'], tsne_df.loc[outlet, 't-SNE2'],
                   c=color, s=120, alpha=0.85, edgecolors='white', linewidths=0.4, zorder=3)
        ax.annotate(outlet, (tsne_df.loc[outlet, 't-SNE1'], tsne_df.loc[outlet, 't-SNE2']),
                    fontsize=7.5, alpha=0.9, xytext=(4, 4), textcoords='offset points')

    handles = [mpatches.Patch(color=BIAS_COLORS[l], label=l) for l in BIAS_COLORS if l != 'Unknown']
    ax.legend(handles=handles, loc='best', title='Political Lean', fontsize=9, framealpha=0.3)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('t-SNE of News Outlets by Phrase-Usage Fingerprint', fontsize=13)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hierarchical Clustering Dendrogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_dendrogram(linkage_mat, labels, save_path=None):
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    fig, ax = plt.subplots(figsize=(18, 8))
    scipy_dendrogram(
        linkage_mat, labels=labels, ax=ax,
        leaf_rotation=75, leaf_font_size=8,
        color_threshold=0.7 * max(linkage_mat[:, 2]),
    )
    ax.set_title('Hierarchical Clustering Dendrogram of News Outlets', fontsize=13)
    ax.set_xlabel('News Outlet', fontsize=10)
    ax.set_ylabel('Ward Linkage Distance', fontsize=10)
    ax.tick_params(axis='x', which='both', labelsize=8)
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Topic Distribution (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_topic_distribution(combined_counts, save_path=None):
    topic_counts = combined_counts.groupby('topic')['TOTAL'].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(topic_counts.index, topic_counts.values,
                   color=plt.cm.plasma(np.linspace(0.2, 0.85, len(topic_counts))),
                   edgecolor='none', height=0.7)

    for bar, val in zip(bars, topic_counts.values):
        ax.text(val + topic_counts.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=8)

    ax.set_xlabel('Total Phrase Mentions', fontsize=11)
    ax.set_ylabel('Topic', fontsize=11)
    ax.set_title('Phrase Mention Volume by Topic', fontsize=13)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Outlet Bias Comparison (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_outlet_bias_comparison(count_matrix, outlet_labels, save_path=None):
    outlet_totals = count_matrix.sum(axis=0)
    bias_order    = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']

    outlet_data = []
    for outlet, bias in outlet_labels.items():
        if outlet in outlet_totals.index and bias in bias_order:
            outlet_data.append({'outlet': outlet, 'bias': bias, 'count': outlet_totals[outlet]})

    df = pd.DataFrame(outlet_data)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(18, 7))
    x_pos = 0
    group_ticks, group_labels = [], []

    for bias in bias_order:
        subset = df[df['bias'] == bias].sort_values('count', ascending=False)
        if subset.empty:
            continue
        start = x_pos
        color = BIAS_COLORS[bias]
        for _, row in subset.iterrows():
            ax.bar(x_pos, row['count'], color=color, alpha=0.85, width=0.8, edgecolor='none')
            ax.text(x_pos, row['count'] + df['count'].max() * 0.005,
                    row['outlet'], rotation=75, fontsize=6.5, ha='left', va='bottom')
            x_pos += 1
        group_ticks.append((start + x_pos - 1) / 2)
        group_labels.append(bias)
        x_pos += 1.5  # gap between groups

    ax.set_xticks(group_ticks)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel('Total Phrase Count', fontsize=11)
    ax.set_title('Phrase Volume per Outlet, Grouped by Political Lean', fontsize=13)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Log-Odds Ratio Chart  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_log_odds_chart(lor_df, save_path=None, top_n=25):
    """
    Horizontal diverging bar chart of the top_n most Right-skewed and
    top_n most Left-skewed phrases by Log-Odds Ratio.
    """
    top_right = lor_df.nlargest(top_n, 'log_odds_ratio')
    top_left  = lor_df.nsmallest(top_n, 'log_odds_ratio')
    combined  = pd.concat([top_left, top_right]).drop_duplicates()
    combined  = combined.sort_values('log_odds_ratio')

    fig, ax = plt.subplots(figsize=(13, max(8, len(combined) * 0.32)))

    colors = ['#3b82f6' if v < 0 else '#ef4444' for v in combined['log_odds_ratio']]
    bars = ax.barh(combined.index, combined['log_odds_ratio'], color=colors,
                   edgecolor='none', height=0.75)

    ax.axvline(0, color='#6b7280', linewidth=1.2, linestyle='--')
    ax.set_xlabel('Log-Odds Ratio  (← Left-skewed  |  Right-skewed ->)', fontsize=10)
    ax.set_ylabel('Phrase', fontsize=10)
    ax.set_title('Most Partisan Phrases by Log-Odds Ratio\n(data-driven, no hardcoding)', fontsize=12)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.grid(axis='x', alpha=0.25)

    left_patch  = mpatches.Patch(color='#3b82f6', label='Left-skewed')
    right_patch = mpatches.Patch(color='#ef4444', label='Right-skewed')
    ax.legend(handles=[left_patch, right_patch], fontsize=9, framealpha=0.3)

    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Partisan Lean Score bar chart  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_partisan_lean_scores(pls_series, outlet_labels, save_path=None):
    """
    Diverging horizontal bar chart of outlet Partisan Lean Scores.
    """
    pls = pls_series.dropna().sort_values()

    fig, ax = plt.subplots(figsize=(12, max(8, len(pls) * 0.28)))

    colors = [BIAS_COLORS.get(outlet_labels.get(o, 'Unknown'), '#9ca3af') for o in pls.index]
    ax.barh(pls.index, pls.values, color=colors, edgecolor='none', height=0.75, alpha=0.85)
    ax.axvline(0, color='#6b7280', linewidth=1.2, linestyle='--')

    ax.set_xlabel('Partisan Lean Score  (← Left  |  Right ->)', fontsize=10)
    ax.set_ylabel('News Outlet', fontsize=10)
    ax.set_title('Data-Driven Partisan Lean Score per Outlet', fontsize=13)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', alpha=0.25)

    handles = [mpatches.Patch(color=BIAS_COLORS[l], label=l) for l in BIAS_COLORS if l != 'Unknown']
    ax.legend(handles=handles, fontsize=8, framealpha=0.3, loc='lower right')

    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Classifier Confusion Matrix  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels, save_path=None):
    """
    Annotated confusion matrix for the bias classifier.
    cm: numpy 2D array, labels: list of class names.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Bias Classifier — Confusion Matrix (5-fold CV)', fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=13,
                    color='white' if cm[i, j] > thresh else '#c9d1d9')

    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Network Graph  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_outlet_network(G, outlet_labels, communities=None, save_path=None):
    """
    Visualise the outlet co-occurrence network.
    Node colour = political lean. Node size = weighted degree. Edge thickness = shared phrases.
    """
    import networkx as nx

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(G, weight='weight', seed=42, k=1.8)

    # Node colours and sizes
    node_colors = [BIAS_COLORS.get(outlet_labels.get(n, 'Unknown'), '#9ca3af') for n in G.nodes()]
    weighted_degrees = dict(G.degree(weight='weight'))
    node_sizes = [200 + weighted_degrees[n] * 0.5 for n in G.nodes()]

    # Edge widths proportional to shared phrases
    max_w = max((d['weight'] for _, _, d in G.edges(data=True)), default=1)
    edge_widths = [0.5 + 3 * G[u][v]['weight'] / max_w for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.25, edge_color='#6b7280')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5, font_color='white')

    handles = [mpatches.Patch(color=BIAS_COLORS[l], label=l) for l in BIAS_COLORS if l != 'Unknown']
    ax.legend(handles=handles, fontsize=9, framealpha=0.3, title='Political Lean')
    ax.set_title('Outlet Phrase Co-occurrence Network\n(node size = weighted degree, edge = shared phrases)',
                 fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Bias Entropy Bar Chart  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_bias_entropy(entropy_series, outlet_labels, save_path=None):
    """Bar chart of Shannon entropy per outlet, coloured by known political lean."""
    s = entropy_series.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = [BIAS_COLORS.get(outlet_labels.get(o, 'Unknown'), '#9ca3af') for o in s.index]
    ax.bar(s.index, s.values, color=colors, edgecolor='none', alpha=0.85)
    ax.set_xlabel('News Outlet', fontsize=11)
    ax.set_ylabel('Shannon Entropy (bits)', fontsize=11)
    ax.set_title('Phrase Distribution Entropy per Outlet\n(lower = more concentrated / extreme language)',
                 fontsize=13)
    plt.xticks(rotation=75, fontsize=7, ha='right')
    ax.grid(axis='y', alpha=0.25)

    handles = [mpatches.Patch(color=BIAS_COLORS[l], label=l) for l in BIAS_COLORS if l != 'Unknown']
    ax.legend(handles=handles, fontsize=8, framealpha=0.3)

    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Jaccard Similarity Heatmap  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_jaccard_heatmap(jaccard_df, outlet_labels, save_path=None):
    """
    Clustered heatmap of pairwise Jaccard similarity between all outlets.
    Rows and columns are reordered via hierarchical clustering so that
    politically similar outlets group together visually.
    """
    # Order outlets by political lean for a structured layout
    bias_order = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
    ordered = sorted(
        [o for o in outlet_labels if o in jaccard_df.index],
        key=lambda x: bias_order.index(outlet_labels[x])
        if outlet_labels.get(x) in bias_order else 2
    )
    remaining = [o for o in jaccard_df.index if o not in ordered]
    all_outlets = ordered + remaining
    sub = jaccard_df.loc[all_outlets, all_outlets]

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.eye(len(sub), dtype=bool)   # mask diagonal (always 1.0)
    sns.heatmap(
        sub, ax=ax,
        cmap='YlOrRd',
        mask=mask,
        vmin=0, vmax=sub.values[~mask].max(),
        linewidths=0.3, linecolor='#0f1117',
        cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.6},
        annot=False,
    )
    ax.set_title(
        'Outlet Pairwise Jaccard Similarity\n(rows/cols ordered Left -> Right)',
        fontsize=13, pad=12
    )
    plt.xticks(rotation=75, fontsize=7, ha='right')
    plt.yticks(fontsize=7)
    plt.tight_layout()
    _savefig(fig, save_path)
    print(f"    Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy stubs (keep backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentiment_distribution(sentiment_df, save_path=None):
    """Kept for backward compatibility. Sentiment is now handled via LOR."""
    pass

def plot_wordcloud(phrases_df, topic, save_path=None):
    if 'PHRASE' not in phrases_df.columns:
        return
    count_col = 'COUNT' if 'COUNT' in phrases_df.columns else phrases_df.columns[-1]
    phrase_dict = dict(zip(phrases_df['PHRASE'], phrases_df[count_col]))
    fig, ax = plt.subplots(figsize=(12, 7))
    wc = WordCloud(width=1200, height=700, background_color='#0f1117',
                   colormap='plasma', max_words=80).generate_from_frequencies(phrase_dict)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud: {topic.replace("_", " ").title()}', fontsize=12)
    plt.tight_layout()
    _savefig(fig, save_path)
