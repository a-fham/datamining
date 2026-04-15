import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = r"C:\Users\Afham Faiyaz Ahmad\Desktop\DATA MINING"
PHRASE_SELECTION_DIR = os.path.join(BASE_DIR, 'phrasebias_data', 'phrase_selection')
PHRASE_COUNTS_DIR = os.path.join(BASE_DIR, 'phrasebias_data', 'phrase_counts')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTLET_LABELS = {
    'breitbart': 'Right', 'dailywire': 'Right', 'fox': 'Right', 'nypost': 'Right',
    'wsj': 'Right', 'nationalreview': 'Right', 'redstate': 'Right',
    'alternet': 'Left', 'huffingtonpost': 'Left', 'motherjones': 'Left',
    'commondreams': 'Left', 'slate': 'Left', 'vox': 'Left', 'dailykos': 'Left',
    'jacobinmag': 'Left', 'intercept': 'Left', 'rawstory': 'Left', 'truthdig': 'Left',
    'counterpunch': 'Left', 'nytimes': 'Center-Left', 'wapo': 'Center-Left',
    'cnn': 'Center', 'nbc': 'Center', 'cbs': 'Center', 'npr': 'Center',
    'pbs': 'Center', 'ap': 'Center', 'atlantic': 'Center', 'economist': 'Center',
    'guardian': 'Center-Left', 'bbc': 'Center', 'aljazeera': 'Center',
    'buzzfeed': 'Center-Left', 'vice': 'Center-Left',
    'federalist': 'Right', 'townhall': 'Right', 'dailycaller': 'Right',
    'americanconservative': 'Right', 'americanspectator': 'Right',
    'pjmedia': 'Right', 'spectator': 'Center-Right', 'infowars': 'Right', 'rt': 'Right'
}

def load_all_data():
    topics = ['abortion', 'blm', 'china', 'climate', 'guns', 'israel', 'palestine', 'russia']
    all_phrases = []
    all_counts = []
    
    for topic in topics:
        phrases_path = os.path.join(PHRASE_SELECTION_DIR, f'{topic}_phrases.csv')
        counts_path = os.path.join(PHRASE_COUNTS_DIR, f'{topic}_counts.csv')
        
        if os.path.exists(phrases_path):
            phrases_df = pd.read_csv(phrases_path)
            phrases_df['topic'] = topic
            all_phrases.append(phrases_df)
            
        if os.path.exists(counts_path):
            counts_df = pd.read_csv(counts_path)
            counts_df['topic'] = topic
            all_counts.append(counts_df)
    
    return pd.concat(all_phrases, ignore_index=True), pd.concat(all_counts, ignore_index=True)

print("=" * 60)
print("HYBRID MEDIA BIAS DETECTION FRAMEWORK")
print("=" * 60)

print("\n[1/6] Loading data...")
phrases_df, counts_df = load_all_data()
print(f"    Loaded {len(phrases_df)} phrases across {phrases_df['topic'].nunique()} topics")

print("\n[2/6] Creating phrase-outlet matrix...")
outlet_cols = [c for c in counts_df.columns if c not in ['PHRASE', 'TOTAL', 'topic', 'Unnamed: 0']]
matrix = counts_df.set_index('PHRASE')[outlet_cols].fillna(0)
print(f"    Matrix shape: {matrix.shape}")

print("\n[3/6] Performing PCA...")
pca = PCA(n_components=2)
vectors = StandardScaler().fit_transform(matrix.T.values)
pca_result = pca.fit_transform(vectors)
pca_df = pd.DataFrame(pca_result, index=matrix.columns, columns=['PC1', 'PC2'])
print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

print("\n[4/6] Clustering outlets...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(StandardScaler().fit_transform(matrix.T.values))
cluster_df = pd.DataFrame({'outlet': matrix.columns, 'cluster': clusters})
print(f"    Cluster distribution: {pd.Series(clusters).value_counts().to_dict()}")

print("\n[5/6] Mining association rules...")
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

binary_matrix = (matrix > 0).astype(int)
transactions = binary_matrix.T.apply(lambda x: list(x[x == 1].index)).tolist()
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

frequent = fpgrowth(df, min_support=0.2, use_colnames=True)
print(f"    Frequent itemsets: {len(frequent)}")

print("\n[6/6] Detecting bias patterns...")
right_outlets = [o for o, l in OUTLET_LABELS.items() if 'Right' in l and o in matrix.columns]
left_outlets = [o for o, l in OUTLET_LABELS.items() if 'Left' in l and o in matrix.columns]

def calc_bias_score(phrase):
    row = matrix.loc[phrase] if phrase in matrix.index else pd.Series([0]*len(matrix.columns), index=matrix.columns)
    right_sum = row[right_outlets].sum() if right_outlets else 0
    left_sum = row[left_outlets].sum() if left_outlets else 0
    total = right_sum + left_sum
    if total == 0:
        return 0
    return (right_sum - left_sum) / total

phrase_bias = {phrase: calc_bias_score(phrase) for phrase in matrix.index[:50]}
biased_right = [p for p, s in phrase_bias.items() if s > 0.2]
biased_left = [p for p, s in phrase_bias.items() if s < -0.2]
print(f"    Right-leaning phrases: {len(biased_right)}")
print(f"    Left-leaning phrases: {len(biased_left)}")

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS...")
print("=" * 60)

print("\n[1/5] Creating PCA plot...")
fig, ax = plt.subplots(figsize=(14, 10))
colors = {'Right': 'red', 'Center-Right': 'orange', 'Center': 'gray', 
          'Center-Left': 'lightblue', 'Left': 'blue'}

for outlet in pca_df.index:
    label = OUTLET_LABELS.get(outlet, 'Unknown')
    color = colors.get(label, 'black')
    ax.scatter(pca_df.loc[outlet, 'PC1'], pca_df.loc[outlet, 'PC2'], 
               c=color, s=100, alpha=0.7)
    ax.annotate(outlet, (pca_df.loc[outlet, 'PC1'], pca_df.loc[outlet, 'PC2']),
               fontsize=7, alpha=0.8)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[l], 
                      markersize=10, label=l) for l in colors.keys()]
ax.legend(handles=handles, loc='best', title='Bias Category')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('News Outlets by Phrase Usage (PCA)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca_outlets.png'), dpi=100)
plt.close()
print("    Saved: pca_outlets.png")

print("\n[2/5] Creating heatmap...")
fig, ax = plt.subplots(figsize=(16, 8))
top_phrases = matrix.sum(axis=1).nlargest(25).index
plot_df = matrix.loc[top_phrases]
sns.heatmap(plot_df.T, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
ax.set_title('Top 25 Phrases Usage by Outlet')
ax.set_xlabel('Phrases')
ax.set_ylabel('Outlets')
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bias_heatmap.png'), dpi=100)
plt.close()
print("    Saved: bias_heatmap.png")

print("\n[3/5] Creating cluster visualization...")
fig, ax = plt.subplots(figsize=(14, 10))
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for cluster_id in range(4):
    mask = clusters == cluster_id
    outlets = matrix.columns[mask]
    for outlet in outlets:
        ax.scatter(pca_df.loc[outlet, 'PC1'], pca_df.loc[outlet, 'PC2'], 
                   c=cluster_colors[cluster_id], s=100, alpha=0.7)
        ax.annotate(outlet, (pca_df.loc[outlet, 'PC1'], pca_df.loc[outlet, 'PC2']),
                   fontsize=7, alpha=0.8)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('News Outlets Clustered by Phrase Usage')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'clusters.png'), dpi=100)
plt.close()
print("    Saved: clusters.png")

print("\n[4/5] Creating topic distribution...")
topic_counts = counts_df.groupby('topic')['TOTAL'].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
topic_counts.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Total Phrase Count')
ax.set_ylabel('Topic')
ax.set_title('Phrase Distribution by Topic')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'topic_distribution.png'), dpi=100)
plt.close()
print("    Saved: topic_distribution.png")

print("\n[5/5] Creating bias comparison...")
bias_scores = pd.DataFrame({
    'outlet': matrix.columns,
    'total_phrases': matrix.sum(axis=0).values
})
bias_scores['bias'] = bias_scores['outlet'].map(OUTLET_LABELS).fillna('Unknown')
bias_scores = bias_scores.sort_values('bias')

fig, ax = plt.subplots(figsize=(14, 8))
bias_colors = bias_scores['bias'].map(colors)
ax.bar(range(len(bias_scores)), bias_scores['total_phrases'], color=bias_colors)
ax.set_xticks(range(len(bias_scores)))
ax.set_xticklabels(bias_scores['outlet'], rotation=90, fontsize=7)
ax.set_xlabel('News Outlet')
ax.set_ylabel('Total Phrase Count')
ax.set_title('Phrase Usage by Outlet (Color = Bias Category)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlet_comparison.png'), dpi=100)
plt.close()
print("    Saved: outlet_comparison.png")

print("\n" + "=" * 60)
print("SAVING RESULTS...")
print("=" * 60)

pca_df.to_csv(os.path.join(PROCESSED_DIR, 'pca_results.csv'))
cluster_df.to_csv(os.path.join(PROCESSED_DIR, 'clusters.csv'), index=False)
matrix.to_csv(os.path.join(PROCESSED_DIR, 'phrase_matrix.csv'))

bias_df = pd.DataFrame([{'phrase': p, 'bias_score': s} for p, s in phrase_bias.items()])
bias_df = bias_df.sort_values('bias_score')
bias_df.to_csv(os.path.join(PROCESSED_DIR, 'phrase_bias_scores.csv'), index=False)

print("\nAnalysis complete!")
print(f"Results: {RESULTS_DIR}")
print(f"Processed data: {PROCESSED_DIR}")
print("\nKey findings:")
print(f"  - {len(biased_right)} phrases associated with right-leaning outlets")
print(f"  - {len(biased_left)} phrases associated with left-leaning outlets")
