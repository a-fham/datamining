# Hybrid Media Bias Detection Framework

A hybrid deep learning and data mining framework for detecting political bias in news media.

## Project Structure

```
DATA MINING/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # Data loading utilities
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── pattern_mining.py  # Pattern mining algorithms
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py          # Visualization functions
│   └── main.py               # Main analysis pipeline
├── phrasebias_data/          # Dataset (download from Kaggle)
│   ├── phrase_selection/     # Topic-specific phrases
│   ├── phrase_counts/        # Phrase usage by outlet
│   └── blacklist.csv        # Exclusion list
├── data/
│   ├── raw/
│   └── processed/           # Generated analysis results
├── results/                  # Visualizations and reports
├── requirements.txt
└── README.md
```

## Features

- **Phrase-level Bias Detection**: Analyze how different news outlets use biased language
- **Multi-topic Analysis**: 28 topics including abortion, climate, guns, Israel/Palestine, etc.
- **49 News Outlets**: Coverage from far-left to far-right sources
- **Pattern Mining**: Association rules (Apriori, FP-Growth) for bias patterns
- **Clustering**: K-Means and Hierarchical clustering of outlets
- **Dimensionality Reduction**: PCA and t-SNE visualization
- **Sentiment Analysis**: Polarity and subjectivity scoring

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd "C:\Users\Afham Faiyaz Ahmad\Desktop\DATA MINING"
python src/main.py
```

## Dataset

Source: [Kaggle Media Bias Dataset](https://www.kaggle.com/datasets/tegmark/mediabias)

Contains phrase counts across 49 news outlets for 28 controversial topics.

## Output

- `results/bias_heatmap.png` - Phrase usage heatmap
- `results/pca_outlets.png` - PCA visualization
- `results/tsne_outlets.png` - t-SNE visualization
- `results/dendrogram.png` - Hierarchical clustering
- `results/topic_distribution.png` - Topic analysis
- `results/outlet_comparison.png` - Bias comparison
- `data/processed/` - CSV outputs
