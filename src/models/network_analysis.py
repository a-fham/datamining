"""
Phrase Co-occurrence Network Analysis
======================================
Builds a weighted, undirected graph where:
  - Nodes  = news outlets
  - Edges  = shared phrases (weight = number of phrases used by both outlets)

This captures structural similarity between outlets beyond simple clustering
and allows PageRank-style influence scoring and community detection.
"""

import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations


def build_outlet_network(count_matrix: pd.DataFrame, min_cooccur: int = 5) -> nx.Graph:
    """
    Build a weighted co-occurrence graph of outlets.

    Two outlets are connected if they share at least `min_cooccur` phrases
    (both used the phrase at least once).  Edge weight = number of shared phrases.
    """
    binary = (count_matrix > 0).astype(int)
    outlets = binary.columns.tolist()

    G = nx.Graph()
    G.add_nodes_from(outlets)

    for o1, o2 in combinations(outlets, 2):
        shared = int((binary[o1] & binary[o2]).sum())
        if shared >= min_cooccur:
            G.add_edge(o1, o2, weight=shared)

    return G


def node_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Compute per-outlet network metrics:
      - degree          : number of outlet neighbours
      - weighted_degree : sum of shared-phrase weights
      - betweenness     : how often an outlet lies on shortest paths (broker role)
      - pagerank        : authority/influence score in the phrase-sharing network
      - clustering_coef : tendency to form tight cliques
    """
    metrics = {}

    degree          = dict(G.degree())
    weighted_degree = dict(G.degree(weight='weight'))
    betweenness     = nx.betweenness_centrality(G, weight='weight', normalized=True)
    pagerank        = nx.pagerank(G, weight='weight')
    clustering      = nx.clustering(G, weight='weight')

    for node in G.nodes():
        metrics[node] = {
            'degree':          degree[node],
            'weighted_degree': weighted_degree[node],
            'betweenness':     round(betweenness[node], 6),
            'pagerank':        round(pagerank[node], 6),
            'clustering_coef': round(clustering[node], 6),
        }

    return pd.DataFrame(metrics).T.sort_values('pagerank', ascending=False)


def detect_communities(G: nx.Graph) -> dict:
    """
    Run the Louvain-style greedy modularity maximisation to find
    communities of outlets with high phrase-sharing density.

    Returns a dict mapping outlet_name -> community_id.
    """
    communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    membership = {}
    for cid, community in enumerate(communities):
        for outlet in community:
            membership[outlet] = cid
    return membership


def get_network_summary(G: nx.Graph) -> dict:
    """High-level statistics for the outlet co-occurrence network."""
    if G.number_of_nodes() == 0:
        return {}

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    return {
        'nodes':             G.number_of_nodes(),
        'edges':             G.number_of_edges(),
        'density':           round(nx.density(G), 4),
        'avg_edge_weight':   round(float(np.mean(weights)), 2) if weights else 0,
        'max_edge_weight':   int(max(weights)) if weights else 0,
        'avg_clustering':    round(nx.average_clustering(G, weight='weight'), 4),
        'is_connected':      nx.is_connected(G),
    }
