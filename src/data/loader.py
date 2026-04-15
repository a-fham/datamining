import os
import pandas as pd
import numpy as np
from src.config import PHRASE_SELECTION_DIR, PHRASE_COUNTS_DIR, BLACKLIST_PATH, TOPICS

def load_blacklist():
    with open(BLACKLIST_PATH, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f if line.strip())

def load_topic_phrases(topic):
    phrase_path = os.path.join(PHRASE_SELECTION_DIR, f'{topic}_phrases.csv')
    counts_path = os.path.join(PHRASE_COUNTS_DIR, f'{topic}_counts.csv')
    
    phrases_df = pd.read_csv(phrase_path)
    counts_df = pd.read_csv(counts_path)
    
    return phrases_df, counts_df

def load_all_topics():
    all_phrases = []
    all_counts = []
    
    for topic in TOPICS:
        try:
            phrases_df, counts_df = load_topic_phrases(topic)
            phrases_df['topic'] = topic
            counts_df['topic'] = topic
            all_phrases.append(phrases_df)
            all_counts.append(counts_df)
        except FileNotFoundError:
            continue
    
    combined_phrases = pd.concat(all_phrases, ignore_index=True) if all_phrases else pd.DataFrame()
    combined_counts = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    
    return combined_phrases, combined_counts

def calculate_outlet_bias_scores(counts_df, phrases_df):
    outlet_columns = [col for col in counts_df.columns if col not in ['PHRASE', 'TOTAL', 'topic']]
    
    phrase_bias = {}
    for _, row in phrases_df.iterrows():
        phrase = row['PHRASE']
        phrase_row = counts_df[counts_df['PHRASE'] == phrase]
        if len(phrase_row) > 0:
            phrase_bias[phrase] = phrase_row[outlet_columns].iloc[0].to_dict()
    
    outlet_scores = {}
    for outlet in outlet_columns:
        total_mentions = counts_df[outlet].sum()
        outlet_scores[outlet] = {
            'total_phrases': total_mentions,
            'unique_phrases': len(counts_df[counts_df[outlet] > 0])
        }
    
    return pd.DataFrame(outlet_scores).T, phrase_bias

def create_phrase_outlet_matrix(counts_df):
    outlet_columns = [col for col in counts_df.columns if col not in ['PHRASE', 'TOTAL', 'topic']]
    matrix = counts_df.set_index('PHRASE')[outlet_columns].fillna(0)
    return matrix

if __name__ == "__main__":
    print("Loading dataset...")
    blacklist = load_blacklist()
    print(f"Blacklist entries: {len(blacklist)}")
    
    combined_phrases, combined_counts = load_all_topics()
    print(f"Total topics loaded: {combined_phrases['topic'].nunique() if not combined_phrases.empty else 0}")
    print(f"Total phrases: {len(combined_phrases) if not combined_phrases.empty else 0}")
