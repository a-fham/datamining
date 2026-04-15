"""
Bias Classifier
===============
Trains a TF-IDF + Logistic Regression classifier to predict the political
leaning (Left / Center / Right) of a phrase based on its textual content.
Saves the trained model and reports precision/recall/F1 on a held-out split.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load


MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODEL_PATH = os.path.join(MODEL_DIR, 'bias_classifier.joblib')


def build_training_set(count_matrix: pd.DataFrame, outlet_bias_labels: dict) -> pd.DataFrame:
    """
    Construct a labeled dataset for supervised learning.

    For every phrase in the count_matrix we assign the label of the outlet
    that used it the most.  Phrases where the plurality outlet has no known
    bias label are dropped.

    Returns a DataFrame with columns ['phrase', 'label'].
    """
    # Map labels to 3-class scheme
    label_map = {
        'Left':         'Left',
        'Center-Left':  'Left',
        'Center':       'Center',
        'Center-Right': 'Right',
        'Right':        'Right',
    }

    records = []
    for phrase, row in count_matrix.iterrows():
        # Outlet that mentioned this phrase the most
        dominant_outlet = row.idxmax()
        raw_label = outlet_bias_labels.get(dominant_outlet)
        mapped = label_map.get(raw_label)
        if mapped is None:
            continue
        records.append({'phrase': str(phrase), 'label': mapped})

    return pd.DataFrame(records).drop_duplicates('phrase').reset_index(drop=True)


def train_classifier(count_matrix: pd.DataFrame, outlet_bias_labels: dict) -> tuple:
    """
    Train and return a (pipeline, training_dataframe) pair.

    The pipeline is TfidfVectorizer → LogisticRegression with L2 penalty.
    Uses 5-fold stratified cross-validation for evaluation.
    """
    df = build_training_set(count_matrix, outlet_bias_labels)
    if df.empty or df['label'].nunique() < 2:
        raise ValueError("Not enough labeled data to train a classifier.")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',   # character n-grams capture morphology well
            ngram_range=(2, 5),
            max_features=15_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            multi_class='multinomial',
            C=1.0,
        )),
    ])

    pipeline.fit(df['phrase'], df['label'])

    # Cross-validated predictions for honest evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(pipeline, df['phrase'], df['label'], cv=cv)

    report = classification_report(df['label'], y_pred_cv, output_dict=False)
    cm = confusion_matrix(df['label'], y_pred_cv, labels=['Left', 'Center', 'Right'])

    # Persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(pipeline, MODEL_PATH)

    return pipeline, df, report, cm


def load_classifier():
    """Load a previously saved classifier from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No saved model at {MODEL_PATH}. Run train_classifier first.")
    return load(MODEL_PATH)


def predict(texts: list, pipeline=None):
    """
    Predict the political lean of a list of text strings.

    Returns a list of (label, probability_dict) tuples.
    """
    if pipeline is None:
        pipeline = load_classifier()

    labels = pipeline.classes_
    probas = pipeline.predict_proba(texts)
    predictions = []
    for text, prob_row in zip(texts, probas):
        label = labels[np.argmax(prob_row)]
        prob_dict = {l: round(float(p), 4) for l, p in zip(labels, prob_row)}
        predictions.append({'text': text, 'predicted_label': label, 'probabilities': prob_dict})
    return predictions
