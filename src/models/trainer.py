"""
Sentiment-Aware Ensemble Bias Classifier
=========================================
Key upgrades over v1:

  1. TRIGRAMS (1,3) — captures multi-word phrases like
     "unjustified killing of" or "radical left activists"

  2. VADER SENTIMENT FEATURES — 4 numeric emotion scores
     (pos, neg, neu, compound) added alongside TF-IDF so the
     model distinguishes FRAMING from TOPIC.

     Example:
       "Trump says to kill all immigrants"
         → high neg sentiment about immigrants → Right framing
       "unjustified killing of immigrants leads to panic"
         → high neg sentiment about the killing action + fear
           for victims → Left framing

  3. IMPROVED OUTLET LABELS — NYT, WaPo, CNN now mapped Left/Center
     more accurately per 2024 AllSides ratings.

Models:
  1. Logistic Regression  (fast, interpretable)
  2. Calibrated Linear SVM (strong on text)
  3. Complement Naive Bayes (probabilistic baseline)
"""

import os
import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODEL_PATH    = os.path.join(PROCESSED_DIR, 'ensemble_bias_classifier.joblib')
LABEL_CLASSES = ['Left', 'Center', 'Right']


# ── Module-level ensemble (must be top-level for joblib pickling) ─────────────

def _aligned_proba(model, X, classes):
    """Return probability matrix aligned to `classes` order."""
    if hasattr(model, 'classes_'):
        model_classes = list(model.classes_)
    else:
        model_classes = list(model.named_steps['clf'].classes_)
    proba   = model.predict_proba(X)
    n       = proba.shape[0]
    aligned = np.zeros((n, len(classes)))
    for i, c in enumerate(classes):
        if c in model_classes:
            aligned[:, i] = proba[:, model_classes.index(c)]
    return aligned


class AlignedEnsemble:
    """
    Soft-voting ensemble over pre-fitted sklearn pipelines.
    Top-level class so joblib can pickle/unpickle it.
    """
    def __init__(self, models, classes):
        self.models   = models
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        classes = list(self.classes_)
        return np.mean([_aligned_proba(m, X, classes) for m in self.models], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── VADER Sentiment Feature Transformer ──────────────────────────────────────

class VaderSentimentFeatures(BaseEstimator, TransformerMixin):
    """
    Converts text into 4 VADER sentiment scores:
      [pos, neg, neu, compound]

    These allow the model to separate FRAMING from TOPIC:
      - "unjustified killing of immigrants" → neg=high, compound<0  → Left framing
      - "illegal aliens flooding the border" → neg=high, compound<0 → Right framing
        BUT "flooding" / "illegal aliens" are strong Right n-grams that override
        while "unjustified killing" is more neutral n-gram-wise.
    """
    def __init__(self):
        self._analyzer = None

    def _get_analyzer(self):
        if self._analyzer is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    def fit(self, X, y=None):
        self._get_analyzer()
        return self

    def transform(self, X):
        analyzer = self._get_analyzer()
        scores = []
        for text in X:
            s = analyzer.polarity_scores(str(text))
            scores.append([s['pos'], s['neg'], s['neu'], s['compound']])
        return np.array(scores, dtype=np.float32)

    def get_feature_names_out(self):
        return np.array(['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound'])


# ── Feature Extraction ────────────────────────────────────────────────────────

def build_feature_extractor(max_features: int = 60_000) -> FeatureUnion:
    """
    THREE feature groups combined:
      1. Word n-grams (1,3): unigrams + bigrams + trigrams
         → captures "unjustified killing of" as a unit
      2. Char n-grams (2,4): sub-word patterns, slang, style
      3. VADER sentiment (4 features): framing signal
    """
    word_tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),           # ← trigrams added
        max_features=max_features // 2,
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        token_pattern=r'\b[a-zA-Z][a-zA-Z0-9\'-]{1,}\b',
    )
    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=max_features // 2,
        sublinear_tf=True,
        min_df=2,
    )
    return FeatureUnion([
        ('word', word_tfidf),
        ('char', char_tfidf),
        ('sentiment', VaderSentimentFeatures()),
    ])


# ── Individual Pipelines ──────────────────────────────────────────────────────

def _lr_pipeline(features):
    return Pipeline([
        ('features', features),
        ('clf', LogisticRegression(
            max_iter=1000, C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            multi_class='multinomial',
        )),
    ])


def _svm_pipeline(features):
    base = SGDClassifier(
        loss='modified_huber',
        alpha=5e-5,
        max_iter=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ('features', features),
        ('clf', CalibratedClassifierCV(base, cv=3, method='isotonic')),
    ])


def _nb_pipeline(features):
    # PassiveAggressiveClassifier handles negative values (unlike ComplementNB)
    # and is excellent for text — often outperforms NB on larger datasets
    from sklearn.linear_model import PassiveAggressiveClassifier
    base = PassiveAggressiveClassifier(
        C=0.5,
        max_iter=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ('features', features),
        ('clf', CalibratedClassifierCV(base, cv=3, method='isotonic')),
    ])


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 60_000,
    verbose: bool = True,
) -> dict:
    """
    Train the sentiment-aware ensemble on df['text'] / df['label'].
    """
    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if verbose:
        print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
        print("  Features: word trigrams + char n-grams + VADER sentiment")

    # --- Model 1: Logistic Regression ---
    if verbose: print("  [1/3] Fitting Logistic Regression (with trigrams + sentiment)...")
    lr = _lr_pipeline(build_feature_extractor(max_features))
    lr.fit(X_train, y_train)

    # --- Model 2: Linear SVM ---
    if verbose: print("  [2/3] Fitting Calibrated SVM...")
    svm = _svm_pipeline(build_feature_extractor(max_features))
    svm.fit(X_train, y_train)

    # --- Model 3: Naive Bayes ---
    if verbose: print("  [3/3] Fitting Complement Naive Bayes...")
    nb = _nb_pipeline(build_feature_extractor(max_features))
    nb.fit(X_train, y_train)

    # --- Ensemble ---
    if verbose: print("  Building soft-voting ensemble...")
    all_classes = sorted(list(set(y_train)))
    ensemble = AlignedEnsemble([lr, svm, nb], all_classes)

    # --- Evaluation ---
    if verbose: print("  Evaluating on test set...")
    y_pred  = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=all_classes)
    cm     = confusion_matrix(y_test, y_pred, labels=all_classes)
    acc    = accuracy_score(y_test, y_pred)

    try:
        le  = LabelEncoder().fit(all_classes)
        roc = roc_auc_score(
            le.transform(y_test), y_proba,
            multi_class='ovr', average='macro',
        )
    except Exception:
        roc = None

    if verbose:
        print(f"\n  Accuracy : {acc:.4f}")
        if roc: print(f"  ROC-AUC  : {roc:.4f}")
        print(f"\n{report}")

    # --- Framing test (the exact examples the user flagged) ---
    if verbose:
        test_cases = [
            ("Trump says to kick out all immigrants",            "Right"),
            ("unjustified killing of immigrants leads to panic", "Left"),
            ("climate activists demand fossil fuel ban",         "Left"),
            ("federal reserve holds interest rates steady",      "Center"),
        ]
        print("  Framing sanity check:")
        for text, expected in test_cases:
            proba  = ensemble.predict_proba([text])[0]
            pred   = all_classes[int(np.argmax(proba))]
            status = "OK" if pred == expected else "WRONG"
            probs  = "  ".join(f"{c}:{p:.2f}" for c, p in zip(all_classes, proba))
            print(f"  [{status}] '{text[:55]}'")
            print(f"          -> {pred} (expected {expected})  [{probs}]")

    # --- Persist ---
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dump(ensemble, MODEL_PATH)
    if verbose: print(f"\n  Model saved -> {MODEL_PATH}")

    return {
        'pipeline':         ensemble,
        'report':           report,
        'confusion_matrix': cm,
        'roc_auc':          roc,
        'accuracy':         acc,
        'classes':          all_classes,
        'test_df':          pd.DataFrame({'text': X_test, 'true': y_test, 'predicted': y_pred}),
    }


def load_model():
    """Load the saved ensemble from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model at {MODEL_PATH}. "
            "Run src/main.py with the dataset in place first."
        )
    return load(MODEL_PATH)


def predict_single(text: str, model=None) -> dict:
    """
    Predict bias for a single string.

    Returns:
        {
          'label':         'Left' | 'Center' | 'Right',
          'confidence':    float,
          'probabilities': {'Left': float, 'Center': float, 'Right': float}
        }
    """
    if model is None:
        model = load_model()

    proba     = model.predict_proba([text])[0]
    classes   = list(model.classes_)
    label     = classes[int(np.argmax(proba))]
    conf      = float(np.max(proba))
    prob_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    return {'label': label, 'confidence': conf, 'probabilities': prob_dict}
