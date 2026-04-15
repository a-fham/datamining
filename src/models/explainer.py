"""
Word Importance Explainer
==========================
Provides fast, coefficient-based word-level explanations for any
bias prediction using the fitted TF-IDF + Logistic Regression pipeline.

No SHAP — uses the LR coefficient matrix directly:
  contribution(word) = coefficient[class][word] * TF-IDF(word in text)

Filtering applied:
  - VADER numeric features (vader_pos, vader_neg, …) are excluded
  - Char n-gram features (starting with space or non-alpha) are excluded
  - Only words actually present in the input text are shown
"""

import numpy as np
import os
from joblib import load

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
LR_PATH = os.path.join(PROCESSED_DIR, 'lr_explainer.joblib')


def save_lr_for_explanation(lr_pipeline):
    """Save the Logistic Regression pipeline separately for speed."""
    from joblib import dump
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dump(lr_pipeline, LR_PATH)


def _is_display_word(name: str) -> bool:
    """Return True only for clean, human-readable word n-gram tokens."""
    if name.startswith('vader_'):         # VADER numeric feature — skip
        return False
    if not name[0].isalpha():             # char n-gram or junk — skip
        return False
    if len(name) < 2:
        return False
    return True


def get_top_words(text: str, lr_pipeline=None, top_n: int = 12) -> dict:
    """
    Return the top words that pushed the prediction toward Left or Right.

    Returns
    -------
    dict with keys:
        'predicted_label' : str
        'top_left_words'  : [(word, score), ...]
        'top_right_words' : [(word, score), ...]
        'word_scores'     : {class_name: [(word, score), ...]}
    """
    if lr_pipeline is None:
        if not os.path.exists(LR_PATH):
            return {'error': 'LR explainer not trained yet. Run main.py first.'}
        lr_pipeline = load(LR_PATH)

    feature_union = lr_pipeline.named_steps['features']
    clf           = lr_pipeline.named_steps['clf']

    # Collect feature names — but track which indices are display-safe
    all_names   = []
    for name, transformer in feature_union.transformer_list:
        if hasattr(transformer, 'get_feature_names_out'):
            all_names.extend(transformer.get_feature_names_out())
        else:
            # VADER transformer: 4 numeric slots, name them explicitly
            all_names.extend(['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound'])
    all_names = np.array(all_names)

    # Boolean mask: True = safe to show in the UI
    display_mask = np.array([_is_display_word(n) for n in all_names])

    # Transform input text
    X_vec = feature_union.transform([text])
    if hasattr(X_vec, 'toarray'):
        X_vec = X_vec.toarray()
    X_vec = X_vec[0]

    classes = list(clf.classes_)
    coef    = clf.coef_    # [n_classes, n_features]

    # Indices of features both present in text AND display-safe
    present_idx = np.where((X_vec > 0) & display_mask)[0]
    if len(present_idx) == 0:
        return {
            'predicted_label': 'Unknown',
            'word_scores': {}, 'top_left_words': [], 'top_right_words': []
        }

    # Per-class contributions
    word_scores_per_class = {}
    for i, cls in enumerate(classes):
        contributions = coef[i, present_idx] * X_vec[present_idx]
        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        word_scores_per_class[cls] = [
            (str(all_names[present_idx[j]]), float(contributions[j]))
            for j in order
        ]

    # Overall prediction from logits
    logits      = coef @ X_vec + clf.intercept_
    pred_idx    = int(np.argmax(logits))
    predicted_label = classes[pred_idx]

    # Left vs Right divergence for display-safe present features
    left_idx  = classes.index('Left')  if 'Left'  in classes else 0
    right_idx = classes.index('Right') if 'Right' in classes else -1

    lr_diff = (coef[right_idx] - coef[left_idx]) * X_vec
    sorted_diff = np.argsort(lr_diff[present_idx])

    left_words = [
        (str(all_names[present_idx[j]]), float(lr_diff[present_idx[j]]))
        for j in sorted_diff[:top_n]
        if lr_diff[present_idx[j]] < 0
    ]
    right_words = [
        (str(all_names[present_idx[j]]), float(lr_diff[present_idx[j]]))
        for j in sorted_diff[::-1][:top_n]
        if lr_diff[present_idx[j]] > 0
    ]

    return {
        'predicted_label': predicted_label,
        'word_scores':     word_scores_per_class,
        'top_left_words':  left_words,
        'top_right_words': right_words,
    }


def format_explanation(explanation: dict) -> str:
    """Pretty-print an explanation dict for terminal output."""
    if 'error' in explanation:
        return f"  [Explainer Error] {explanation['error']}"

    lines = [f"  Predicted: {explanation['predicted_label']}"]
    lines.append(f"\n  Words pulling toward RIGHT (+) or LEFT (-):")
    for word, score in explanation.get('top_right_words', [])[:6]:
        lines.append(f"    [RIGHT +{score:.3f}]  {word}")
    for word, score in explanation.get('top_left_words', [])[:6]:
        lines.append(f"    [LEFT  {score:.3f}]  {word}")
    return '\n'.join(lines)
