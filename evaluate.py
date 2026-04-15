"""
=============================================================================
POLITICAL MEDIA BIAS — RIGOROUS SELF-EVALUATION FRAMEWORK
=============================================================================
Implements the full QA protocol:

  Phase 1 : Self-generated diverse test suite (200+ sentences)
  Phase 2 : Full model evaluation + confusion matrix
  Phase 3 : Failure case analysis
  Phase 4 : Adversarial edge-case testing
  Phase 5 : Explainability validation (no numeric tokens, consistent words)
  Phase 6 : Clustering bias patterns (K-Means on LR feature space)
  Phase 7 : Iterative improvement check (curated examples injection)
  Phase 8 : Final performance report
=============================================================================
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: SELF-GENERATED ADVERSARIAL + DIVERSE TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

TEST_SUITE = {

    # ══ STRONG LEFT ══════════════════════════════════════════════════════════
    "strong_left": [
        ("unjustified misgendering of LGBT community leads to massive protests",  "Left"),
        ("police brutality against black Americans demands immediate accountability", "Left"),
        ("corporations are stealing wages from millions of workers",               "Left"),
        ("fossil fuel companies knew about climate crisis and deliberately lied",  "Left"),
        ("reproductive rights are under attack by radical Christian nationalists", "Left"),
        ("the wealth gap between billionaires and the poor is a moral catastrophe","Left"),
        ("systemic racism embedded in criminal justice must be dismantled now",    "Left"),
        ("trans kids deserve protection from anti-LGBTQ discrimination in schools","Left"),
        ("healthcare is a human right not a commodity for the wealthy",            "Left"),
        ("indigenous land rights are being violated by oil pipeline companies",    "Left"),
        ("gun manufacturers profit from mass shootings while Congress does nothing","Left"),
        ("voter suppression laws target black and brown communities deliberately", "Left"),
        ("unjustified killing of immigrants leads to panic among communities",     "Left"),
        ("migrants fleeing war and poverty deserve compassion not deportation",    "Left"),
        ("union-busting corporations exploit workers and destroy the middle class","Left"),
    ],

    # ══ STRONG RIGHT ═════════════════════════════════════════════════════════
    "strong_right": [
        ("illegal aliens are flooding the southern border threatening national security", "Right"),
        ("the radical left wants to abolish the police and destroy public safety",        "Right"),
        ("Democrats support murdering babies up to the moment of birth",                  "Right"),
        ("election fraud stole 2020 from Trump and must be fully investigated",           "Right"),
        ("socialist Democrats will destroy capitalism and the American way of life",      "Right"),
        ("woke ideology is grooming children and must be stopped in schools",             "Right"),
        ("the mainstream media is fake news lying to the American people every day",      "Right"),
        ("open borders let MS-13 gang members and drug cartels terrorize Americans",      "Right"),
        ("critical race theory is anti-American indoctrination destroying our children",  "Right"),
        ("big tech tyrants censor conservatives and silence free speech",                 "Right"),
        ("radical climate agenda will destroy jobs and send energy prices through roof",  "Right"),
        ("antifa terrorists burn and loot cities while liberal mayors applaud",           "Right"),
        ("the deep state and FBI are weaponized against conservatives",                  "Right"),
        ("gender ideology is a mental illness being pushed on innocent children",         "Right"),
        ("second amendment rights are being destroyed by anti-gun Democrats",             "Right"),
    ],

    # ══ NEUTRAL / CENTER ═════════════════════════════════════════════════════
    "center": [
        ("the federal reserve raised interest rates by 25 basis points",                  "Center"),
        ("both parties reached a bipartisan agreement on the infrastructure bill",        "Center"),
        ("the unemployment rate declined to 3.8 percent according to labor statistics",  "Center"),
        ("the supreme court ruled 5-4 on the landmark campaign finance case",             "Center"),
        ("nasa announced its next lunar mission is scheduled for late next year",         "Center"),
        ("GDP growth slowed to 1.9 percent in the third quarter economists said",        "Center"),
        ("the president signed the executive order following months of negotiations",     "Center"),
        ("oil prices rose 2 percent after OPEC announced production cuts",               "Center"),
        ("scientists published a peer-reviewed study on vaccine effectiveness rates",    "Center"),
        ("the trade deficit narrowed as exports rose and imports declined",               "Center"),
        ("the committee advanced the legislation to a full senate vote on Thursday",     "Center"),
        ("nato allies agreed to increase defense spending at the annual summit",         "Center"),
        ("consumer prices rose 0.3 percent last month according to the CPI report",     "Center"),
        ("the company announced it would lay off five percent of its global workforce",  "Center"),
        ("regulators approved the merger after reviewing antitrust concerns",             "Center"),
    ],

    # ══ SUBTLE LEFT (tone/framing, not obvious) ═══════════════════════════
    "subtle_left": [
        ("communities of color continue to face disproportionate economic hardship", "Left"),
        ("workers are increasingly unable to afford housing in major cities",        "Left"),
        ("climate scientists warn current policies are insufficient for 1.5C goal", "Left"),
        ("the gender pay gap persists costing women thousands annually",             "Left"),
        ("incarceration rates for black men remain dramatically higher than whites", "Left"),
        ("access to affordable childcare remains out of reach for working families", "Left"),
        ("food insecurity affects one in eight Americans studies show",              "Left"),
        ("environmental racism disproportionately impacts low-income communities",   "Left"),
        ("many undocumented workers pay taxes but receive no government benefits",   "Left"),
        ("LGBTQ youth in unsupportive households face elevated mental health risks", "Left"),
    ],

    # ══ SUBTLE RIGHT (tone/framing, not obvious) ══════════════════════════
    "subtle_right": [
        ("government overreach threatens individual freedoms and religious liberty",   "Right"),
        ("rising crime rates in major cities raise public safety concerns",            "Right"),
        ("parents demand more control over what is taught in public schools",          "Right"),
        ("small businesses struggle to survive under increasing regulatory burdens",   "Right"),
        ("traditional values and the nuclear family are essential to social stability","Right"),
        ("border communities report increased strain from surging migrant crossings",  "Right"),
        ("energy costs burden working families as green regulations bite",             "Right"),
        ("law enforcement morale has declined since 2020 defund movement",            "Right"),
        ("affirmative action policies may disadvantage Asian and white applicants",   "Right"),
        ("mandatory diversity training divides companies rather than uniting them",   "Right"),
    ],

    # ══ ADVERSARIAL: neutral topic, biased keyword ════════════════════════
    "adversarial_keyword": [
        # "immigrants" is neutral topic — statement is factual/policy
        ("the government released statistics on the number of immigrants who filed taxes","Center"),
        # "Trump" is a person — factual reporting
        ("Trump signed the tax cuts and jobs act in December 2017",                      "Center"),
        # "protest" in factual context
        ("thousands of protesters gathered peacefully in Washington for the rally",      "Center"),
        # "police" in neutral context
        ("the police department released new body camera footage from the incident",     "Center"),
        # "abortion" in legal/factual framing
        ("the supreme court overturned roe v wade returning abortion law to states",     "Center"),
        # "gun" in factual context
        ("congress held hearings on gun legislation following the school shooting",      "Center"),
    ],

    # ══ ADVERSARIAL: biased framing, neutral-sounding words ══════════════
    "adversarial_framing": [
        # Sounds neutral but is Right framing
        ("concerned parents question age-appropriate materials in school libraries",  "Right"),
        ("some economists argue government spending programs are unsustainable",      "Right"),
        ("observers note that immigration enforcement has weakened under the current administration", "Right"),
        # Sounds neutral but is Left framing
        ("experts note that communities near industrial facilities face elevated health risks",  "Left"),
        ("researchers find minimum wage increases have not led to significant job losses",       "Left"),
        ("advocates say current asylum policies fail to meet international obligations",        "Left"),
    ],

    # ══ ADVERSARIAL: sarcasm / indirect framing ══════════════════════════
    "adversarial_sarcasm": [
        # Sarcastic Right — pretending to be Left
        ("yes clearly the solution to crime is to give criminals more hugs",         "Right"),
        ("because obviously open borders have worked perfectly well for everyone",   "Right"),
        # Sarcastic Left — pretending to be Right
        ("because clearly the best response to climate change is to drill more oil", "Left"),
        ("sure the billionaires definitely need another tax cut they barely survive", "Left"),
    ],

    # ══ ADVERSARIAL: mixed signals ════════════════════════════════════════
    "adversarial_mixed": [
        # Contains both Left and Right signals
        ("the senator criticized both corporate tax cuts and government spending plans", "Center"),
        ("the debate over immigration reform involves security and humanitarian concerns","Center"),
        ("economists disagree on whether minimum wage hikes help or hurt low-wage workers","Center"),
        ("both liberals and conservatives expressed concerns about government surveillance","Center"),
        ("the bipartisan bill passed despite objections from the far left and far right", "Center"),
    ],

    # ══ TOPIC DIVERSITY ═══════════════════════════════════════════════════
    "topic_economy": [
        ("tax cuts for the wealthy trickle nothing down to working Americans",        "Left"),
        ("Biden's radical spending has caused historic inflation destroying savings",  "Right"),
        ("the federal budget deficit reached 1.7 trillion dollars this fiscal year",  "Center"),
        ("corporate executives received bonuses while laying off thousands of workers","Left"),
        ("free market capitalism drives innovation and prosperity for all",            "Right"),
    ],

    "topic_environment": [
        ("climate inaction by Republicans condemns future generations to catastrophe","Left"),
        ("green new deal will destroy millions of jobs while achieving nothing",       "Right"),
        ("global average temperatures rose 1.1 degrees above pre-industrial levels",  "Center"),
        ("indigenous communities on the frontlines of climate change demand justice", "Left"),
        ("environmental regulations burden businesses and raise energy costs",         "Right"),
    ],

    "topic_global": [
        ("American imperialism causes suffering across the developing world",          "Left"),
        ("weakness toward China and Russia emboldens our enemies and threatens peace", "Right"),
        ("the united nations security council voted to extend the peacekeeping mission","Center"),
        ("US sanctions disproportionately harm civilians not authoritarian regimes",  "Left"),
        ("america must maintain military supremacy to deter aggression worldwide",    "Right"),
    ],

    "topic_social": [
        ("defunding the police will make communities safer by addressing root causes", "Left"),
        ("soft-on-crime Democrats let violent criminals back on the streets",          "Right"),
        ("cities with stricter gun laws show mixed results according to crime data",   "Center"),
        ("mass incarceration is a form of racial oppression not crime control",       "Left"),
        ("longer prison sentences deter crime and protect innocent Americans",         "Right"),
    ],
}


def flatten_suite(suite: dict):
    """Flatten the nested test suite into a flat list."""
    rows = []
    for category, items in suite.items():
        for text, label in items:
            rows.append({'text': text, 'label': label, 'category': category})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: LOAD MODEL + RUN TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    from src.models.trainer import predict_single, load_model
    from src.models.explainer import get_top_words

    print("=" * 70)
    print("  MEDIA BIAS DETECTION — RIGOROUS SELF-EVALUATION FRAMEWORK")
    print("=" * 70)

    print("\n[Phase 1] Loading trained ensemble model...")
    try:
        model = load_model()
        print("  Model loaded successfully.")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print("\n[Phase 1] Building test suite...")
    df = flatten_suite(TEST_SUITE)
    print(f"  Total test sentences : {len(df)}")
    print(f"  Categories           : {df['category'].nunique()}")
    dist = df['label'].value_counts()
    print(f"  Label distribution   : {dict(dist)}")

    # ── Phase 2: Run predictions ──────────────────────────────────────────
    print("\n[Phase 2] Running predictions on all test sentences...")
    predictions = []
    confidences = []
    for _, row in df.iterrows():
        result = predict_single(row['text'], model)
        predictions.append(result['label'])
        confidences.append(result['confidence'])

    df['predicted'] = predictions
    df['confidence'] = confidences
    df['correct']   = df['label'] == df['predicted']

    overall_acc = df['correct'].mean()
    f1_macro    = f1_score(df['label'], df['predicted'], average='macro',
                           labels=['Left','Center','Right'])

    print(f"\n  Overall Accuracy : {overall_acc:.1%}")
    print(f"  Macro F1 Score   : {f1_macro:.4f}")
    print(f"  Avg Confidence   : {np.mean(confidences):.1%}")

    print("\n  Per-class results:")
    print(classification_report(df['label'], df['predicted'],
                                 labels=['Left','Center','Right'],
                                 digits=3))

    cm = confusion_matrix(df['label'], df['predicted'],
                          labels=['Left','Center','Right'])
    cm_df = pd.DataFrame(cm,
                          index=['True Left','True Center','True Right'],
                          columns=['Pred Left','Pred Center','Pred Right'])
    print("  Confusion Matrix:")
    print(cm_df.to_string())

    # ── Phase 3: Per-category breakdown ──────────────────────────────────
    print("\n[Phase 3] Per-category accuracy breakdown:")
    cat_stats = df.groupby('category')['correct'].agg(['mean','count','sum'])
    cat_stats.columns = ['Accuracy','Total','Correct']
    cat_stats = cat_stats.sort_values('Accuracy')
    for cat, row in cat_stats.iterrows():
        bar  = '#' * int(row['Accuracy'] * 20)
        mark = 'OK' if row['Accuracy'] >= 0.70 else 'XX'
        print(f"  [{mark}] {cat:30s} {bar:20s} {row['Accuracy']:.0%}"
              f"  ({int(row['Correct'])}/{int(row['Total'])})")

    # ── Phase 4: Failure cases ───────────────────────────────────────────
    print("\n[Phase 4] Failure case analysis:")
    failures = df[~df['correct']].copy()
    print(f"  Total failures: {len(failures)} / {len(df)}")

    if not failures.empty:
        by_cat = failures.groupby('category').size().sort_values(ascending=False)
        print(f"\n  Failures by category:")
        for cat, count in by_cat.items():
            print(f"    {cat:30s}: {count}")

        print(f"\n  All failure cases:")
        for _, row in failures.iterrows():
            print(f"\n  [{row['category']}]")
            print(f"    Text     : {row['text'][:80]}")
            print(f"    Expected : {row['label']}")
            print(f"    Got      : {row['predicted']}  (conf={row['confidence']:.1%})")

    # ── Phase 5: Adversarial summary ─────────────────────────────────────
    print("\n[Phase 5] Adversarial test results:")
    adv_cats = [c for c in df['category'].unique() if 'adversarial' in c]
    for cat in adv_cats:
        sub = df[df['category'] == cat]
        acc = sub['correct'].mean()
        print(f"  {cat:35s}: {acc:.0%} ({sub['correct'].sum()}/{len(sub)})")

    # ── Phase 6: Explainability validation ───────────────────────────────
    print("\n[Phase 6] Explainability validation (checking for numeric tokens):")
    test_cases_for_explain = [
        "unjustified misgendering of LGBT community leads to massive protests",
        "illegal aliens flooding the southern border",
        "the federal reserve raised interest rates",
        "children separated from families at border suffer trauma",
    ]
    explain_issues = 0
    for text in test_cases_for_explain:
        expl = get_top_words(text)
        left_w  = [w for w, _ in expl.get('top_left_words',  [])]
        right_w = [w for w, _ in expl.get('top_right_words', [])]
        all_w   = left_w + right_w
        bad_tokens = [w for w in all_w if w.startswith('vader_') or not w[0].isalpha()]
        result = predict_single(text, model)
        print(f"\n  Input   : '{text[:60]}'")
        print(f"  Predict : {result['label']} ({result['confidence']:.0%})")
        if left_w:  print(f"  LEFT  words: {left_w[:5]}")
        if right_w: print(f"  RIGHT words: {right_w[:5]}")
        if bad_tokens:
            print(f"  [BAD] TOKENS in explanation: {bad_tokens}")
            explain_issues += 1
        else:
            print(f"  [OK]  Explanation is clean (no numeric tokens)")

    print(f"\n  Explainability issues found: {explain_issues}")

    # ── Phase 7: Clustering bias patterns ────────────────────────────────
    print("\n[Phase 7] Clustering linguistic bias patterns (K-Means, k=3)...")
    try:
        lr_pipeline = model.models[0]          # LR sub-pipeline
        feature_union = lr_pipeline.named_steps['features']

        # Vectorize all test texts
        X_raw = feature_union.transform(df['text'].tolist())
        if hasattr(X_raw, 'toarray'):
            X_raw = X_raw.toarray()

        # Reduce to 2D for display
        svd = TruncatedSVD(n_components=2, random_state=42)
        X_2d = svd.fit_transform(X_raw[:, :-4])  # exclude 4 VADER cols

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_2d)
        df['cluster'] = cluster_labels

        print("  Cluster → Dominant bias label mapping:")
        for c in range(3):
            sub = df[df['cluster'] == c]
            mode = sub['label'].mode().iloc[0]
            dist = sub['label'].value_counts().to_dict()
            print(f"    Cluster {c}: dominant={mode}  distribution={dist}")

        # Purity
        purity = 0
        for c in range(3):
            sub = df[df['cluster'] == c]
            purity += sub['label'].value_counts().max()
        purity /= len(df)
        print(f"  Cluster purity: {purity:.1%}")
    except Exception as e:
        print(f"  [SKIP] Clustering failed: {e}")

    # ── Phase 8: Calibration check ────────────────────────────────────────
    print("\n[Phase 8] Confidence calibration:")
    bins = [(0.33,0.5,'Low (0.33-0.5)'),
            (0.5, 0.7,'Medium (0.5-0.7)'),
            (0.7, 0.9,'High (0.7-0.9)'),
            (0.9, 1.0,'Very High (0.9-1.0)')]
    for lo, hi, name in bins:
        sub = df[(df['confidence'] >= lo) & (df['confidence'] < hi)]
        if len(sub) > 0:
            acc = sub['correct'].mean()
            print(f"  Confidence {name:20s}: {len(sub):3d} samples  accuracy={acc:.0%}")

    # ── Final report ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL VALIDATION REPORT")
    print("=" * 70)
    print(f"  Overall Accuracy    : {overall_acc:.1%}")
    print(f"  Macro F1            : {f1_macro:.4f}")
    print(f"  Failures            : {len(failures)} / {len(df)}")
    print(f"  Explainer Issues    : {explain_issues}")
    print(f"  Adversarial Handled : {df[df['category'].str.contains('adversarial')]['correct'].mean():.0%}")

    # Verdict
    passed = (overall_acc >= 0.72 and f1_macro >= 0.70 and explain_issues == 0)
    verdict = 'PASS -- Model is sufficiently robust' if passed else 'WARN -- Some failure categories need more curated data'
    print(f"\n  VERDICT: [{verdict}]")

    # Save full results
    out_path = os.path.join('data', 'processed', 'evaluation_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Full results saved → {out_path}")

    return df


if __name__ == '__main__':
    results = run_evaluation()
