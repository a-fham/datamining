"""
Media Bias Detection — Interactive Streamlit App
=================================================
Run with:
    streamlit run app.py

Three tabs:
  1. Article Analyzer  — paste text, get bias prediction + word explanations
  2. Outlet Profile    — select outlet, see all metrics & similar outlets
  3. Explorer          — browse all charts, filter by topic, download CSVs
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Media Bias Detector',
    page_icon='📰',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0d1117; }
    .block-container { padding: 2rem 3rem; max-width: 1400px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #161b22; border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 24px;
        background: transparent; color: #8b949e;
        font-weight: 500; font-size: 15px;
    }
    .stTabs [aria-selected="true"] { background: #1f6feb !important; color: white !important; }

    /* Metric cards */
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 20px 24px;
        margin: 8px 0;
    }
    .metric-card h3 { color: #8b949e; font-size: 13px; font-weight: 500; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { color: #c9d1d9; font-size: 28px; font-weight: 700; margin: 8px 0 0; }

    /* Bias badges */
    .badge-left   { background: #1d3a6e; color: #60a5fa; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 20px; display: inline-block; }
    .badge-center { background: #2a2a2a; color: #9ca3af; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 20px; display: inline-block; }
    .badge-right  { background: #3d1a1a; color: #f87171; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 20px; display: inline-block; }

    /* Word pills */
    .word-left  { background: #1d3a6e; color: #93c5fd; border-radius: 6px; padding: 3px 10px; margin: 3px; display: inline-block; font-size: 13px; }
    .word-right { background: #3d1a1a; color: #fca5a5; border-radius: 6px; padding: 3px 10px; margin: 3px; display: inline-block; font-size: 13px; }
    .word-neutral { background: #1f2937; color: #9ca3af; border-radius: 6px; padding: 3px 10px; margin: 3px; display: inline-block; font-size: 13px; }

    /* Section headers */
    .section-header { color: #c9d1d9; font-size: 18px; font-weight: 600; margin: 24px 0 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }

    div[data-testid="stTextArea"] textarea {
        background: #161b22 !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; border-radius: 10px !important;
        font-size: 14px !important;
    }
    div[data-testid="stSelectbox"] > div { background: #161b22 !important; }
    .stButton button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 8px;
        padding: 10px 32px; font-weight: 600; font-size: 15px;
        transition: all 0.2s;
    }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(31,111,235,0.4); }
</style>
""", unsafe_allow_html=True)

BIAS_COLORS = {
    'Left':        '#3b82f6',
    'Center-Left': '#60a5fa',
    'Center':      '#6b7280',
    'Center-Right': '#f97316',
    'Right':       '#ef4444',
    'Unknown':     '#9ca3af',
}

# ── Cached data loaders ───────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


@st.cache_resource(show_spinner='Loading bias analysis data...')
def load_all_data():
    """Load all precomputed CSVs from data/processed/."""
    def _read(name, **kw):
        path = os.path.join(PROCESSED_DIR, name)
        if os.path.exists(path):
            return pd.read_csv(path, **kw)
        return None

    result = {
        'bias_report':  _read('outlet_bias_report.csv', index_col=0),
        'lor_df':       _read('log_odds_ratios.csv',    index_col=0),
        'chi_df':       _read('chi_square_results.csv', index_col=0),
        'pca_df':       _read('pca_results.csv',        index_col=0),
        'tsne_df':      _read('tsne_results.csv',       index_col=0),
        'node_df':      _read('network_node_metrics.csv', index_col=0),
        'jaccard_df':   _read('jaccard_similarity_matrix.csv', index_col=0),
        'pairs_df':     _read('top_outlet_pairs.csv'),
        'matrix_df':    _read('phrase_outlet_matrix.csv', index_col=0),
        'communities':  _read('network_communities.csv', index_col=0),
    }
    # Deduplicate phrase indices (same phrase can appear in multiple topics)
    for key in ('lor_df', 'chi_df', 'matrix_df'):
        if result[key] is not None and result[key].index.duplicated().any():
            result[key] = result[key][~result[key].index.duplicated(keep='first')]
    return result


@st.cache_resource(show_spinner='Loading ML classifier...')
def load_classifier():
    """Load the trained ensemble classifier."""
    model_path = os.path.join(PROCESSED_DIR, 'ensemble_bias_classifier.joblib')
    if not os.path.exists(model_path):
        return None
    from joblib import load
    return load(model_path)


@st.cache_resource(show_spinner='Loading explainer...')
def load_explainer():
    """Load the LR pipeline for word explanations."""
    from src.models.explainer import LR_PATH
    if not os.path.exists(LR_PATH):
        return None
    from joblib import load
    return load(LR_PATH)


from src.config import OUTLET_BIAS_LABELS


def bias_badge(label: str) -> str:
    css = {'Left': 'badge-left', 'Right': 'badge-right', 'Center': 'badge-center'}
    cls = css.get(label, 'badge-center')
    return f'<span class="{cls}">{label}</span>'


def confidence_bar(probs: dict):
    colors = {'Left': '#3b82f6', 'Center': '#6b7280', 'Right': '#ef4444'}
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    colors_list = [colors.get(l, '#9ca3af') for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors_list, line=dict(width=0)),
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=40, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, range=[0, 110], showticklabels=False,
                   color='#8b949e'),
        yaxis=dict(color='#c9d1d9', tickfont=dict(size=14)),
        font=dict(color='#c9d1d9'),
    )
    return fig


def word_importance_chart(left_words, right_words, top_n=10):
    """Horizontal diverging bar chart of word influence."""
    words_l = [(w, -abs(s)) for w, s in (left_words or [])[:top_n]]
    words_r = [(w,  abs(s)) for w, s in (right_words or [])[:top_n]]
    all_words = words_l + words_r
    if not all_words:
        return None

    df = pd.DataFrame(all_words, columns=['word', 'score']).sort_values('score')
    colors = ['#3b82f6' if s < 0 else '#ef4444' for s in df['score']]

    fig = go.Figure(go.Bar(
        x=df['score'], y=df['word'], orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
    ))
    fig.add_vline(x=0, line_color='#6b7280', line_width=1, line_dash='dash')
    fig.update_layout(
        height=max(300, len(df) * 26),
        margin=dict(l=0, r=20, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#30363d', color='#8b949e',
                   title=dict(text='<-- Left        Right -->', font=dict(size=11))),
        yaxis=dict(color='#c9d1d9', tickfont=dict(size=12)),
        font=dict(color='#c9d1d9'),
        title=dict(text='Word Contributions', font=dict(size=13, color='#c9d1d9'), x=0.5),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding: 32px 0 16px;">
    <div style="font-size: 48px; margin-bottom: 8px;">📰</div>
    <h1 style="color:#c9d1d9; font-size: 36px; font-weight: 700; margin: 0;">
        Media Bias Detector
    </h1>
    <p style="color:#8b949e; font-size: 16px; margin-top: 8px;">
        Hybrid ML + Data Mining framework for detecting political bias in news media
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
data     = load_all_data()
model    = load_classifier()
explainer_lr = load_explainer()

# Tab layout
tab1, tab2, tab3 = st.tabs(['📄  Article Analyzer', '🏢  Outlet Profile', '🔍  Explorer'])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ARTICLE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">Analyze Any Text for Political Bias</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1], gap='large')

    with col_input:
        sample_texts = {
            '-- paste your own text --': '',
            'Breitbart style (Right)': (
                "The Biden administration's open border policies have allowed millions of illegal aliens "
                "to flood into the country, overwhelming border patrol and threatening national security. "
                "Democrats continue to ignore American citizens while kowtowing to radical left activists."
            ),
            'HuffPost style (Left)': (
                "Republican lawmakers have once again blocked critical climate legislation, choosing to "
                "side with fossil fuel corporations over the planet's future. Activists are demanding "
                "urgent action as record temperatures devastate vulnerable communities worldwide."
            ),
            'Reuters style (Center)': (
                "The Federal Reserve raised interest rates by 25 basis points on Wednesday, its tenth "
                "consecutive increase, as policymakers seek to bring inflation back to their 2% target "
                "while avoiding a sharp downturn in the labor market."
            ),
        }
        choice = st.selectbox('Or load a sample:', list(sample_texts.keys()))
        default_text = sample_texts[choice]

        user_text = st.text_area(
            'Paste article text or headline:',
            value=default_text,
            height=220,
            placeholder='Paste any news article, headline, or paragraph here...',
            label_visibility='collapsed',
        )

        analyze_btn = st.button('Analyze Bias', use_container_width=True)

    with col_result:
        if analyze_btn or (choice != '-- paste your own text --' and default_text):
            text_to_analyze = user_text.strip()
            if len(text_to_analyze) < 10:
                st.warning('Please enter at least a sentence of text.')
            elif model is None:
                st.error(
                    'ML classifier not trained yet.\n\n'
                    'Place the All-The-News CSV in `data/raw/` and run `python src/main.py`.'
                )
            else:
                from src.models.trainer import predict_single
                result = predict_single(text_to_analyze, model)

                label = result['label']
                conf  = result['confidence']
                probs = result['probabilities']

                st.markdown(f"""
                <div class="metric-card" style="text-align:center; margin-bottom: 16px;">
                    <h3>Predicted Political Lean</h3>
                    <div style="margin: 12px 0;">{bias_badge(label)}</div>
                    <div style="color:#8b949e; font-size: 14px;">Confidence: <b style="color:#c9d1d9">{conf:.1%}</b></div>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(confidence_bar(probs), use_container_width=True)

                # Word explanations
                if explainer_lr is not None:
                    from src.models.explainer import get_top_words
                    explanation = get_top_words(text_to_analyze, explainer_lr, top_n=10)

                    if 'error' not in explanation:
                        left_words  = explanation.get('top_left_words', [])
                        right_words = explanation.get('top_right_words', [])

                        fig = word_importance_chart(left_words, right_words)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        if left_words or right_words:
                            st.markdown('<div class="section-header" style="font-size:14px;">Key Words Found</div>',
                                        unsafe_allow_html=True)
                            pills_html = ''
                            for w, s in right_words[:6]:
                                pills_html += f'<span class="word-right">+{abs(s):.2f} {w}</span> '
                            for w, s in left_words[:6]:
                                pills_html += f'<span class="word-left">{s:.2f} {w}</span> '
                            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; color:#8b949e; padding: 60px 20px;">
                <div style="font-size: 48px; opacity: 0.3;">📊</div>
                <p style="margin-top: 16px; font-size: 15px;">
                    Enter some text and click <b>Analyze Bias</b>
                </p>
                <p style="font-size: 13px; margin-top: 8px; opacity: 0.7;">
                    Works on headlines, paragraphs, or full articles
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Similar outlets section ──
    if (analyze_btn or choice != '-- paste your own text --') and model is not None and data.get('jaccard_df') is not None:
        st.markdown('<div class="section-header">Most Similarly-Biased Outlets</div>', unsafe_allow_html=True)
        if data['pairs_df'] is not None and label in ['Left', 'Center', 'Right']:
            # Find top outlets with the same predicted label
            br = data['bias_report']
            if br is not None:
                matching = br[br['known_label'].str.contains(label, na=False)]
                if not matching.empty:
                    top_matches = matching.sort_values('partisan_lean_score',
                                    ascending=(label == 'Left')).head(5)
                    cols = st.columns(min(len(top_matches), 5))
                    for i, (outlet, row) in enumerate(top_matches.iterrows()):
                        with cols[i]:
                            color = BIAS_COLORS.get(row.get('known_label', 'Unknown'), '#9ca3af')
                            st.markdown(f"""
                            <div class="metric-card" style="text-align:center; border-top: 3px solid {color};">
                                <h3>{outlet.upper()}</h3>
                                <div class="value" style="font-size:18px;">{row.get('known_label','?')}</div>
                                <div style="color:#8b949e; font-size:12px; margin-top:4px;">
                                    PLS: {row.get('partisan_lean_score', 0):+.3f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OUTLET PROFILE
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Outlet Deep Dive</div>', unsafe_allow_html=True)

    br = data.get('bias_report')
    if br is None:
        st.error('Run `python src/main.py` first to generate outlet metrics.')
    else:
        all_outlets = sorted(br.index.tolist())
        selected_outlet = st.selectbox('Select a news outlet:', all_outlets, index=0)

        row = br.loc[selected_outlet]
        known_label   = row.get('known_label', 'Unknown')
        pls           = row.get('partisan_lean_score', 0)
        entropy       = row.get('bias_entropy', 0)
        pei           = row.get('phrase_exclusivity_index', 0)
        color         = BIAS_COLORS.get(known_label, '#9ca3af')

        st.markdown(f'<div style="color:{color}; font-size:22px; font-weight:700; margin: 8px 0;">{selected_outlet.upper()} — {known_label}</div>',
                    unsafe_allow_html=True)

        # Metric cards row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card" style="border-top: 3px solid {color};">
                <h3>Partisan Lean Score</h3><div class="value">{pls:+.4f}</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <h3>Bias Entropy (bits)</h3><div class="value">{entropy:.2f}</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <h3>Phrase Exclusivity</h3><div class="value">{pei:.2%}</div></div>""",
                unsafe_allow_html=True)
        with c4:
            node_df = data.get('node_df')
            pr_val = node_df.loc[selected_outlet, 'pagerank'] if (node_df is not None and selected_outlet in node_df.index) else 0
            st.markdown(f"""<div class="metric-card">
                <h3>PageRank (Network)</h3><div class="value">{pr_val:.5f}</div></div>""",
                unsafe_allow_html=True)

        col_lor, col_pca = st.columns(2, gap='large')

        # Top distinctive phrases
        with col_lor:
            st.markdown('<div class="section-header">Most Distinctive Phrases (by Log-Odds)</div>',
                        unsafe_allow_html=True)
            lor = data.get('lor_df')
            matrix = data.get('matrix_df')
            if lor is not None and matrix is not None and selected_outlet in matrix.columns:
                outlet_lor = lor.copy()
                outlet_lor['outlet_count'] = matrix[selected_outlet]
                used = outlet_lor[outlet_lor['outlet_count'] > 0].sort_values(
                    'log_odds_ratio', ascending=False
                )
                top_r = used.head(8)
                top_l = used.tail(8)
                combined = pd.concat([top_r, top_l]).drop_duplicates()

                fig = go.Figure(go.Bar(
                    x=combined['log_odds_ratio'],
                    y=combined.index,
                    orientation='h',
                    marker=dict(
                        color=['#ef4444' if v > 0 else '#3b82f6' for v in combined['log_odds_ratio']],
                        line=dict(width=0),
                    ),
                ))
                fig.add_vline(x=0, line_color='#6b7280', line_width=1, line_dash='dash')
                fig.update_layout(
                    height=350, margin=dict(l=0, r=10, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(color='#8b949e', gridcolor='#30363d',
                               title=dict(text='Log-Odds (Right+ / Left-)')),
                    yaxis=dict(color='#c9d1d9', tickfont=dict(size=11)),
                    font=dict(color='#c9d1d9'),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('Phrase data not available.')

        # PCA position
        with col_pca:
            st.markdown('<div class="section-header">Position in PCA Space</div>',
                        unsafe_allow_html=True)
            pca_df = data.get('pca_df')
            if pca_df is not None:
                fig = go.Figure()
                for outlet_name, outlet_row in pca_df.iterrows():
                    is_selected = (outlet_name == selected_outlet)
                    ol = OUTLET_BIAS_LABELS.get(outlet_name, 'Unknown')
                    fig.add_trace(go.Scatter(
                        x=[outlet_row['PC1']], y=[outlet_row['PC2']],
                        mode='markers+text' if is_selected else 'markers',
                        text=[outlet_name] if is_selected else [],
                        textposition='top center',
                        marker=dict(
                            size=16 if is_selected else 8,
                            color=BIAS_COLORS.get(ol, '#9ca3af'),
                            line=dict(width=3 if is_selected else 0, color='white'),
                            opacity=1 if is_selected else 0.5,
                        ),
                        name=outlet_name if is_selected else '',
                        showlegend=is_selected,
                    ))
                fig.update_layout(
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
                    xaxis=dict(color='#8b949e', gridcolor='#30363d', title='PC1'),
                    yaxis=dict(color='#8b949e', gridcolor='#30363d', title='PC2'),
                    font=dict(color='#c9d1d9'),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Similar outlets
        st.markdown('<div class="section-header">Most Similar Outlets (Jaccard)</div>',
                    unsafe_allow_html=True)
        jaccard = data.get('jaccard_df')
        if jaccard is not None and selected_outlet in jaccard.index:
            sim_row = jaccard.loc[selected_outlet].drop(selected_outlet).sort_values(ascending=False).head(6)
            cols = st.columns(len(sim_row))
            for i, (out, score) in enumerate(sim_row.items()):
                lbl = OUTLET_BIAS_LABELS.get(out, 'Unknown')
                c = BIAS_COLORS.get(lbl, '#9ca3af')
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center; border-top: 2px solid {c};">
                        <h3>{lbl}</h3>
                        <div class="value" style="font-size:16px;">{out}</div>
                        <div style="color:#8b949e; font-size:12px; margin-top:6px;">
                            Jaccard: {score:.3f}
                        </div>
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)

    # Summary stats row
    br = data.get('bias_report')
    lor = data.get('lor_df')
    chi = data.get('chi_df')
    node_df = data.get('node_df')

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        n_outlets = len(br) if br is not None else 0
        st.markdown(f'<div class="metric-card"><h3>Outlets Analyzed</h3><div class="value">{n_outlets}</div></div>',
                    unsafe_allow_html=True)
    with s2:
        n_phrases = len(lor) if lor is not None else 0
        st.markdown(f'<div class="metric-card"><h3>Phrases Scored</h3><div class="value">{n_phrases:,}</div></div>',
                    unsafe_allow_html=True)
    with s3:
        n_sig = int(chi['significant'].sum()) if chi is not None and 'significant' in chi.columns else 0
        st.markdown(f'<div class="metric-card"><h3>Significant Phrases (p<0.05)</h3><div class="value">{n_sig:,}</div></div>',
                    unsafe_allow_html=True)
    with s4:
        n_edges = 0
        pairs = data.get('pairs_df')
        if pairs is not None: n_edges = len(pairs)
        st.markdown(f'<div class="metric-card"><h3>Outlet Pairs Analyzed</h3><div class="value">{n_edges:,}</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Chart selector
    chart_options = ['Partisan Lean Scores', 'PCA Scatter', 't-SNE Scatter',
                     'Log-Odds Ratio', 'Jaccard Similarity Matrix',
                     'Bias Entropy', 'Network Graph (PageRank)']
    selected_chart = st.selectbox('Choose chart:', chart_options)

    if selected_chart == 'Partisan Lean Scores' and br is not None:
        pls = br['partisan_lean_score'].sort_values()
        colors_list = [BIAS_COLORS.get(OUTLET_BIAS_LABELS.get(o, 'Unknown'), '#9ca3af') for o in pls.index]
        fig = go.Figure(go.Bar(
            x=pls.values, y=pls.index, orientation='h',
            marker=dict(color=colors_list, line=dict(width=0)),
        ))
        fig.add_vline(x=0, line_color='#6b7280', line_width=1.5, line_dash='dash')
        fig.update_layout(
            height=max(500, len(pls) * 20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
            xaxis=dict(color='#8b949e', gridcolor='#30363d',
                       title='Partisan Lean Score (Left- / Right+)'),
            yaxis=dict(color='#c9d1d9', tickfont=dict(size=10)),
            font=dict(color='#c9d1d9'),
            title=dict(text='Data-Driven Partisan Lean Score per Outlet', font=dict(size=15)),
            margin=dict(l=0, r=20, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_chart in ('PCA Scatter', 't-SNE Scatter'):
        df_key = 'pca_df' if selected_chart == 'PCA Scatter' else 'tsne_df'
        scatter_df = data.get(df_key)
        if scatter_df is not None:
            x_col = scatter_df.columns[0]
            y_col = scatter_df.columns[1]
            scatter_df = scatter_df.copy()
            scatter_df['outlet'] = scatter_df.index
            scatter_df['label'] = scatter_df['outlet'].map(lambda o: OUTLET_BIAS_LABELS.get(o, 'Unknown'))
            scatter_df['color'] = scatter_df['label'].map(lambda l: BIAS_COLORS.get(l, '#9ca3af'))
            fig = go.Figure()
            for lbl in ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right', 'Unknown']:
                sub = scatter_df[scatter_df['label'] == lbl]
                if sub.empty: continue
                fig.add_trace(go.Scatter(
                    x=sub[x_col], y=sub[y_col], mode='markers+text',
                    text=sub['outlet'], textposition='top center',
                    name=lbl,
                    marker=dict(size=10, color=BIAS_COLORS.get(lbl, '#9ca3af'),
                                line=dict(width=0.5, color='white')),
                    textfont=dict(size=8, color='#c9d1d9'),
                ))
            fig.update_layout(
                height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
                xaxis=dict(color='#8b949e', gridcolor='#30363d', title=x_col),
                yaxis=dict(color='#8b949e', gridcolor='#30363d', title=y_col),
                font=dict(color='#c9d1d9'), legend=dict(bgcolor='rgba(0,0,0,0)'),
                title=dict(text=selected_chart, font=dict(size=15)),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == 'Log-Odds Ratio' and lor is not None:
        top_n = st.slider('Top N phrases per side:', 10, 40, 20)
        top_r = lor.nlargest(top_n, 'log_odds_ratio')
        top_l = lor.nsmallest(top_n, 'log_odds_ratio')
        combined = pd.concat([top_l, top_r]).drop_duplicates().sort_values('log_odds_ratio')
        colors_list = ['#3b82f6' if v < 0 else '#ef4444' for v in combined['log_odds_ratio']]
        fig = go.Figure(go.Bar(
            x=combined['log_odds_ratio'], y=combined.index,
            orientation='h', marker=dict(color=colors_list, line=dict(width=0)),
        ))
        fig.add_vline(x=0, line_color='#6b7280', line_width=1.5, line_dash='dash')
        fig.update_layout(
            height=max(500, len(combined) * 22),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
            xaxis=dict(color='#8b949e', gridcolor='#30363d',
                       title='Log-Odds Ratio (Left- / Right+)'),
            yaxis=dict(color='#c9d1d9', tickfont=dict(size=10)),
            font=dict(color='#c9d1d9'),
            title=dict(text='Most Partisan Phrases by Log-Odds Ratio', font=dict(size=15)),
            margin=dict(l=0, r=20, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == 'Jaccard Similarity Matrix':
        jaccard = data.get('jaccard_df')
        if jaccard is not None:
            bias_order = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']
            ordered = sorted(
                [o for o in OUTLET_BIAS_LABELS if o in jaccard.index],
                key=lambda x: bias_order.index(OUTLET_BIAS_LABELS[x])
                if OUTLET_BIAS_LABELS.get(x) in bias_order else 2
            )
            remaining = [o for o in jaccard.index if o not in ordered]
            all_out = ordered + remaining
            sub = jaccard.loc[all_out, all_out]
            np.fill_diagonal(sub.values, 0)
            fig = go.Figure(go.Heatmap(
                z=sub.values, x=sub.columns, y=sub.index,
                colorscale='YlOrRd', colorbar=dict(title='Jaccard'),
            ))
            fig.update_layout(
                height=700, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
                xaxis=dict(tickfont=dict(size=8), color='#8b949e', tickangle=75),
                yaxis=dict(tickfont=dict(size=8), color='#8b949e'),
                font=dict(color='#c9d1d9'),
                title=dict(text='Outlet Pairwise Jaccard Similarity', font=dict(size=15)),
                margin=dict(l=0, r=0, t=40, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == 'Bias Entropy' and br is not None:
        ent = br['bias_entropy'].sort_values(ascending=False)
        colors_list = [BIAS_COLORS.get(OUTLET_BIAS_LABELS.get(o, 'Unknown'), '#9ca3af') for o in ent.index]
        fig = go.Figure(go.Bar(
            x=ent.index, y=ent.values,
            marker=dict(color=colors_list, line=dict(width=0)),
        ))
        fig.update_layout(
            height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
            xaxis=dict(tickangle=75, tickfont=dict(size=8), color='#8b949e'),
            yaxis=dict(color='#8b949e', gridcolor='#30363d', title='Shannon Entropy (bits)'),
            font=dict(color='#c9d1d9'),
            title=dict(text='Phrase Distribution Entropy per Outlet', font=dict(size=15)),
            margin=dict(l=0, r=0, t=40, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_chart == 'Network Graph (PageRank)' and node_df is not None:
        fig = go.Figure(go.Bar(
            x=node_df['pagerank'].sort_values(ascending=False).values,
            y=node_df['pagerank'].sort_values(ascending=False).index,
            orientation='h',
            marker=dict(
                color=[BIAS_COLORS.get(OUTLET_BIAS_LABELS.get(o, 'Unknown'), '#9ca3af')
                       for o in node_df['pagerank'].sort_values(ascending=False).index],
                line=dict(width=0),
            ),
        ))
        fig.update_layout(
            height=max(500, len(node_df) * 18),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
            xaxis=dict(color='#8b949e', gridcolor='#30363d', title='PageRank Score'),
            yaxis=dict(color='#c9d1d9', tickfont=dict(size=10)),
            font=dict(color='#c9d1d9'),
            title=dict(text='Outlet Influence (PageRank in Co-occurrence Network)', font=dict(size=15)),
            margin=dict(l=0, r=20, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Download section ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Download Data</div>', unsafe_allow_html=True)
    dl_cols = st.columns(4)
    files = [
        ('outlet_bias_report.csv',        'Bias Report'),
        ('log_odds_ratios.csv',           'Log-Odds Ratios'),
        ('jaccard_similarity_matrix.csv', 'Jaccard Matrix'),
        ('chi_square_results.csv',        'Chi-Square Tests'),
    ]
    for i, (fname, label) in enumerate(files):
        fpath = os.path.join(PROCESSED_DIR, fname)
        with dl_cols[i % 4]:
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    st.download_button(label=f'Download {label}', data=f,
                                       file_name=fname, mime='text/csv',
                                       use_container_width=True)
            else:
                st.button(f'{label} (not ready)', disabled=True, use_container_width=True)
