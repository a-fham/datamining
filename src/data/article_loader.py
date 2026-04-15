"""
Article Loader
==============
Smart sampler for the large All-The-News dataset (~3.4 GB CSV).
Maps publication names to Left/Center/Right bias labels and
samples a balanced subset for classifier training.

Expected file: data/raw/all-the-news-2-1.csv  OR  all-the-news.csv
(place any CSV from the David McKinley Kaggle dataset here)

Columns used: 'title', 'article', 'publication'
"""

import os
import re
import pandas as pd
import numpy as np

# ── Publication → 3-class bias label ──────────────────────────────────────────
# Source: AllSides.com + MBFC ratings, collapsed to 3 classes for clarity.
PUBLICATION_BIAS = {
    # Left
    'Buzzfeed News':         'Left',
    'Atlantic':              'Left',
    'New Yorker':            'Left',
    'Vox':                   'Left',
    'Talking Points Memo':   'Left',
    'Hyperallergic':         'Left',
    'People':                'Left',
    'Guardian':              'Left',
    'Vice':                  'Left',
    'Refinery 29':           'Left',
    'Jacobin':               'Left',
    'Intercept':             'Left',
    'Mother Jones':          'Left',
    'The Intercept':         'Left',
    'Washington Post':       'Left',    # AllSides 2024: Lean Left
    'New York Times':        'Left',    # AllSides 2024: Lean Left
    'CNN':                   'Left',    # AllSides 2024: Lean Left
    'MSNBC':                 'Left',    # AllSides 2024: Left
    'HuffPost':              'Left',
    'Huffington Post':       'Left',
    'Slate':                 'Left',
    'Rolling Stone':         'Left',

    # Center
    'Reuters':               'Center',
    'Associated Press':      'Center',
    'NPR':                   'Center',
    'CBS News':              'Center',
    'ABC News':              'Center',
    'NBC News':              'Center',
    'USA Today':             'Center',
    'Business Insider':      'Center',
    'CNBC':                  'Center',
    'Forbes':                'Center',
    'Hill':                  'Center',
    'Politico':              'Center',
    'Economist':             'Center',
    'BBC News':              'Center',
    'Bloomberg':             'Center',
    'Axios':                 'Center',
    'National Review':       'Center',

    # Right
    'Breitbart':             'Right',
    'Fox News':              'Right',
    'New York Post':         'Right',
    'Daily Mail':            'Right',
    'Daily Caller':          'Right',
    'Daily Wire':            'Right',
    'Federalist':            'Right',
    'Townhall':              'Right',
    'Washington Examiner':   'Right',
    'American Spectator':    'Right',
    'Infowars':              'Right',
    'Western Journal':       'Right',
    'PJ Media':              'Right',
    'Red State':             'Right',
    'New York Sun':          'Right',
}

LABEL_ORDER = ['Left', 'Center', 'Right']

BASE_DIR      = os.path.join(os.path.dirname(__file__), '..', '..')
RAW_DATA_DIR  = os.path.join(BASE_DIR, 'data', 'raw')


def _curated_framing_examples() -> pd.DataFrame:
    """
    ~120 hand-crafted examples covering framing patterns that TF-IDF
    bag-of-words alone cannot distinguish from topic alone.

    These teach the model:
      - LEFT:   sympathy for immigrants, criticism of violence/enforcement
      - RIGHT:  anti-immigration, border security, enforcement framing
      - CENTER: neutral policy/economic reporting
    """
    examples = [
        # ── LEFT: sympathy for immigrants / criticism of violence ─────────────
        ("unjustified killing of immigrants leads to panic among communities", "Left"),
        ("migrants fleeing violence deserve protection and safe harbor", "Left"),
        ("the brutal treatment of asylum seekers at the border is unconscionable", "Left"),
        ("undocumented immigrants are human beings who deserve dignity and rights", "Left"),
        ("children separated from families at the border suffer lasting trauma", "Left"),
        ("the cruelty of deportation policies tears apart families and communities", "Left"),
        ("immigrants enrich our culture and economy and deserve a path to citizenship", "Left"),
        ("refugee families escaping war need compassion not criminalization", "Left"),
        ("the targeting of immigrant communities by ICE causes widespread fear", "Left"),
        ("white supremacist violence against minorities is rising and must be condemned", "Left"),
        ("police brutality against black communities demands immediate accountability", "Left"),
        ("gun violence kills thousands every year and republicans refuse to act", "Left"),
        ("the murder of journalists and activists must be investigated and punished", "Left"),
        ("climate change is an existential crisis fueled by corporate greed", "Left"),
        ("healthcare is a human right not a privilege for the wealthy", "Left"),
        ("income inequality has reached historic levels while the rich get richer", "Left"),
        ("voter suppression laws disproportionately harm black and brown voters", "Left"),
        ("fossil fuel companies knew about climate change and deliberately misled the public", "Left"),
        ("the killing of unarmed black men by police demands systemic reform", "Left"),
        ("wage theft by corporations harms millions of low-income workers", "Left"),
        ("indigenous communities face ongoing discrimination and land rights violations", "Left"),
        ("activists demand justice for victims of state-sponsored violence", "Left"),
        ("the humanitarian crisis at the border demands compassion not cruelty", "Left"),
        ("corporate lobbying corrupts democracy and suppresses workers' rights", "Left"),
        ("reproductive rights are under attack by radical conservative lawmakers", "Left"),
        ("trans rights are human rights and must be protected by law", "Left"),
        ("systemic racism embedded in the criminal justice system must be dismantled", "Left"),
        ("the wealth gap between the ultra-rich and working poor is unsustainable", "Left"),
        ("medicaid cuts will devastate the most vulnerable Americans", "Left"),
        ("senator calls for investigation into police killings of unarmed civilians", "Left"),

        # ── LEFT: LGBT / trans / misgendering / protest framing ──────────────
        ("unjustified misgendering of LGBT community leads to massive protests", "Left"),
        ("misgendering transgender individuals is a form of discrimination and harassment", "Left"),
        ("trans rights activists protest against bathroom bills that target their community", "Left"),
        ("the misgendering of non-binary people causes serious psychological harm", "Left"),
        ("LGBT youth face higher rates of depression and suicide due to discrimination", "Left"),
        ("pride protests erupted across cities demanding equal rights for the queer community", "Left"),
        ("anti-transgender legislation is an attack on the dignity of trans people", "Left"),
        ("conversion therapy is harmful pseudoscience that must be banned immediately", "Left"),
        ("drag performers and trans activists rally against discriminatory state laws", "Left"),
        ("gay marriage equality is a fundamental right that must be protected", "Left"),
        ("homophobia and transphobia in schools harm LGBTQ students and must be addressed", "Left"),
        ("protests erupted after police brutality against peaceful demonstrators", "Left"),
        ("thousands marched in solidarity demanding an end to racial injustice", "Left"),
        ("demonstrators took to the streets to protest against gun violence inaction", "Left"),
        ("massive rallies called for action on climate change and environmental justice", "Left"),
        ("activists protest corporate greed and demand living wages for workers", "Left"),
        ("protesters demand accountability for police killings of unarmed civilians", "Left"),
        ("community organizers lead peaceful demonstrations for housing rights", "Left"),
        ("student protesters demand universities divest from fossil fuels", "Left"),
        ("women marched in protest against restrictions on reproductive rights", "Left"),
        ("indigenous activists protest pipeline construction through sacred tribal lands", "Left"),
        ("disability rights advocates protest cuts to medicaid and social services", "Left"),

        # ── RIGHT: covering protests as threats / disorder ────────────────────
        ("antifa rioters clashed with police leaving a trail of destruction", "Right"),
        ("BLM protesters blocked highways and burned businesses in Democrat cities", "Right"),
        ("radical leftists riot and loot under the guise of peaceful protest", "Right"),
        ("out-of-control mobs destroy property while liberal mayors order police to stand down", "Right"),
        ("George Soros funds radical protest groups that attack American values", "Right"),
        ("leftist mob culture threatens law and order across Democrat-run cities", "Right"),

        ("Trump says to kick out all illegal immigrants and secure the border", "Right"),
        ("illegal aliens are flooding across the southern border overwhelming agents", "Right"),
        ("the open border policy is a national security threat and must be stopped", "Right"),
        ("democrats refuse to enforce immigration laws allowing criminals to roam free", "Right"),
        ("sanctuary cities protect criminal illegal immigrants from deportation", "Right"),
        ("the border invasion must be stopped to protect American citizens and jobs", "Right"),
        ("illegal immigrants are taking American jobs and draining public resources", "Right"),
        ("gang members and drug traffickers are exploiting weak border security", "Right"),
        ("radical left wants open borders and to abolish ICE endangering Americans", "Right"),
        ("the Biden amnesty plan will reward illegal behavior and invite more invasion", "Right"),
        ("unvetted migrants bring disease and crime say border patrol agents", "Right"),
        ("critical race theory indoctrinates children with anti-American propaganda", "Right"),
        ("gun rights are under attack by radical leftist anti-second amendment Democrats", "Right"),
        ("the mainstream media is lying to Americans and covering up the truth", "Right"),
        ("big tech censors conservatives and suppresses free speech", "Right"),
        ("woke ideology is destroying American values and traditional culture", "Right"),
        ("parents must stop radical gender ideology from being taught in schools", "Right"),
        ("socialist Democrats want to defund police and destroy public safety", "Right"),
        ("election fraud stole the 2020 election and must be investigated", "Right"),
        ("religious liberty is under attack by the radical secular left", "Right"),
        ("abortion is murder and Democrats support killing unborn babies", "Right"),
        ("the deep state is working to undermine the will of the American people", "Right"),
        ("antifa terrorists are attacking innocent Americans with impunity", "Right"),
        ("rampant crime in Democrat-run cities shows liberalism has failed Americans", "Right"),
        ("the radical left is pushing socialism that will destroy the American economy", "Right"),
        ("energy independence was destroyed by Biden's anti-fossil fuel agenda", "Right"),
        ("inflation is crushing American families because of reckless Democrat spending", "Right"),
        ("the fentanyl crisis is caused by Biden's open border policy", "Right"),
        ("China is exploiting weak Democrats who refuse to stand up for America", "Right"),
        ("cancel culture silences conservatives and destroys free expression", "Right"),

        # ── CENTER: neutral policy, economic, factual reporting ───────────────
        ("the federal reserve raised interest rates by 25 basis points on Wednesday", "Center"),
        ("congress passed a bipartisan infrastructure bill with broad support", "Center"),
        ("the unemployment rate fell to 3.7 percent according to the bureau of labor statistics", "Center"),
        ("the supreme court ruled 6-3 on the landmark environmental case", "Center"),
        ("the commerce department reported GDP growth of 2.1 percent last quarter", "Center"),
        ("both parties have agreed to extend the debt ceiling through next year", "Center"),
        ("the state department issued a travel advisory for the region", "Center"),
        ("nato leaders met in Brussels to discuss defense spending commitments", "Center"),
        ("the world health organization declared the outbreak a public health emergency", "Center"),
        ("oil prices fell after OPEC announced it would increase production", "Center"),
        ("the company reported quarterly earnings above analyst expectations", "Center"),
        ("lawmakers on both sides of the aisle called for an independent investigation", "Center"),
        ("the trade deficit narrowed as exports increased and imports declined", "Center"),
        ("central banks worldwide are grappling with persistent inflationary pressures", "Center"),
        ("the president signed the executive order authorizing the review", "Center"),
        ("scientists published new research on the effectiveness of the vaccine", "Center"),
        ("the census bureau released new population data showing demographic shifts", "Center"),
        ("negotiations between the two countries resumed after a diplomatic breakthrough", "Center"),
        ("the pentagon confirmed the military exercise would proceed as scheduled", "Center"),
        ("the treasury department announced new sanctions targeting the regime", "Center"),
        ("regulators approved the merger after reviewing antitrust concerns", "Center"),
        ("local officials declared a state of emergency following the natural disaster", "Center"),
        ("the committee voted to advance the legislation to the full senate floor", "Center"),
        ("economists project moderate growth over the next fiscal year", "Center"),
        ("the administration outlined new guidelines for the regulatory framework", "Center"),
        ("technology companies reported record profits driven by cloud computing demand", "Center"),
        ("housing starts increased for the third consecutive month", "Center"),
        ("the international monetary fund revised its global growth forecast downward", "Center"),
        ("consumer confidence index rose to its highest level in six months", "Center"),
        ("the federal budget deficit increased to 1.7 trillion dollars this year", "Center"),
    ]

    rows = [{'text': t, 'label': l, 'publication': 'curated'} for t, l in examples]
    df = pd.DataFrame(rows)
    # Repeat 3x to give them meaningful weight against 23k real articles
    return pd.concat([df] * 3, ignore_index=True)

def _find_csv() -> str:
    """Find the first CSV in data/raw/ that looks like the All-The-News dataset."""
    candidates = [
        'all-the-news-2-1.csv',
        'all-the-news.csv',
        'articles1.csv',
        'articles2.csv',
        'articles3.csv',
    ]
    for name in candidates:
        path = os.path.join(RAW_DATA_DIR, name)
        if os.path.exists(path):
            return path

    # Fall back: first CSV in raw dir
    for f in os.listdir(RAW_DATA_DIR):
        if f.endswith('.csv'):
            return os.path.join(RAW_DATA_DIR, f)

    raise FileNotFoundError(
        f"No CSV found in {RAW_DATA_DIR}. "
        "Please place the All-The-News CSV there."
    )


def clean_text(text: str) -> str:
    """Minimal cleaning: collapse whitespace, strip URLs and email addresses."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_articles(
    max_per_label: int = 8000,
    use_title_only: bool = False,
    min_article_len: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and balance the All-The-News dataset.

    Parameters
    ----------
    max_per_label : int
        Maximum articles per bias class (Left / Center / Right).
        Total dataset size will be at most 3 * max_per_label.
    use_title_only : bool
        If True, use only the headline title (faster, smaller features).
        If False, use first 500 chars of article body + title (more signal).
    min_article_len : int
        Drop articles whose body text is shorter than this.
    random_state : int
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame with columns: ['text', 'label', 'publication']
    """
    csv_path = _find_csv()
    print(f"  Loading articles from: {os.path.basename(csv_path)}")

    # Read in chunks to avoid OOM on 3 GB file
    chunks = []
    chunksize = 50_000
    needed_cols_options = [
        ['title', 'article', 'publication'],
        ['title', 'content', 'publication'],
        ['headline', 'article', 'outlet'],
    ]

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        on_bad_lines='skip',
        low_memory=False,
        encoding='utf-8',
        encoding_errors='replace',
    )

    label_counts = {l: 0 for l in LABEL_ORDER}
    total_needed = max_per_label * len(LABEL_ORDER)

    for chunk in reader:
        # Normalise column names
        chunk.columns = [c.lower().strip() for c in chunk.columns]

        # Find publication / title / body columns
        pub_col = next((c for c in ['publication', 'outlet', 'source'] if c in chunk.columns), None)
        title_col = next((c for c in ['title', 'headline'] if c in chunk.columns), None)
        body_col = next((c for c in ['article', 'content', 'body', 'text'] if c in chunk.columns), None)

        if pub_col is None or title_col is None:
            continue

        chunk['bias_label'] = chunk[pub_col].map(PUBLICATION_BIAS)
        chunk = chunk.dropna(subset=['bias_label'])

        if body_col:
            chunk['body'] = chunk[body_col].fillna('').astype(str)
            chunk = chunk[chunk['body'].str.len() >= min_article_len]

        if chunk.empty:
            continue

        # Build text field
        if use_title_only or body_col is None:
            chunk['text'] = chunk[title_col].fillna('').apply(clean_text)
        else:
            chunk['text'] = (
                chunk[title_col].fillna('').astype(str) + ' ' +
                chunk[body_col].fillna('').astype(str).str[:600]
            ).apply(clean_text)

        chunk = chunk[chunk['text'].str.len() > 20]

        # Sample per label to stay balanced
        for label in LABEL_ORDER:
            still_need = max_per_label - label_counts[label]
            if still_need <= 0:
                continue
            sub = chunk[chunk['bias_label'] == label]
            if sub.empty:
                continue
            take = min(len(sub), still_need)
            sub = sub.sample(n=take, random_state=random_state)
            chunks.append(sub[['text', 'bias_label', pub_col]].rename(
                columns={'bias_label': 'label', pub_col: 'publication'}
            ))
            label_counts[label] += take

        if sum(label_counts.values()) >= total_needed:
            break

    if not chunks:
        raise ValueError(
            "Could not load any labeled articles. "
            "Check that your CSV has 'publication', 'title', and 'article' columns, "
            "and that publication names match PUBLICATION_BIAS."
        )

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates('text').reset_index(drop=True)
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 20]

    print(f"  Articles loaded per label:")
    for label, count in df['label'].value_counts().items():
        print(f"    {label:8s}: {count:,}")

    # Inject curated framing examples (teach Left framing for immigrant sympathy etc.)
    curated = _curated_framing_examples()
    df = pd.concat([df, curated], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    print(f"  + {len(curated)} curated framing examples added")
    print(f"  Total: {len(df):,} articles")

    return df
