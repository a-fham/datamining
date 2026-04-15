import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'phrasebias_data')
PHRASE_SELECTION_DIR = os.path.join(DATA_DIR, 'phrase_selection')
PHRASE_COUNTS_DIR = os.path.join(DATA_DIR, 'phrase_counts')
BLACKLIST_PATH = os.path.join(DATA_DIR, 'blacklist.csv')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

TOPICS = [
    'abortion', 'affirmative_action', 'blm', 'china', 'church_state',
    'climate', 'espionage', 'guns', 'human_rights', 'islam', 'israel',
    'judaism', 'military_spending', 'native_americans', 'nuclear_weapons',
    'oil', 'palestine', 'police', 'prisons', 'private_finance',
    'public_finance', 'russia', 'sexual_harassment', 'tech_censorship',
    'universities', 'us_immigration', 'venezuela', 'yemen'
]

NEWS_OUTLETS = [
    'aljazeera', 'alternet', 'americanconservative', 'americanspectator', 'ap',
    'atlantic', 'bbc', 'breitbart', 'buzzfeed', 'cbs', 'cnn', 'commondreams',
    'conversation', 'counterpunch', 'dailycaller', 'dailykos', 'dailymail',
    'dailywire', 'economist', 'federalist', 'fox', 'guardian', 'huffingtonpost',
    'infowars', 'intercept', 'jacobinmag', 'motherjones', 'nationalreview',
    'nbc', 'npr', 'nypost', 'nytimes', 'pbs', 'pjmedia', 'rawstory', 'redstate',
    'rt', 'slate', 'spectator', 'townhall', 'truthdig', 'usatoday', 'vice', 'vox',
    'wapo', 'wsj'
]

OUTLET_BIAS_LABELS = {
    'breitbart': 'Right',
    'dailywire': 'Right',
    'fox': 'Right',
    'nypost': 'Right',
    'wsj': 'Right',
    'nationalreview': 'Right',
    'redstate': 'Right',
    'federalist': 'Right',
    'townhall': 'Right',
    'dailycaller': 'Right',
    'alternet': 'Left',
    'huffingtonpost': 'Left',
    'motherjones': 'Left',
    'commondreams': 'Left',
    'slate': 'Left',
    'vox': 'Left',
    'dailykos': 'Left',
    'jacobinmag': 'Left',
    'intercept': 'Left',
    'nytimes': 'Center-Left',
    'wapo': 'Center-Left',
    'cnn': 'Center',
    'nbc': 'Center',
    'abc': 'Center',
    'cbs': 'Center',
    'npr': 'Center',
    'pbs': 'Center',
    'ap': 'Center',
    'reuters': 'Center',
    'atlantic': 'Center',
    'economist': 'Center',
    'guardian': 'Center-Left',
    'bbc': 'Center',
    'aljazeera': 'Center',
    'buzzfeed': 'Center-Left',
    'vice': 'Center-Left',
    'rawstory': 'Left',
    'truthdig': 'Left',
    'counterpunch': 'Left',
    'americanconservative': 'Right',
    'americanspectator': 'Right',
    'pjmedia': 'Right',
    'spectator': 'Center-Right',
    'infowars': 'Right',
    'rt': 'Right'
}
