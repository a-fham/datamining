import sys, os
sys.path.insert(0, '.')
from src.data.article_loader import load_articles
from src.models.trainer import train, predict_single
from src.models.explainer import save_lr_for_explanation, get_top_words

print('Loading articles...')
df = load_articles(max_per_label=8000)
print()
print('Training ensemble...')
results = train(df, verbose=True)
model = results['pipeline']

# Save LR explainer
try:
    save_lr_for_explanation(model.models[0])
    print('LR explainer saved.')
except Exception as e:
    print(f'Could not save explainer: {e}')

print()
print('=== FRAMING TEST ===')
tests = [
    ('unjustified misgendering of LGBT community leads to massive protests', 'Left'),
    ('Trump says to kick out all immigrants',                               'Right'),
    ('unjustified killing of immigrants leads to panic',                    'Left'),
    ('migrants fleeing violence deserve protection',                        'Left'),
    ('illegal aliens flooding the southern border',                         'Right'),
    ('anti-transgender legislation attacks trans dignity',                  'Left'),
    ('federal reserve holds interest rates steady',                         'Center'),
    ('antifa rioters destroy businesses in Democrat cities',                'Right'),
    ('pride protests demand equal rights for queer community',              'Left'),
    ('the company reported quarterly earnings above expectations',           'Center'),
    ('children separated from families at border suffer lasting trauma',    'Left'),
    ('misgendering transgender people is harassment and discrimination',     'Left'),
]
for text, expected in tests:
    r      = predict_single(text, model)
    status = 'OK   ' if r['label'] == expected else 'WRONG'
    p      = r['probabilities']
    print(f'[{status}] {text[:55]:55s} -> {r["label"]} (exp {expected})')
    print(f'         L:{p.get("Left",0):.2f}  C:{p.get("Center",0):.2f}  R:{p.get("Right",0):.2f}')

# Also test explainer filtering
print()
print('=== EXPLAINER TEST (no vader_ tokens should appear) ===')
expl = get_top_words('unjustified misgendering of LGBT community leads to massive protests')
print('Left words:', [w for w, s in expl.get('top_left_words',[])])
print('Right words:', [w for w, s in expl.get('top_right_words',[])])
