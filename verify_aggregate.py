"""Verify archived aggregate estimates without accessing participant data."""
from pathlib import Path
import csv
import math

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / 'aggregate' / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b, label, tol=1e-7):
    if not math.isclose(float(a), float(b), abs_tol=tol, rel_tol=tol):
        raise ValueError(f'{label}: {a} differs from {b}')


cohorts = read('core_refreshed_cohort.csv')
pooled = read('core_refreshed_meta.csv')
for row in pooled:
    inputs = [r for r in cohorts if (r['model'], r['effect']) == (row['model'], row['effect'])]
    if len(inputs) != 3:
        raise ValueError('Expected three independent cohort estimates')
    weights = [1 / float(r['se']) ** 2 for r in inputs]
    fixed = sum(w * float(r['estimate']) for w, r in zip(weights, inputs)) / sum(weights)
    se = math.sqrt(1 / sum(weights))
    close(fixed, row['fixed'], 'Fixed-effect estimate')
    close(fixed - 1.959963984540054 * se, row['fixed_lower'], 'Fixed lower CI')
    close(fixed + 1.959963984540054 * se, row['fixed_upper'], 'Fixed upper CI')

domains = read('domain_refreshed_meta.csv')
for method in ['fixed', 'random']:
    ranked = sorted(domains, key=lambda r: float(r[method + '_p']))
    previous = 1.0
    for rank in range(len(ranked), 0, -1):
        row = ranked[rank - 1]
        previous = min(previous, float(row[method + '_p']) * len(ranked) / rank)
        close(previous, row[method + '_BH_p'], 'BH-adjusted p value')

for row in read('model_paired_differences.csv'):
    matched = [r for r in read('model_contrasts.csv') if r['cohort'] == row['cohort'] and r['contrast'] == 'standardized_B_minus_A']
    spline = next(r for r in matched if r['model'] == row['model'])
    categorical = next(r for r in matched if r['model'] == 'categorical')
    close(float(spline['estimate']) - float(categorical['estimate']), row['estimate'], 'Paired contrast point estimate')

print('PASS: 9 fixed-effect estimates and CIs, 20 BH values, 12 paired point contrasts.')
print('Scope: aggregate arithmetic only. No participant-level rerun or random-effects implementation validation.')
