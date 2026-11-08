import argparse
import hashlib
import io
import json
import platform
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import train_test_split

from shopper_models import (ROOT, FEATURES, TARGET, METRICS, MODEL_FILES,
                            candidate_parameters, evaluate_pipeline,
                            make_pipeline, normalise_frame)

SOURCE = 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--execution-location', default='Local reference run; not BITS Virtual Lab evidence')
    args = parser.parse_args()
    raw_path = ROOT / 'data/online_shoppers_intention.csv'
    raw_path.parent.mkdir(exist_ok=True)
    if not raw_path.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        raw_path.write_bytes(archive.read('online_shoppers_intention.csv'))
    raw = pd.read_csv(raw_path)
    x_raw, y_raw = normalise_frame(raw)
    usable = x_raw.assign(Revenue=y_raw).drop_duplicates().reset_index(drop=True)
    x, y = normalise_frame(usable)
    if x.shape[1] < 12 or len(x) < 500:
        raise ValueError('Dataset does not satisfy the assignment size requirements.')
    if x.duplicated().any():
        raise ValueError('Identical features have conflicting labels; use a grouped split.')
    indices = np.arange(len(usable))
    build_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y,
                                          random_state=args.seed)
    train_idx, val_idx = train_test_split(build_idx, test_size=0.25,
                                         stratify=y.iloc[build_idx], random_state=args.seed)
    for split_name, ids in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        path = ROOT / ('test_data.csv' if split_name == 'test' else f'data/{split_name}_data.csv')
        usable.iloc[ids].to_csv(path, index=False)
    test_set, val_set = set(test_idx), set(val_idx)
    pd.DataFrame({'clean_row': indices, 'split': [
        'test' if i in test_set else 'validation' if i in val_set else 'train'
        for i in indices]}).to_csv(ROOT / 'results/split_manifest.csv', index=False)
    candidates, chosen = [], {}
    for name, grid in candidate_parameters().items():
        for params in grid:
            start = time.perf_counter()
            pipeline = make_pipeline(name, args.seed, params)
            pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
            result = evaluate_pipeline(pipeline, x.iloc[val_idx], y.iloc[val_idx])[0]
            entry = {'Model': name, 'parameters': params, **result,
                     'fit_and_validation_seconds': time.perf_counter() - start}
            candidates.append(entry)
            print(name, params, 'validation MCC', round(result['MCC'], 5), flush=True)
        ranked = sorted([r for r in candidates if r['Model'] == name],
                        key=lambda r: (r['MCC'], r['AUC']), reverse=True)
        chosen[name] = ranked[0]
    # Decide the overall candidate before touching held-out test outcomes.
    winner = max(chosen, key=lambda n: (chosen[n]['MCC'], chosen[n]['AUC']))
    results, matrices, reports, predictions = [], {}, {}, usable.iloc[test_idx].copy()
    final_parameters = {}
    for name, entry in chosen.items():
        model = make_pipeline(name, args.seed, entry['parameters'])
        model.fit(x.iloc[build_idx], y.iloc[build_idx])
        joblib.dump(model, ROOT / 'model' / MODEL_FILES[name], compress=3)
        metrics, matrix, report, predicted, score, _ = evaluate_pipeline(
            model, x.iloc[test_idx], y.iloc[test_idx])
        results.append({'Model': name, **metrics})
        matrices[name], reports[name] = matrix.tolist(), report
        predictions[name + ' prediction'] = predicted
        predictions[name + ' score'] = score
        final_parameters[name] = entry['parameters']
    pd.DataFrame(results).to_csv(ROOT / 'results/test_metrics.csv', index=False)
    pd.DataFrame(candidates).to_csv(ROOT / 'results/validation_search.csv', index=False)
    predictions.to_csv(ROOT / 'results/test_predictions.csv', index=False)
    (ROOT / 'results/confusion_matrices.json').write_text(json.dumps(matrices, indent=2))
    (ROOT / 'results/classification_reports.json').write_text(json.dumps(reports, indent=2))
    metadata = {
        'reference_run_utc': datetime.now(timezone.utc).isoformat(),
        'execution_location': args.execution_location,
        'source_url': SOURCE,
        'source_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'seed': args.seed, 'raw_rows': len(raw), 'clean_rows': len(usable),
        'duplicates_removed': len(raw) - len(usable), 'features': FEATURES,
        'excluded_feature': 'PageValues', 'positive_class': 'Revenue=1 (purchase)',
        'class_counts': {str(k): int(v) for k, v in y.value_counts().items()},
        'splits': {n: {'rows': len(ids), 'purchases': int(y.iloc[ids].sum())}
                   for n, ids in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]},
        'validation_winner': winner, 'selection_rule': 'Validation MCC, then validation AUC',
        'chosen_parameters': final_parameters,
        'versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'joblib': joblib.__version__},
    }
    (ROOT / 'results/experiment.json').write_text(json.dumps(metadata, indent=2))
    print('\nHeld-out results\n' + pd.DataFrame(results).to_string(index=False))
    print('\nValidation-selected candidate:', winner)


if __name__ == '__main__':
    main()
