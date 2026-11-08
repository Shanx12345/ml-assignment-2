"""Check split isolation, independent metric arithmetic, serialization and UI execution."""
import io
import json
import math
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from shopper_models import ROOT, MODEL_FILES, FEATURES, evaluate_pipeline, normalise_frame


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def must_reject(frame, fragment):
    try:
        normalise_frame(frame)
    except ValueError as exc:
        check(fragment in str(exc), str(exc))
    else:
        raise AssertionError('Invalid data was accepted: ' + fragment)


def main():
    tests = pd.read_csv(ROOT / 'test_data.csv')
    training = pd.read_csv(ROOT / 'data/train_data.csv')
    validation = pd.read_csv(ROOT / 'data/validation_data.csv')
    x, y = normalise_frame(tests)
    signatures = []
    for split in [training, validation, tests]:
        sx, _ = normalise_frame(split)
        signatures.append(set(pd.util.hash_pandas_object(sx, index=False)))
    check(not signatures[0] & signatures[1], 'Training and validation overlap.')
    check(not signatures[0] & signatures[2], 'Training and test overlap.')
    check(not signatures[1] & signatures[2], 'Validation and test overlap.')
    saved = pd.read_csv(ROOT / 'results/test_metrics.csv').set_index('Model')
    checks = []
    for name, file in MODEL_FILES.items():
        model = joblib.load(ROOT / 'model' / file)
        got, matrix, _, prediction, score, _ = evaluate_pipeline(model, x, y)
        tn, fp, fn, tp = map(int, matrix.ravel())
        expected = {
            'Accuracy': (tp + tn) / len(y),
            'Precision': tp / (tp + fp) if tp + fp else 0,
            'Recall': tp / (tp + fn) if tp + fn else 0,
            'F1': 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0,
            'MCC': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        }
        # Mann-Whitney rank expression is independent of roc_auc_score.
        ranks = pd.Series(score).rank(method='average').to_numpy()
        n_pos, n_neg = int(y.sum()), int((1-y).sum())
        expected['AUC'] = (ranks[y.to_numpy() == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        for key, value in expected.items():
            check(abs(got[key] - value) < 1e-12, name + ' independent formula mismatch: ' + key)
            check(abs(saved.loc[name, key] - value) < 1e-12, name + ' saved metric mismatch: ' + key)
        one_class = tests[tests.Revenue == 0].head(20)
        ox, oy = normalise_frame(one_class)
        check(evaluate_pipeline(model, ox, oy)[0]['AUC'] is None, 'One-class AUC must be N/D.')
        # A second read of a saved model must produce the same predictions.
        check(np.array_equal(prediction, joblib.load(ROOT/'model'/file).predict(x)), 'Serialization drift.')
        checks.append(name + ': six metrics, one-class AUC and saved predictions agree')
    must_reject(tests.drop(columns='Revenue'), 'Missing required columns')
    must_reject(tests.drop(columns=FEATURES[0]), 'Missing required columns')
    must_reject(tests.assign(Revenue='maybe'), 'Revenue:')
    must_reject(tests.assign(BounceRates=np.inf), 'finite')
    must_reject(tests.assign(BounceRates=1.5), 'between 0 and 1')
    must_reject(tests.assign(ProductRelated='broken'), 'non-numeric')
    must_reject(tests.iloc[:0], 'no data rows')
    # UI must evaluate both bundled and uploaded CSVs without invoking fit.
    from streamlit.testing.v1 import AppTest
    with patch.object(Pipeline, 'fit', side_effect=AssertionError('App attempted retraining')):
        at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
        check(not at.exception, str(at.exception))
        check(len(at.dataframe[0].value) == 5, 'Missing all-model comparison.')
        for name in MODEL_FILES:
            at.sidebar.selectbox[0].select(name).run()
            check(not at.exception, name + ': UI failed')
        custom = tests.iloc[:80].to_csv(index=False).encode()
        with patch('streamlit.file_uploader', return_value=io.BytesIO(custom)):
            at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
            check(not at.exception, str(at.exception))
            check(at.metric[0].value == '80', 'Upload did not replace bundled CSV.')
        bad = tests.drop(columns='Revenue').to_csv(index=False).encode()
        with patch('streamlit.file_uploader', return_value=io.BytesIO(bad)):
            at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
            check(not at.exception and len(at.error) == 1, 'Bad upload must display an error.')
        single = tests[tests.Revenue == 0].head(20).to_csv(index=False).encode()
        with patch('streamlit.file_uploader', return_value=io.BytesIO(single)):
            at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
            check(not at.exception and len(at.warning) >= 1, 'Single-class upload handling failed.')
    checks += ['Training, validation and test feature rows are disjoint',
               'Invalid schema, labels, numbers and empty data rejected',
               'All five model selections executed in Streamlit AppTest',
               'Uploaded 80-row CSV replaced the bundled test set',
               'Missing target and single-class uploads handled in the UI',
               'No Pipeline.fit call occurred during UI checks']
    (ROOT/'results/verification.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2))
    print('\n'.join(checks))


if __name__ == '__main__':
    main()
