"""Streamlit evaluation interface. Only loads saved pipelines; never fits them."""
import io
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from shopper_models import (ROOT, FEATURES, TARGET, METRICS, MODEL_FILES,
                            evaluate_pipeline, normalise_frame)

st.set_page_config(page_title='Shopper conversion model review', page_icon='🛒', layout='wide')


@st.cache_resource
def load_models():
    # Only repository-owned artifacts are loaded, never an uploaded pickle.
    return {name: joblib.load(ROOT / 'model' / filename)
            for name, filename in MODEL_FILES.items()}


@st.cache_data(show_spinner=False)
def review_csv(csv_bytes):
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    if len(frame) > 5000:
        raise ValueError('Upload test data only, with at most 5,000 rows.')
    x, y = normalise_frame(frame)
    review, rows = {}, []
    for name, fitted in load_models().items():
        result = evaluate_pipeline(fitted, x, y)
        review[name] = result
        rows.append({'Model': name, **result[0]})
    extras = [c for c in frame.columns if c not in FEATURES + [TARGET]]
    return frame, x, y, pd.DataFrame(rows), review, extras


st.title('Shopper conversion model review')
st.write('Compare five classifiers on the same labelled e-commerce test sessions. '
         'Purchase is the positive class. Choose a model to inspect its errors.')
st.caption('Scope: retrospective session classification using 16 predictors; PageValues is excluded.')

with st.sidebar:
    st.header('Evaluation data')
    upload = st.file_uploader('Upload a labelled test CSV', type=['csv'])
    st.caption('Include all 16 input features and Revenue. Upload test data only; maximum 5,000 rows.')
    selected = st.selectbox('Model', list(MODEL_FILES))
    try:
        sample_bytes = (ROOT / 'test_data.csv').read_bytes()
    except OSError:
        st.error('Missing test_data.csv. Run train_models.py on BITS Virtual Lab first.')
        st.stop()
    st.download_button('Download bundled test CSV', sample_bytes, 'test_data.csv', 'text/csv')
    with st.expander('Required CSV columns'):
        st.code('\n'.join(FEATURES + [TARGET]), language=None)

try:
    if not (ROOT / 'results/experiment.json').exists():
        raise ValueError('Missing experiment metadata. Run train_models.py first.')
    metadata = json.loads((ROOT / 'results/experiment.json').read_text())
    incoming = upload.getvalue() if upload is not None else sample_bytes
    frame, x, y, comparison, details, ignored = review_csv(incoming)
except (ValueError, OSError, EOFError, ImportError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f'Cannot evaluate this CSV: {exc}')
    st.stop()

st.info('Using uploaded CSV.' if upload is not None else 'Using bundled held-out test_data.csv. Upload a CSV to evaluate another labelled test set.')
if ignored:
    st.warning('Ignored extra columns: ' + ', '.join(ignored))
if y.nunique() < 2:
    st.warning('This CSV contains only one actual class. AUC is undefined and shown as N/D. '
               'Precision, recall, F1 and MCC follow scikit-learn zero-division conventions; '
               'interpret them with the confusion matrix.')

first = next(iter(load_models().values()))
encoder = first.named_steps['prepare'].named_transformers_['categories'].named_steps['encode']
from shopper_models import CATEGORIES
unseen = []
for column, known in zip(CATEGORIES, encoder.categories_):
    values = set(x[column].dropna().unique()) - set(known)
    if values:
        unseen.append(f'{column}: {", ".join(sorted(values))}')
if unseen:
    st.warning('Unseen categories are encoded as all zeros for their feature: ' + '; '.join(unseen))

summary_columns = st.columns(4)
summary_columns[0].metric('Test sessions', f'{len(y):,}')
summary_columns[1].metric('Purchases', f'{int(y.sum()):,}')
summary_columns[2].metric('Purchase rate', f'{y.mean():.2%}')
summary_columns[3].metric('Majority-class accuracy', f'{max(y.mean(), 1-y.mean()):.2%}')

st.subheader('All five models on this CSV')
st.dataframe(comparison.set_index('Model').style.format('{:.4f}', na_rep='N/D'),
             width='stretch')
st.download_button('Download comparison metrics', comparison.to_csv(index=False),
                   'model_comparison.csv', 'text/csv')
st.caption('Precision, recall and F1 refer to Revenue=1. AUC uses a continuous margin or probability; '
           'other metrics use each saved classifier\'s native predictions.')

metrics, matrix, report, predicted, score, score_label = details[selected]
st.subheader(f'Inspect {selected}')
metric_columns = st.columns(6)
for column, key in zip(metric_columns, METRICS):
    column.metric(key, 'N/D' if metrics[key] is None else f'{metrics[key]:.4f}')

left, right = st.columns(2)
with left:
    st.write('**Confusion matrix**')
    matrix_frame = pd.DataFrame(matrix, index=['Actual no purchase', 'Actual purchase'],
                               columns=['Predicted no purchase', 'Predicted purchase'])
    st.dataframe(matrix_frame, width='stretch')
    tn, fp, fn, tp = matrix.ravel()
    st.write(f'{tp} purchases found; {fn} purchases missed; {fp} non-purchases flagged; '
             f'{tn} non-purchases correctly rejected.')
with right:
    st.write('**Classification report**')
    st.dataframe(pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T
                 .style.format('{:.4f}'), width='stretch')

if y.nunique() == 2:
    with st.expander('Compare ROC curves'):
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, detail in details.items():
            fpr, tpr, _ = roc_curve(y, detail[4])
            ax.plot(fpr, tpr, label=f'{name} ({detail[0]["AUC"]:.3f})')
        ax.plot([0, 1], [0, 1], '--', color='grey', label='Chance ranking')
        ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with st.expander('Review rows and download predictions'):
    exported = frame.copy()
    exported['PredictedRevenue'] = predicted
    exported['PurchaseScore'] = score
    st.caption(score_label + '. A decision margin is not a calibrated probability.')
    st.dataframe(exported.head(50), width='stretch')
    st.download_button('Download selected model predictions', exported.to_csv(index=False),
                       'predictions.csv', 'text/csv')

with st.expander('Experiment and interpretation'):
    st.write(f'Validation-selected candidate: **{metadata["validation_winner"]}**. '
             'Selection used validation MCC, with AUC as the tie-breaker. '
             'The uploaded CSV does not change this original selection.')
    st.write('Models were trained offline and are loaded as complete preprocessing and '
             'classification pipelines. No training happens in this app.')
    st.write('These observational results do not establish whether targeting a session '
             'would cause an additional purchase. Session-end features must be available '
             'at the intended prediction time before live use.')
    st.json({'seed': metadata['seed'], 'splits': metadata['splits'],
             'versions': metadata['versions'], 'parameters': metadata['chosen_parameters']})
