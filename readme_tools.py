"""Build the required README sections from the measured experiment artifacts."""
import argparse
import json
from pathlib import Path

import pandas as pd

from shopper_models import ROOT


def markdown_table(headers, rows):
    lines = ['| ' + ' | '.join(map(str, headers)) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def model_observations():
    stats = pd.read_csv(ROOT/'results/test_metrics.csv').set_index('Model')
    matrices = json.loads((ROOT/'results/confusion_matrices.json').read_text())
    descriptions = {
        'Logistic Regression': 'Linear reference with class balancing. Its validation selection status is reported in the overall decision below.',
        'Decision Tree': 'A shallow nonlinear rule set, limited by maximum depth and minimum leaf size. The error counts below show its precision and recall trade-off.',
        'KNN': 'Distance-weighted neighbours favour the majority class in this run. Its relatively high accuracy hides weak purchase detection.',
        'Naive Bayes': 'Correlated predictors and non-Gaussian indicator features limit its assumptions. Read purchase detection alongside the false-positive count.',
        'Support Vector Machine': 'A balanced nonlinear margin classifier using the RBF kernel. Test scores describe its performance; the initial selection uses validation results.',
    }
    rows = []
    for name, s in stats.iterrows():
        tn, fp = matrices[name][0]
        fn, tp = matrices[name][1]
        rows.append([name, f'{descriptions[name]} Purchases detected: {tp}/{tp+fn}; '
                     f'false positives: {fp}; precision {s.Precision:.4f}; recall {s.Recall:.4f}.'])
    winner = json.loads((ROOT/'results/experiment.json').read_text())['validation_winner']
    test_best = stats['MCC'].idxmax()
    rows.append(['Overall winner for this experiment',
                 f'{winner} was selected before test evaluation, using validation MCC and then AUC. '
                 f'{test_best} has the strongest held-out MCC in this particular run. '
                 'A fresh validation or prospective experiment is needed before changing the selected model.'])
    return rows


def build_readme(repository_url=None, app_url=None):
    meta = json.loads((ROOT/'results/experiment.json').read_text())
    metrics = pd.read_csv(ROOT/'results/test_metrics.csv')
    comparison = markdown_table(list(metrics.columns),
        [[row.iloc[0], *[f'{v:.4f}' for v in row.iloc[1:]]] for _, row in metrics.iterrows()])
    observations = markdown_table(['ML Model Name', 'Observation about model performance'], model_observations())
    baseline = (meta['splits']['test']['rows'] - meta['splits']['test']['purchases'])/meta['splits']['test']['rows']
    return f'''# Shopper conversion model comparison

This is an explained learning reference for ML Assignment 2. The assignment permits AI only for learning support. Understand, adapt and rerun the work on BITS Virtual Lab before preparing your own submission. The bundled experiment was run in an assistant workspace and is not BITS Lab evidence.

## a. Problem statement

Classify whether an e-commerce session ends in a purchase, using observed session attributes. Revenue=1 means a purchase; Revenue=0 means no purchase. The practical question is whether the models identify purchasers rather than simply predict the larger non-purchase class. This is retrospective session classification, not a validated live targeting system.

## b. Dataset description

Source: [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

The downloaded CSV has 12,330 rows, 17 predictors and one target. PageValues is excluded because its transaction-related definition raises an availability and leakage concern. That leaves 16 predictors: nine numeric and seven categorical. There are no missing cells in the source CSV. Removing 125 identical retained rows leaves 12,205 rows, with 1,908 purchases and 10,297 non-purchases. These counts were computed from the downloaded file.

Numeric predictors: Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, SpecialDay.

Categorical predictors: Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend. Integer category codes are encoded as categories, not treated as magnitudes. The target is excluded from the input features.

The dataset meets both minimums: 16 retained predictors exceeds 12, and 12,205 retained rows exceeds 500. UCI dataset credit: Sakar, C. and Kastro, Y. (2018), DOI [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), CC BY 4.0. Data modifications: PageValues excluded, duplicate retained rows removed, target and category representations normalised, and train/validation/test partitions created.

## c. Github Repository Link

{repository_url or 'Not configured. Supply the real repository URL after creating and checking your repository.'}

Live Streamlit application: {app_url or 'Not deployed. A real Community Cloud URL must be added after deployment.'}

## d. Models used

All five models use the same retained dataset and identical partitions. The split is stratified: 7,323 training rows, 2,441 validation rows and 2,441 held-out test rows. Random seed: {meta['seed']}. Each family has three candidate settings; validation MCC selects the setting, with validation AUC as tie-breaker. The overall candidate is also selected before test evaluation. Each selected pipeline is then refitted on the 9,764 training plus validation rows and evaluated once on the same held-out test set.

Preprocessing is fitted inside each pipeline: median imputation, log1p for six non-negative count/duration columns, numerical standardisation, most-frequent categorical imputation and one-hot encoding. Pipelines contain both the transformations and the classifier. The app loads these saved pipelines and never trains on an uploaded CSV.

Logistic Regression, Decision Tree and SVM use balanced class weights. KNN and Gaussian Naive Bayes use their standard unweighted fitting. These are comparisons of documented configured pipelines, not a claim that each model family has been exhaustively optimised. Gaussian Naive Bayes is used because Multinomial Naive Bayes cannot accept the negative values produced by standardisation.

### Test metric comparison

{comparison}

All numbers are measured results from the bundled reference run, rounded to four decimals for display. Precision, recall and F1 use the purchase class, not macro or weighted averages. AUC uses continuous decision scores for Logistic Regression and SVM, and purchase probabilities for the other models. The other five metrics use native predict outputs. The test set has 382 purchases and 2,059 non-purchases. An always-no-purchase baseline has accuracy {baseline:.4f}, purchase precision/recall/F1/MCC of zero by the stated zero-division convention, and AUC 0.5000 for a constant score.

### Observations on model performance

{observations}

No model exceeds the majority baseline on accuracy in this experiment. Several detect substantially more purchases; whether that is useful depends on the cost of false positives and missed purchases. No monetary benefit or causal lift is established. The models are not calibrated probability estimators, and the SVM output is a margin. The ranking can change with feature availability, split, class weights, preprocessing and decision threshold. Only a small parameter search and one random split were used.

### Files and execution

The repository includes app.py, requirements.txt, README.md, test_data.csv, five .joblib pipelines in model/, training source, shared preprocessing source, source data, split files, metrics, predictions, confusion matrices and a verification script. The Streamlit app displays upload, selection, all-model metrics, confusion matrix, classification report, ROC comparison and prediction downloads.

From a Python 3.12 environment on BITS Virtual Lab, run:

```bash
python -m pip install -r requirements.txt
python train_models.py --execution-location "BITS Virtual Lab"
python verify_project.py
python readme_tools.py
python -m streamlit run app.py
```

Use the execution-location option only when the command is actually run on BITS Virtual Lab. Capture a real screenshot there. The included reference artifacts do not satisfy that proof requirement. Details for GitHub, Community Cloud and the single-PDF submission are in WORKED_GUIDE.md.

### Reproducibility and limitations

Execution recorded in current artifacts: {meta['execution_location']}.
Source SHA-256: {meta['source_sha256']}.
Versions: Python {meta['versions']['python']}, scikit-learn {meta['versions']['scikit-learn']}, NumPy {meta['versions']['numpy']}, pandas {meta['versions']['pandas']}, SciPy {meta['versions']['scipy']}, joblib {meta['versions']['joblib']}.

Install the pinned requirements and use Python 3.12 for the closest reproduction. Retraining creates new model artifacts; changing seeds, preprocessing or package versions may change the results. Never edit metric values to match the reference. Run readme_tools.py after retraining so the tables reflect the current artifacts. If you change the experiment design, rewrite the descriptive text and observations to match it.

Features represent observed sessions. Excluding PageValues alone does not establish that all other features were available early enough for a live decision. The dataset has no usable per-row timestamp for a temporal deployment test. This random split therefore estimates performance on similar sampled sessions, not future conversion lift.

### Documentation

- [scikit-learn 1.8 evaluation metrics](https://scikit-learn.org/1.8/modules/model_evaluation.html)
- [scikit-learn leakage and preprocessing guidance](https://scikit-learn.org/1.8/common_pitfalls.html)
- [scikit-learn Gaussian Naive Bayes](https://scikit-learn.org/1.8/modules/naive_bayes.html)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-url')
    parser.add_argument('--app-url')
    args = parser.parse_args()
    (ROOT/'README.md').write_text(build_readme(args.repo_url, args.app_url))
    print('README.md regenerated from current experiment results.')


if __name__ == '__main__':
    main()
