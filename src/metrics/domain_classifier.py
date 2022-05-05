import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


# Macros
N_FOLDS = 5
SEED = 2020
REF_LABEL = 0
MONIT_LABEL = 1
R_STATE = 1
LABEL_COL = 'label'
LGBM_PARAMS = {'max_depth': 6, 'reg_alpha': 15, 'random_state': R_STATE}

class DomainClassifier():
    """ Implementation of 'domain classifier', a multivariate drift metric proposed by Simona Maggio
    and Du Phan in the white paper 'A Primer on Data Drift & Drift Detection Techniques'.
    Link: https://pages.dataiku.com/data-drift-detection-techniques

    The algorithm consists of training a binary classifier that must differentiate samples
    with a reference label (data without drift) and monitoring (may have drift).

    If the data distribution is very similar, the classifier will have difficulties to differentiate between
    samples and will present an accuracy close to 0.5, as the drift increases we have an increase
    or decreased accuracy.
    """
    def __init__(self):
        pass

    def _get_accuracy(self, X, y):
        clf = LGBMClassifier(**LGBM_PARAMS)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        return np.mean(scores)

    def _create_dataset(self, monit_df: pd.DataFrame) -> pd.DataFrame:
        monit_df[LABEL_COL] = MONIT_LABEL
        ref_df = self.reference.copy()

        ref_count = ref_df.shape[0]
        monit_count = monit_df.shape[0]
        n = min(ref_count, monit_count)

        monit_df = monit_df.sample(n, random_state=R_STATE)
        ref_df = ref_df.sample(n, random_state=R_STATE)
        df = pd.concat([ref_df, monit_df])
 
        X = df.drop(LABEL_COL, axis=1)
        y = df[LABEL_COL]
        return X, y

    def fit(self, df: pd.DataFrame):
        df[LABEL_COL] = REF_LABEL
        self.reference = df.copy()
        return self

    def evaluate(self, df: pd.DataFrame) -> dict:
        X, y = self._create_dataset(df.copy())
        accuracy = self._get_accuracy(X, y)
        return {'dc_accuracy': accuracy}