import pandas as pd
from typing import Tuple
from sklearn.decomposition import PCA


# Macros
R_STATE = 2022
VERBOSE = False
DEFAULT_PCA_IMPUTE_VALUE = -9999


class Metabase():
    def __init__(
        self,
        target_col: str,
        prediction_col: str,
        pca_n_components: Tuple[int, float] = None,
        verbose: bool = VERBOSE,
    ):
        # Update object properties with init params
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.pca_n_components = pca_n_components
        self.verbose = verbose

        # Other properties
        self.size = 0
        self.new_batch_size = 0
        self.new_row_id = 0
        self.new_target_id = 0

    def _reduce_dim(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.pca_n_components:
            return dataframe

        dataframe = dataframe.fillna(DEFAULT_PCA_IMPUTE_VALUE)
        if not hasattr(self, 'pca'):
            svd_solver = "auto" if self.pca_n_components > 1 else "full"
            self.pca = PCA(
                n_components=self.pca_n_components,
                svd_solver=svd_solver,
                random_state=R_STATE
            ).fit(dataframe)

        if self.verbose:
            n_comp = self.pca.n_components_
            variance = '{0:.2f}'.format(
                sum(self.pca.explained_variance_ratio_) * 100)
            print(
                f'Dim reduction - keeping {n_comp} components explaining {variance}% of variance')
        return pd.DataFrame(self.pca.transform(dataframe))

    def fit(self, first_metabase: pd.DataFrame) -> None:
        """Fit offline data"""
        df = first_metabase.copy().reset_index(drop=True)
        self.metabase = df
        
        df_size = df.shape[0]
        self.new_row_id = df_size
        self.new_target_id = df_size
        self.learning_window_size = df_size

    def get_train_metabase(self) -> pd.DataFrame:
        # Clear new batch counter
        self.new_batch_size = 0

        # Get metabase prepared for model training
        metabase = self.get_raw()
        metabase = self._reduce_dim(metabase)

        # Remove instances with unknown target
        metabase = metabase.dropna(subset=[self.target_col])
        train_metabase = metabase.tail(self.learning_window_size)
        try:
            return train_metabase.drop(self.prediction_col, axis=1)
        except:
            return train_metabase

    def get_raw(self) -> pd.DataFrame:
        return self.metabase.copy()

    def update(self, new_line: pd.DataFrame) -> None:
        """Update meta base with new online data"""
        new_line["id"] = self.new_row_id
        new_line = new_line.set_index("id")
        self.metabase = pd.concat([self.metabase, new_line])
        self.new_row_id += 1

    def update_target(self, y: float) -> None:
        """Update meta base with upcoming target"""
        self.new_batch_size += 1
        self.metabase.at[self.new_target_id, self.target_col] = y
        self.new_target_id += 1

    def update_predictions(self, y: float) -> None:
        self.metabase[self.prediction_col] = y
