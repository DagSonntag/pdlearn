import sklearn
from sklearn.compose import ColumnTransformer
import pandas as pd
from pdlearn.helper_functions import add_docstring


class ColumnTransformer(sklearn.compose.ColumnTransformer):
    sub_class_docstring = """
        A wrapper of the sklearn ColumnTransformer to return the data in a DataFrame format.
        Note that it is extremely inefficient compared to the original transformer (no parallelization).
        
        Original sklearn docstring:
        """
    __doc__ = sub_class_docstring + sklearn.compose.ColumnTransformer.__doc__

    @add_docstring(sklearn.compose.ColumnTransformer.fit_transform.__doc__)
    def fit_transform(self, X: pd.DataFrame, y=None):
        dfs = []
        for pipeline_config in self.transformers:
            res = pipeline_config[1].fit_transform(X[pipeline_config[2]])
            dfs.append(res)
        return pd.concat(dfs, axis=1)

    @add_docstring(sklearn.compose.ColumnTransformer.transform.__doc__)
    def transform(self, X: pd.DataFrame, y=None):
        dfs = []
        for pipeline_config in self.transformers:
            res = pipeline_config[1].transform(X[pipeline_config[2]])
            dfs.append(res)
        return pd.concat(dfs, axis=1)
