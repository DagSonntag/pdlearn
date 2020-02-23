import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pdlearn.helper_functions import add_docstring


class StandardScaler(sklearn.preprocessing.StandardScaler):
    sub_class_docstring = """
        A wrapper of the sklearn StandardScaler to keep the pandas DataFrame structure.
        
        Original sklearn docstring:
        """
    __doc__ = sub_class_docstring + sklearn.preprocessing.StandardScaler.__doc__

    @add_docstring(sklearn.preprocessing.StandardScaler.transform.__doc__)
    def transform(self, X: pd.DataFrame, y=None):
        x_scaled = pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)
        return x_scaled

    @add_docstring(sklearn.preprocessing.StandardScaler.inverse_transform.__doc__)
    def inverse_transform(self, X: pd.DataFrame, y=None):
        x_inv_scaled = pd.DataFrame(super().inverse_transform(X), columns=X.columns, index=X.index)
        return x_inv_scaled
