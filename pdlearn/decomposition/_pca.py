import sklearn
from sklearn.decomposition import PCA
import pandas as pd
from pdlearn.helper_functions import add_docstring


class PCA(sklearn.decomposition.PCA):
    sub_class_docstring = """
        A wrapper of the sklearn PCA to keep the pandas DataFrame structure.
        
        Original sklearn docstring:
        """
    __doc__ = sub_class_docstring + sklearn.decomposition.PCA.__doc__

    @add_docstring(sklearn.decomposition.PCA.fit.__doc__)
    def fit(self, X: pd.DataFrame(), y=None):
        # logger.debug('Fit PCAPandasFriendly')
        self.cols = X.columns
        return super().fit(X)

    @add_docstring(sklearn.decomposition.PCA.fit_transform.__doc__)
    def fit_transform(self, X: pd.DataFrame(), y=None):
        # logger.debug('Fit_transform PCAPandasFriendly')
        self.cols = X.columns
        X_pc_space = super().fit_transform(X)
        return pd.DataFrame(X_pc_space, index=X.index,
                            columns=["PC{}".format(i + 1) for i in range(X_pc_space.shape[1])])

    @add_docstring(sklearn.decomposition.PCA.transform.__doc__)
    def transform(self, X: pd.DataFrame, y=None):
        # logger.debug('Transforming PCAPandasFriendly')
        x_pc_space = super().transform(X)
        return pd.DataFrame(x_pc_space, index=X.index,
                            columns=["PC{}".format(i + 1) for i in range(x_pc_space.shape[1])])

    @add_docstring(sklearn.decomposition.PCA.inverse_transform.__doc__)
    def inverse_transform(self, X: pd.DataFrame, y=None):
        # logger.debug('Inverse Transforming PCAPandasFriendly')
        x_normal_space = pd.DataFrame(super().inverse_transform(X), columns=self.cols, index=X.index)
        return x_normal_space
