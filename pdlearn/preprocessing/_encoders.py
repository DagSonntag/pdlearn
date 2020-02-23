import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from pdlearn.helper_functions import add_docstring


class OneHotEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    sub_class_docstring = """
        A wrapper of the sklearn OneHotEncoder to keep the pandas DataFrame structure.
        
        Original sklearn docstring:
        """
    __doc__ = sub_class_docstring + sklearn.preprocessing.OneHotEncoder.__doc__

    def __init__(self, sparse=True, **args):
        self.encoder = sklearn.preprocessing.OneHotEncoder(**args, sparse=sparse)
        self.categories_ = None
        self.sparse = sparse
        self.old_column_structure = None

    @add_docstring(sklearn.preprocessing.OneHotEncoder.fit.__doc__)
    def fit(self, X: pd.DataFrame, y=None):
        self.old_column_structure = X.columns
        res = self.encoder.fit(X)
        self.categories_ = {col: list(encoding) for col, encoding in
                            zip(self.old_column_structure, self.encoder.categories_)}
        return self

    @add_docstring(sklearn.preprocessing.OneHotEncoder.transform.__doc__)
    def transform(self, X: pd.DataFrame, y=None):
        # Make sure that the column_structure is correct
        x_new = X.loc[:, self.old_column_structure]
        # Encode
        x_new = self.encoder.transform(x_new)
        # Add structure back
        if self.sparse:
            x_new = pd.DataFrame.sparse.from_spmatrix(x_new,
                                                      columns=["{}_{}".format(col, encoding) for col in
                                                               self.old_column_structure for encoding in
                                                               self.categories_[col]],
                                                      index=X.index)
        else:
            x_new = pd.DataFrame(x_new,
                                 columns=["{}_{}".format(col, encoding) for col in self.old_column_structure for
                                          encoding in self.categories_[col]],
                                 index=X.index)
        return x_new


class OrdinalEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    sub_class_docstring = """
        A wrapper of the sklearn ordinal encoder to keep the pandas DataFrame structure.
        Note how the new categorization is stored under self.categories_

        Original sklearn docstring:
        """
    __doc__ = sub_class_docstring + sklearn.preprocessing.OrdinalEncoder.__doc__

    def __init__(self, **args):
        self.encoder = sklearn.preprocessing.OrdinalEncoder(**args)
        self.categories_ = None
        self.column_structure = None

    @add_docstring(sklearn.preprocessing.OrdinalEncoder.fit.__doc__)
    def fit(self, X: pd.DataFrame, y=None):
        # Save column structure
        self.column_structure = X.columns
        # Fit
        res = self.encoder.fit(X)
        # Save category data
        self.categories_ = {col: list(encoding) for col, encoding in
                            zip(self.column_structure, self.encoder.categories_)}
        return self

    @add_docstring(sklearn.preprocessing.OrdinalEncoder.transform.__doc__)
    def transform(self, X: pd.DataFrame, y=None):
        # Make sure that the column_structure is correct
        x_new = X.loc[:, self.column_structure]
        # Encode
        x_new = self.encoder.transform(x_new)
        # Add structure back
        x_new = pd.DataFrame(x_new, columns=self.column_structure, index=X.index)
        return x_new

    @add_docstring(sklearn.preprocessing.OrdinalEncoder.inverse_transform.__doc__)
    def inverse_transform(self, X: pd.DataFrame, y=None):
        # Make sure that the column_structure is correct
        x_new = X.loc[:, self.column_structure]
        # Inverse encode
        x_new = self.encoder.inverse_transform(x_new)
        # Add structure back
        x_new = pd.DataFrame(x_new, columns=self.column_structure, index=X.index)
        return x_new


