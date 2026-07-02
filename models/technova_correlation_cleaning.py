from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.80):
        self.threshold = threshold
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        X_num = X.select_dtypes(include=[np.number])

        corr_matrix = X_num.corr().abs()

        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.columns_to_drop_ = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.threshold)]

        return self

    def transform(self, X):
        X_out = X.copy()

        cols_to_drop = [col for col in self.columns_to_drop_ if col in X_out.columns]
        X_out.drop(columns=cols_to_drop, inplace=True)

        return X_out