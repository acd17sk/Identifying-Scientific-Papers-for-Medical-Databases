from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class W2V(BaseEstimator):

    def __init__(self, mode, n_w_features=50, n_features=50):
        self.n_w_features = n_w_features
        self.n_features = n_features
        self.mode = mode
        self.model = None
        self.vocabulary = None

    def get_model(self):
        return self.model


    def fit(self, dataset):
        self.model = Word2Vec(sentences=dataset, sg=self.mode, size=self.n_w_features)
        self.vocabulary = set(self.model.wv.vocab)
        return self

    def transform(self, dataset):
        res = np.empty((len(dataset), self.n_w_features*self.n_features))
        for i, text in enumerate(dataset):
            row = np.array([])
            sentence = np.append(text, ['#']*(self.n_features-len(text))) if len(text) < self.n_features else np.array(text[:self.n_features])
            for w in sentence:
                row = np.append(row, self.model.wv[w] if w in self.vocabulary else [np.zeros(self.n_w_features)])
            res[i] = row
        return res

    def fit_transform(self, dataset, Y=None):
        self.fit(dataset)
        return self.transform(dataset)

