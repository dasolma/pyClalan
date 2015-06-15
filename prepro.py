# coding: utf-8

__author__ = 'dasolma'
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import re


class LinkRemover(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            post = ' '.join(re.sub("(\w+:\/\/\S+)","",post).split())
            result[i] = post

        return result

