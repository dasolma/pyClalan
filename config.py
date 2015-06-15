__author__ = 'dasolma'
from prepro import LinkRemover
from modelanalizer import ModelAnalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
#from vectorizers import LinguisticAndPolarity, TfidfLinguisticAndPolarity
from pipelines import *

transformers = [
      #[ None, ('ual', LinkRemover() ) ],
      [ [('vect', CountVectorizer()) ],
        [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())],
      ]
     ]

classifiers = [('mNB', MultinomialNB()),
               ('svm', LinearSVC()),
               ('LR', LogisticRegression()),
               #('SGD', SGDClassifier('perceptron')),
               #('Random forest', RandomForestClassifier())
              ]

params = {'vect__ngram_range':[(1,1),(1,2),(1,3)],
          'tfidf__smooth_idf':[True, False],
          'tfidf__use_idf': [True, False],
          'tfidf__sublinear_tf': [True, False],
          'mNB__alpha':[0, 0.1, 0.5, 1.0],
          'svm__C':[0.1, 10, 100],
          'tfidf+ling__sublinear_tf': [True, False],
          'tfidf+ling__ngram_range':[(1,1),(1,2),(1,3)],
          'tfidf+ling__smooth_idf':[True, False],
          'tfidf+ling__use_idf': [True, False],
          }



pipelines =  name_pipelines(compose_pipelines(transformers, classifiers))

train_words=1000000
test_words=100000