__author__ = 'dasolma'

def compose_pipelines(transformers, classifiers):

    if transformers == None or len(transformers) == 0:
        if isinstance(classifiers, list): return classifiers
        else: return [classifiers]

    result = []

    item = transformers[0]
    if isinstance(item, tuple):
        for subpipe in compose_pipelines(transformers[1:], classifiers):
            if isinstance(subpipe, tuple): subpipe = [subpipe]
            result.append([item] + subpipe)
    else:
        for item in transformers[0]:
            for subpipe in compose_pipelines(transformers[1:], classifiers):
                if isinstance(subpipe, tuple): subpipe = [subpipe]
                if item is None:
                    result.append(subpipe)
                else:
                    if isinstance(item, list):
                        result.append(item + subpipe)
                    else:
                        result.append([item] + subpipe)

    return result

def name_pipelines(pipelines):

    result = []
    for pipe in pipelines:
        names = [name for name,proc in pipe]
        #print names
        name = "->".join(names)
        result.append((name, pipe))


    return result

'''
from tweetprepro import TextExtractor, UserAndLinkReplacer, EmoticonsReplacer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from tweetestimators import LinguisticAndPolarity


tranformers = [('extr', TextExtractor()),
      [ None, ('emo', EmoticonsReplacer() ) ],
      [ None, ('ual', UserAndLinkReplacer() ) ],
      [ ('ling', LinguisticAndPolarity()),
        [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]
      ]
     ]

classifiers = [('NB', MultinomialNB()),
               ('svm', LinearSVC()),
               ('LR', LogisticRegression()),
               ('SGD', SGDClassifier()),
               ('Random forest', RandomForestClassifier())
              ]


pipelines =  name_pipelines(compose_pipelines(tranformers, classifiers))

for e in  pipelines:
    print e
'''