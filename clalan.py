__author__ = 'dasolma'
import pickle
import sys, getopt
import os.path
from config import *
from dataretriever import *
from modelanalizer import LanguageClassifier
from sklearn.metrics import confusion_matrix
import sys
import warnings
warnings.filterwarnings('ignore')


def main(argv):
   inputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hl:i:o:dt",["ifile=","ofile=", "languajes=", "download="])
   except getopt.GetoptError:
      print 'clalan.py -l <languajes> [-i <inputfile>] [-o <outputfile>] [-d] [-t]'
      sys.exit(2)

   inputfile = None
   download = False
   languages = ["es"]
   train = False
   inputfile = inputfile
   for opt, arg in opts:
      if opt == '-h':
         print 'clalan.py -l <languajes> -i <inputfile> [-o <outputfile>] [-d] [-t]'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         sys.stdout = open(arg, 'wb+')
      elif opt in ("-l", "languajes"):
          languages = arg.split(",")
      elif opt in ("-d", "download"):
          download = True
      elif opt in ("-t", "train"):
          train = True

   data_train,target_train,data_test,target_test, downloaded = \
       get_data(languages,train_words=train_words, test_words=test_words, download=download)


   if downloaded or not os.path.isfile(inputfile) or train:
        classifier = LanguageClassifier(languages, pipelines, params)
        classifier.fit(data_train,target_train)

        classifier.save(inputfile)
   else:
        print 'Loading the classifier'
        classifier = LanguageClassifier.load(inputfile)

   classifier.score(data_test,target_test)

   print ""
   print 'Confusion matrix over train train:'
   pred = classifier.best_classifier.predict(data_train)
   print_conf_matrix(confusion_matrix(pred, target_train), languages)
   print ""

   print 'Confusion matrix over test train:'
   pred = classifier.best_classifier.predict(data_test)
   print_conf_matrix(confusion_matrix(pred, target_test), languages)
   print ""


def print_conf_matrix(mat, languages):
    row_format ="{:>10}" * len(mat[0])
    row_format = "{:<5}" + row_format
    print row_format.format("", *languages)
    for i in range(len(languages)):
        r = mat[i]
        lang = languages[i]
        print row_format.format(lang, *r)


if __name__ == "__main__":
   main(sys.argv[1:])
