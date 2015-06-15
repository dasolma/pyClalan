__author__ = 'dasolma'
import sys
from datetime import datetime
import random
import os.path

def get_data(languages=["es"], train_words = 1000000, test_words=100000, words_by_line = 20, download=False):

    data_train,target_train, downloaded = _get_data(languages, total_words=train_words, words_by_line=20,
                                        random_chunk=False, download=download)
    data_test,target_test, d = _get_data(languages, total_words=test_words,
                                      words_by_line=20, download=download, train=False)

    return  data_train,target_train,data_test,target_test, downloaded

def _get_data(languages=["es"], total_words = None, offset_words = 0,
              words_by_line = 20, random_chunk=True, download = False, train=True):
    if total_words is None: total_words = sys.maxint

    data = []
    target = []
    downloaded = False
    for i,lang in zip(range(len(languages)), languages):
        file = "%s/%s.txt"%('train' if train else 'test',lang)
        if os.path.isfile(file):
            fp = open(file, "rb")
            lines = fp.readlines()
            included_word_count = 0
            word_count = 0
            for line in lines:
                sentences = filter(None, line.strip().split("."))
                if len(sentences) > 1:
                    sizes = [len(sentence) for sentence in sentences]
                    if min(sizes) > 20:
                        chunk_size = random.randint(1,words_by_line)
                        for s in chunks(line.split(), random.randint(1,words_by_line)):
                            word_count += len(s)
                            if word_count > offset_words:
                                included_word_count += len(s)

                                data.append(" ".join(s))
                                target.append(i)

                                if included_word_count > total_words:
                                    break

                        if included_word_count > total_words:
                                break
            fp.close()

        elif download:
            downloaded = True

            print "Downloading corpus for %s"%lang
            corpus = _get_corpus(lang, total_words*2, train=train)

            fp = open(file, "wb")
            fp.write(corpus)
            fp.close()

        else:
            print "Error: not found corpus for %s language"%lang
            print "You can use -d option to download it."
            raise Exception("not found corpus for %s language"%lang)


    if not train: download = False

    return data, target, downloaded



def _get_corpus(lang, word_limit = 1500000, train=True):
    import wikipedia

    lines =  open("train/people.txt", "rb").readlines()
    terms = [l.split(",")[1].strip() for l in lines]
    if not train: terms.reverse()

    corpus = ""
    wikipedia.set_lang(lang)
    last_time = datetime.now()
    for term in terms:
        try:
            #print term
            ny = wikipedia.page(term)
            corpus += ny.content.encode('utf-8')
            l = len(corpus.split())

            # word_limit ----------> 1
            # l -------------------> x
            # wl * x = l -> x = l/w * 100
            #print l

            if (datetime.now() - last_time).seconds > 10:
                last_time = datetime.now()
                per = (float(l) / word_limit) * 100
                print "%f%% completed"%per

            if l > word_limit: break
        except:
            pass

    #print len(corpus[lang].split())
    return corpus


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

'''
#print _get_data(["es", "be", "pl"], total_words=20)
#print _get_data(["es", "be", "pl"], total_words=3, offset_words=20)

data_train,target_train,data_test,target_test = get_data(["es", "be", "pl"], train_words=8000, test_words=4000)


print len(data_train)
print len(data_test)

_get_data(languages=["fr"], total_words=1000 )

'''
