from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
#from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import json
import time
import numpy as np
import operator
from pprint import pprint as pp

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")

#INFILE = "/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/reviews_objLV.json"
#OUTFILE = '/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/reviews_sentiment_naive_bayesLV.json'

INFILE = "/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/reviews_train_objLV.json"
OUTFILE = '/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/reviews_train_sentiment_naive_bayesLV.json'

PRINT_EVERY = 10000

def read_data(fname):
  f = open(fname, 'rb')
  texts = []
  ys = []
  ids = dict()
  for i,line in enumerate(f):
    rec = json.loads(line.strip())
    texts.append(rec["text"])
    ys.append(rec["stars"])
    ids[i] = {'rid':rec["review_id"],'funny':rec['votes']['funny'], 'useful':rec['votes']['useful'],'cool':rec['votes']['cool'],'user_id':rec['user_id'],'business_id':rec['business_id'] }
  f.close()
  return texts, np.array(ys), ids

def vectorize(texts, vocab=[]):
  vectorizer = CountVectorizer(min_df=0, stop_words="english") 
  if len(vocab) > 0:
    vectorizer = CountVectorizer(min_df=0, stop_words="english", 
      vocabulary=vocab)
  X = vectorizer.fit_transform(texts)
  return vectorizer.vocabulary_, X

"""def vectorize(texts, vocab=[]):
  vectorizer = TfidfVectorizer(min_df=1,norm='l2', smooth_idf=True, stop_words="english",sublinear_tf=True,  use_idf=True)
  if len(vocab) > 0:
    vectorizer = TfidfVectorizer(min_df=1,norm='l2', smooth_idf=True, stop_words="english",sublinear_tf=True,  use_idf=True, vocabulary=vocab)
  X = vectorizer.fit_transform(texts)
  return vectorizer.vocabulary_, X"""


def cross_validate(X, y, nfeats):
  logger.info("Model Validation")
  nrows = X.shape[0]
  kfold = KFold(nrows, 10)
  scores = []
  print "-"*10
  print "nfeats:%d"%nfeats
  print "accuracy\tf1\tprecision\trecall"
  for train, test in kfold:
    Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
    clf = MultinomialNB()

    clf.fit(Xtrain, ytrain)
    #score = clf.score(X_test, y_test)
    #pp(score)
    y_pred = clf.predict(Xtest)

    accuracy = accuracy_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred, average="macro")
    precision = precision_score(ytest, y_pred, average="macro")
    recall = recall_score(ytest, y_pred, average="macro")
    scores.append((accuracy, f1, precision, recall))
    print "%2.3F\t%2.3f\t%2.3f\t%2.3f"%(accuracy, f1, precision, recall)
    

  mean_accuracy = np.mean([x[0] for x in scores])
  mean_f1 = np.mean([x[1] for x in scores])
  mean_precision = np.mean([x[2] for x in scores])
  print "mean_accuracy:%2.2f"%mean_accuracy
  print "mean_f1:%2.2f"%mean_f1
  print "mean_precision:%2.2f"%mean_precision



def sorted_features(V, X, y, topN):
  iv = {v:k for k, v in V.items()}
  chi2_scores = chi2(X, y)[0]
  top_features = [(x[1], iv[x[0]], x[0]) 
    for x in sorted(enumerate(chi2_scores), 
    key=operator.itemgetter(1), reverse=True)]
  print "TOP 10 FEATURES FOR"
  for top_feature in top_features[0:30]:
    print "%7.3f  %s (%d)" % (top_feature[0], top_feature[1], top_feature[2])
  return [x[1] for x in top_features]

def proc(X, y, ids, out_file):
  #http://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
  logger.info("Sentiment Analysis")
  ret = dict()
  
  train = X.shape[0]
  pp( X.shape)
  Xtrain =  X
  ytrain =  y
  clf = MultinomialNB()
  clf.fit(Xtrain, ytrain)
  

  fo = open(out_file,'w')
  error = 0
  for i,text in enumerate(Xtrain):
    if i % PRINT_EVERY == 0:
      logger.info("Working on %d"%i)

    y_pred = clf.predict(text)
    y_pred_proba = clf.predict_proba(text)
    y_pred_log_proba = clf.predict_log_proba(text)
    review_id = ids[i]['rid']
    funny = ids[i]['funny']
    useful = ids[i]['useful']
    cool = ids[i]['cool']
    user_id = ids[i]['user_id']
    business_id = ids[i]['business_id']

    yip = y_pred[0] - 1
    if yip < 0:
      yip = 0

    if y[i] != y_pred[0]:
      error += 1
    #print review_id,y_pred[0],y_pred_proba[0][yip]
    line = json.dumps({'review_id':review_id, 'user_id':user_id, 'business_id':business_id, 'y':y[i],'y_pred':y_pred[0], 'y_pred_proba':y_pred_proba[0][yip], 'y_pred_log_proba':y_pred_log_proba[0][yip], 'funny':funny, 'useful':useful,'cool':cool})
    fo.write("%s\n"%line)

  fo.close()
  print "error:",error
  print "ratio:",100.0*float(error)/float(train)
  logger.info("generated file %s"%out_file)

def main():
  logger.info("Start")
  start = time.clock()
  out_file = OUTFILE
  texts, ys, ids = read_data(INFILE)
  
  y = ys
  V, X = vectorize(texts)
  cross_validate(X, y, -1)
  sorted_feats = sorted_features(V, X, y, 30)
  # find the best number of features = 30000
  """for nfeats in [1000, 3000, 10000, 30000, 100000]:
    V, X = vectorize(texts, sorted_feats[0:nfeats])
    cross_validate(X, y, nfeats)"""
  
  nfeats = 30000
  V, X = vectorize(texts, sorted_feats[0:nfeats])
  cross_validate(X, y, nfeats)

  proc(X, y, ids, out_file)
  elapsed = (time.clock() - start)      
  logger.info("done in %d secs"%int(elapsed))

if __name__ == "__main__":
  main()
