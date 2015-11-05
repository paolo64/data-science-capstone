from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
#from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import json
import time
import numpy as np
import operator

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")

def read_data(fname):
  f = open(fname, 'rb')
  texts = []
  ys = []
  for line in f:
    rec = json.loads(line.strip())
    texts.append(rec["text"])
    ys.append(rec["stars"])
  f.close()
  return texts, np.array(ys)

def vectorize(texts, vocab=[]):
  vectorizer = CountVectorizer(min_df=0, stop_words="english") 
  if len(vocab) > 0:
    vectorizer = CountVectorizer(min_df=0, stop_words="english", 
      vocabulary=vocab)
  X = vectorizer.fit_transform(texts)
  return vectorizer.vocabulary_, X

def cross_validate(X, y, nfeats):
  #http://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
  nrows = X.shape[0]
  kfold = KFold(nrows, 10)
  scores = []
  for train, test in kfold:
    Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    y_pred = clf.predict(Xtest)
    print(f1_score(ytest, y_pred, average="macro"))
    print(precision_score(ytest, y_pred, average="macro"))
    print(recall_score(ytest, y_pred, average="macro")) 
    #accuracy = accuracy_score(ytest, ypred)
    #precision = precision_score(ytest, ypred)
    #recall = recall_score(ytest, ypred)
    #scores.append((accuracy, precision, recall))
  """print ",".join([ufc_val, str(nfeats), 
    str(np.mean([x[0] for x in scores])),
    str(np.mean([x[1] for x in scores])),
    str(np.mean([x[2] for x in scores]))])"""

def sorted_features(V, X, y, topN):
  iv = {v:k for k, v in V.items()}
  chi2_scores = chi2(X, y)[0]
  top_features = [(x[1], iv[x[0]], x[0]) 
    for x in sorted(enumerate(chi2_scores), 
    key=operator.itemgetter(1), reverse=True)]
  print "TOP 10 FEATURES FOR"
  for top_feature in top_features[0:10]:
    print "%7.3f  %s (%d)" % (top_feature[0], top_feature[1], top_feature[2])
  return [x[1] for x in top_features]

def main():
  logger.info("Start")
  start = time.clock()

  texts, ys = read_data("/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/reviews_objLV.json")
  print ",".join(["attrtype", "nfeats", "accuracy", "precision", "recall"])
  
  y = ys
  V, X = vectorize(texts)
  cross_validate(X, y, -1)
  sorted_feats = sorted_features(V, X, y, 30)
  for nfeats in [1000, 3000, 10000, 30000, 100000]:
    V, X = vectorize(texts, sorted_feats[0:nfeats])
    cross_validate(X, y, nfeats)
  
  elapsed = (time.clock() - start)      
  logger.info("done in %d secs"%int(elapsed))

if __name__ == "__main__":
  main()
