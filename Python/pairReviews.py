# -*- coding: utf-8 -*-
import json
import sys,os
from datetime import datetime

import numpy as np
from sklearn import preprocessing
from pprint import pprint as pp
import time
from staticScoreUser import StaticScoreUser
import itertools

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")

PRINT_EVERY = 10000

DATA_DIR='/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/'
#INFILE_REVIEWS = 'reviews_sentiment_naive_bayesLV.json'
INFILE_REVIEWS = 'reviews_train_sentiment_naive_bayesLV.json'
INFILE_USER_SCORE = 'usersLV_score.json'
#OUTFILE = 'pair_reviewsLV.json'
OUTFILE = 'pair_reviews_trainLV.json'

"""
It reads user_score, reviews and reviews_pred_naive_bayesLV.
for each user
    get list of user's reviews
    for each pair of review
        write to outfile user_id, user_score, rev1, rev1_stars,rev1_votes, rev1_pred_proba, rev1_pred_log_proba,rev2, rev2_stars,rev2_votes, rev2_pred_proba, rev2_pred_log_proba,
"""


class PairReviews:

    def __init__(self,data_dir=DATA_DIR, infile_reviews=INFILE_REVIEWS, infile_user_scores=INFILE_USER_SCORE):
        self.data_dir = data_dir
        self.infile_reviews = infile_reviews
        self.infile_user_scores = infile_user_scores
        self.data = list()


    def loadDictJson(self, infile, k):
        start = time.time()
        ret = dict()
        inFile = os.path.join(self.data_dir, infile)
        logger.info("loading file '%s'"%(inFile))
        with open(inFile) as f:
            for i,line in enumerate(f):
                rec = json.loads(line.strip())
                ret[rec[k]] = rec

        end = time.time()
        logger.info("loaded file '%s' [time:%2.2f secs]"%(inFile,end-start))
        return ret

    def saveJson(self, outJsonFile, data):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        for x in data:
            fo.write("%s\n"%json.dumps(x))
        fo.close()
        logger.info("out file:%s"%outFile)
   
    def reviewsByUsers(self,reviews):
        ret = dict()
        for k in reviews.keys():
            rev = reviews[k]
            uid = rev['user_id']
            if uid not in ret:
                ret[uid] = list()

            ret[uid].append(rev['review_id'])    

        return ret
          

    def revCombimations(self, users, reviews, reviews_by_users, outfile):
        not_valid_users = 0
        users_no_reviews = 0
        count = 0
        outFile = os.path.join(self.data_dir,outfile)
        fo = open(outFile, 'w')
        # loop un users
        for k in users.keys():
            user = users[k]

            user_id = user['user_id']
            scores = user['scores']
            user_score_minmax = scores['minmax']
            user_score_scaled = scores['scaled']
            user_score_score = scores['score']
            if user_id not in reviews_by_users:
                logger.warn("user_id not in reviews [%s]"%user_id)
                users_no_reviews += 1
                
            else:    
                user_reviews = reviews_by_users[user_id]            
                len_user_reviews = len(user_reviews)
                if len(user_reviews) < 2:
                    logger.info("user:%s - num reviews:%d"%(user_id,len_user_reviews))
                    not_valid_users += 1
                    continue

                for comb in itertools.combinations(user_reviews, 2):
                    x = {
                        'user_id':user_id,
                        'user_score_score':user_score_score,
                        'user_score_minmax':user_score_minmax,
                        'user_score_scaled':user_score_scaled,
                        'rev1': reviews[comb[0]],
                        'rev2': reviews[comb[1]]
                    }
                    fo.write("%s\n"%json.dumps(x))
                    count += 1

                    if not count % PRINT_EVERY:
                        print count
            
        fo.close()
        logger.info("count:%d"%count)
        logger.info("not_valid_users:%d"%not_valid_users)
        logger.info("users_no_reviews:%d"%users_no_reviews)
        logger.info("out file:%s"%outFile)

    def main(self):
        logger.info("Start")
        start = time.clock()
        # load reviews
        reviews = self.loadDictJson(self.infile_reviews, 'review_id')

        # load user scores
        users = self.loadDictJson(self.infile_user_scores, 'user_id')

        # reviewsByUsers
        reviews_by_users = self.reviewsByUsers(reviews)

        # rev combinations
        self.revCombimations(users, reviews, reviews_by_users, OUTFILE)
        
        elapsed = (time.clock() - start)
        logger.info("done in %d secs"%int(elapsed))



if __name__ == "__main__":
    pr = PairReviews()
    pr.main()

