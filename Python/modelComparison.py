# -*- coding: utf-8 -*-
import json
import sys,os
from datetime import datetime

import random
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
INFILE_USER_IDS = 'users_with_mt10_friendsLV.txt'
INFILE_USERS = 'usersLV.json'
#INFILE_REVIEWS = 'reviews_sentiment_naive_bayesLV.json'
INFILE_REVIEWS_TRAIN = 'reviews_test_objLV.json'

INFILE_RANK_BUSINESS_PAGERANK = 'out_rank_business_trainLV-PageRank.json'
INFILE_RANK_BUSINESS_INDEGREE = 'out_rank_business_trainLV-inDegreeCentrality.json'
INFILE_RANK_RATING = 'restaurants_trainLV.csv'

OUTFILE = 'out.json'

TOTAL_USERS = 10
NUM_SAMPES = 2   

"""
It reads user_score, reviews and reviews_pred_naive_bayesLV.
for each user
    get list of user's reviews
    for each pair of review
        write to outfile user_id, user_score, rev1, rev1_stars,rev1_votes, rev1_pred_proba, rev1_pred_log_proba,rev2, rev2_stars,rev2_votes, rev2_pred_proba, rev2_pred_log_proba,
"""


class ModelComparison:

    def __init__(self,data_dir=DATA_DIR, infile_user_ids=INFILE_USER_IDS, infile_users=INFILE_USERS, infile_reviews=INFILE_REVIEWS_TRAIN, infile_rank_pagerank=INFILE_RANK_BUSINESS_PAGERANK, infile_rank_indegree=INFILE_RANK_BUSINESS_INDEGREE):
        self.data_dir = data_dir
        self.infile_user_ids = infile_user_ids
        self.infile_users = infile_users
        self.infile_reviews = infile_reviews
        self.infile_rank_pagerank = infile_rank_pagerank
        self.infile_rank_indegree = infile_rank_indegree

    def loadUserIds(self):
        inFile = os.path.join(self.data_dir, self.infile_user_ids)      
        ret = list()
        with open(inFile) as f:
            for line in f:
                line = line.strip()
                ret.append(line)
        
        random.shuffle(ret)
        logger.info("loaded file '%s'"%(inFile))
        return set(ret[:TOTAL_USERS])  


    def loadUsersFriends(self, users_ids):
        inFile = os.path.join(self.data_dir, self.infile_users)
        retD = dict()
        retS = set()
        fi = open(inFile)
        data = json.loads(fi.read())
        fi.close()
        for x in data:
            user_id = x['user_id']     
            if user_id in users_ids:
                retS.add(user_id)
                retD[user_id] = x['friends']
                #retS.union(x['friends'])
                for fr in x['friends']:
                    retS.add(fr)

        logger.info("loaded file '%s'"%(inFile))                
        return retD,retS


    def loadUsersBusiness(self,users_friends_set):
        inFile = os.path.join(self.data_dir, self.infile_reviews)
        retD = dict()
        with open(inFile) as f:
            for line in f:
                d = json.loads(line)
                user_id = d['user_id']
                if user_id in users_friends_set:
                    if user_id not in retD:
                        retD[user_id] = list()    
                    retD[user_id].append(d['business_id'])
        logger.info("loaded file '%s'"%(inFile))   
        return retD    

    def loadObjJson(self,infile):
        inFile = os.path.join(self.data_dir, infile)
        retL = list()
        with open(inFile) as f:
            for line in f:
                retL.append(json.loads(line))            
        logger.info("loaded file '%s'"%(inFile))   
        return retL

    def saveJson(self, outJsonFile, data):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        for x in data:
            fo.write("%s\n"%json.dumps(x))
        fo.close()
        logger.info("out file:%s"%outFile)
   
          

    def getBusinessFromUsers(self, sample, users_friends, users_business):
        retS = set()
        usersS = set()
        # get users ahd thier friends
        for user in sample:
            usersS.add(user)
            for fr in users_friends[user]:
                usersS.add(fr)

        # get business
        for u in usersS:
            if u in users_business:
                for b in users_business[u]:
                    retS.add(b)

        return retS                


    def main(self):
        logger.info("Start")
        start = time.clock()

        # random users_ids
        user_ids = self.loadUserIds()

        # users --> friends dict
        users_friends, users_friends_set = self.loadUsersFriends(user_ids)
        logger.info("len of users_friends_set:%d"%len(users_friends_set))

        # user --> business
        users_business = self.loadUsersBusiness(users_friends_set)
        logger.info("len of users_business:%d"%len(users_business))

        # rank business pagerank
        rank_pagerank = self.loadObjJson(self.infile_rank_pagerank)
        predicted_pagerank = [x['business_id'] for x in rank_pagerank]


         # rank business indegree
        rank_indegree = self.loadObjJson(self.infile_rank_indegree)
        predicted_indegree = [x['business_id'] for x in rank_indegree]

        all_users = users_friends.keys()
        print "ALL"
        pp(all_users)
        samples = np.split(np.array(range(TOTAL_USERS)), NUM_SAMPES)
        for chunk in samples:
            sample = [all_users[i] for i in chunk]
            pp(sample)
            actual = self.getBusinessFromUsers(sample, users_friends, users_business)
            print "ACTUAL"
            pp(actual)
            logger.info("len of actual:%d"%len(actual))

        
        elapsed = (time.clock() - start)
        logger.info("done in %d secs"%int(elapsed))



if __name__ == "__main__":
    mc = ModelComparison()
    mc.main()

