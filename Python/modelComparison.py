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
import csv
import ml_metrics as ml

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

TOTAL_USERS = 10000
NUM_SAMPES = 100
K_MAP = 30  

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

    def readCvs(self, infile):
        inFile = os.path.join(self.data_dir,infile)
        ret = dict()
        retL = list()

        with open(inFile) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ret[row['business_id']] = row
                retL.append(row['business_id'])
        logger.info("loaded file:%s"%inFile)
        return ret, retL   

    def loadUserIds(self, static=False):
        inFile = os.path.join(self.data_dir, self.infile_user_ids)      
        ret = list()
        with open(inFile) as f:
            for line in f:
                line = line.strip()
                ret.append(line)
        
        random.shuffle(ret)
        logger.info("loaded file '%s'"%(inFile))
        """ret = [u'YU0QdsIw-XvELsqxnujFMw',
                u'xIGQ9_TjQU3nawhizqxmXA',
                u'pqBcK71JaJGXUHceYtkNpA',
                u'6yc1WfoPnb1TIK1a1RavMw',
                u'zB80pJq2F7x0ZpmrDIE7TA',
                u'hzIUraWIydhiEoPBBryP7w',
                u'VIUqdgqoqnii4OKs5-01mA',
                u'jfQi2X4dyAHaxf63irqQGw',
                u'-Eg2U9-Vf1TjwEy_eqmnEA',
                u'Pn1H5qHdisZMzsBbMR6KGA']"""
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
        #print "SAMPLE:"
        #pp(sample)
        for user in sample:
            usersS.add(user)
            for fr in users_friends[user]:
                usersS.add(fr)

        # get business
        error = 0
        for u in usersS:
            if u in users_business:
                for b in users_business[u]:
                    retS.add(b)
            else:
                error +=1 
                #logger.error(u)    

        logger.info("num users_no_business:%d"%error)
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

        # rating
        business_rating_dict, predicted_rating  = self.readCvs(INFILE_RANK_RATING)

        # rank business indegree
        rank_indegree = self.loadObjJson(self.infile_rank_indegree)
        predicted_indegree = [x['business_id'] for x in rank_indegree]

        all_users = users_friends.keys()
        pp(all_users)
        samples = np.split(np.array(range(TOTAL_USERS)), NUM_SAMPES)
        maps = {'pagerank':[],'indegree':[],'rating':[]}
        for i,chunk in enumerate(samples):
            logger.info("sample: %d"%i)
            sample = [all_users[x] for x in chunk]
            actual = self.getBusinessFromUsers(sample, users_friends, users_business)
            logger.info("len of actual:%d"%len(actual))

            map_pagerank = ml.mapk([actual], [predicted_pagerank], k=K_MAP)
            logger.info("map_pagerank[%d]:%6.6f"%(i,map_pagerank))
            maps['pagerank'].append(map_pagerank)
            
            map_indegree = ml.mapk([actual], [predicted_indegree], k=K_MAP)
            logger.info("map_indegree[%d]:%6.6f"%(i,map_indegree))
            maps['indegree'].append(map_indegree)

            map_rating = ml.mapk([actual], [predicted_rating], k=K_MAP)
            logger.info("map_rating[%d]:%6.6f"%(i,map_rating))
            maps['rating'].append(map_rating)

        
        pp(maps)

        maps_stats = {'pagerank':{},'indegree':{},'rating':{}}
        for x in maps.keys():
            maps_stats[x]['mean'] = np.mean(maps[x])
            maps_stats[x]['std'] = np.std(maps[x])

        pp(maps_stats)

        elapsed = (time.clock() - start)
        logger.info("done in %d secs"%int(elapsed))



if __name__ == "__main__":
    mc = ModelComparison()
    mc.main()

