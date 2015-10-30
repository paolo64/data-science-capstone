# -*- coding: utf-8 -*-
import json
import sys,os
#from pandas.io.json import json_normalize
import pandas as pd

from pprint import pprint as pp
import time

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")


DATA_DIR='/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/'

"""INFILES = {
    'business':'yelp_academic_dataset_business.json',
    'reviews':'yelp_academic_dataset_review.json',
    'uesr':'yelp_academic_dataset_user.json'}"""

INFILES = {
    'business':'yelp_academic_dataset_business.json',
    }
OUTFILE = ''

BUSINESS_FILTER = {'categories':set(["Restaurants"])}


class FilterJsonFiles:

    def __init__(self,data_dir=DATA_DIR, infiles=INFILES):
        self.data_dir = data_dir
        self.infiles = infiles
        self.data = dict()


    @staticmethod
    def loadJson(inJsonFile):
        # read in yelp data
        yelp_data = list()
        with open(inJsonFile) as f:
            for line in f:
                yelp_data.append(json.loads(line))
        return yelp_data

    @staticmethod
    def loadJsonFilter(inJsonFile, criteria):
        # read in yelp data
        yelp_data = list()
        filter_key = criteria.keys()[0]
        filter_set = criteria[filter_key]
        with open(inJsonFile) as f:
            for line in f:
                d = json.loads(line)
                if filter_key in d:
                    cat_set = set(d[filter_key])
                    if len(cat_set.intersection(filter_set)) > 0:
                        yelp_data.append(d)
                else:
                    print "WARN:",line        
        return yelp_data

    @staticmethod
    def loadJsonFilterSet(inJsonFile, criteria):
        # read in yelp data
        yelp_data = list()
        filter_key = criteria.keys()[0]
        filter_set = set(criteria[filter_key])
        with open(inJsonFile) as f:
            for line in f:
                d = json.loads(line)             
                if d[filter_key] in filter_set:
                    yelp_data.append(d)
        return yelp_data

    @staticmethod
    def filterLV_Nevada(inList):
        ret = list()
        for x in inList:
            if x['state'] == 'NV' and x['city'] == "Las Vegas":
                ret.append(x)
        return ret            

    @staticmethod
    def dim(inList):
        return len(inList), len(inList[0].keys())

    def loadAllJson(self):   

        count = 0
        
        # load business
        curr_file = 'yelp_academic_dataset_business.json'
        logger.info("Loading file:'%s'"%curr_file)
        start = time.time()
        inJsonFile = os.path.join(self.data_dir,curr_file)
        logger.info("Jsonfile:'%s'"%inJsonFile)
        businessTmp = self.loadJsonFilter(inJsonFile,BUSINESS_FILTER)
        business = self.filterLV_Nevada(businessTmp)
        end = time.time()
        logger.info("business: %d x %d"%(self.dim(business)))


        # review
        curr_file = 'yelp_academic_dataset_review.json'
        logger.info("Loading file:'%s'"%curr_file)
        start = time.time()
        inJsonFile = os.path.join(self.data_dir,curr_file)
        logger.info("Jsonfile:'%s'"%inJsonFile)
        review_filter = {'business_id':[x['business_id'] for x in business]}
        review = self.loadJsonFilterSet(inJsonFile,review_filter)
        end = time.time()
        logger.info("review: %d x %d"%(self.dim(review)))


         # user
        curr_file = 'yelp_academic_dataset_user.json'
        logger.info("Loading file:'%s'"%curr_file)
        start = time.time()
        inJsonFile = os.path.join(self.data_dir,curr_file)
        logger.info("Jsonfile:'%s'"%inJsonFile)
        user_filter = {'user_id':[x['user_id'] for x in review]}
        user = self.loadJsonFilterSet(inJsonFile,user_filter)
        end = time.time()
        logger.info("user: %d x %d"%(self.dim(user)))


        logger.info("Loaded file:'%s' in %d secs"%(curr_file, int((end-start))))
    
    def main(self):
        logger.info("Start")
        start = time.clock()
        data = self.loadAllJson()
        #today = time.strftime("%Y%m%d")
        elapsed = (time.clock() - start)
        
        print "done in %d secs"%int(elapsed)

if __name__ == "__main__":
    fjf = FilterJsonFiles()
    fjf.main()
