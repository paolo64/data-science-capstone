# -*- coding: utf-8 -*-
import json
import sys,os
#from pandas.io.json import json_normalize
import pandas as pd
import numpy as np

from pprint import pprint as pp
import time

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")


DATA_DIR='/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/'

OUTFILE_REST = 'restaurantsLV.csv'
OUTFILE_REV = 'reviewsLV.json'
OUTFILE_OBJ_REV = 'reviews_objLV.json'
OUTFILE_OBJ_REV_TRAIN = 'reviews_train_objLV.json'
OUTFILE_OBJ_REV_TEST = 'reviews_test_objLV.json'
OUTFILE_USR = 'usersLV.json'
OUTFILE_USR_WITH_MORE_X_FRIENDS = 'users_with_mt%s_friendsLV.txt'

BUSINESS_FILTER = {'categories':set(["Restaurants"])}

MIN_NUM_OF_FRIENDS = 50
CUT_PERC_REVIEWS = 70


class FilterJsonFiles:

    def __init__(self,data_dir=DATA_DIR, infiles=None):
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

    def saveJson(self, outJsonFile, data):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        json.dump(data,fo,indent=3)
        fo.close()
        logger.info("generated out file: %s"%outFile)

    def saveObjJson(self, outJsonFile, data):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        for x in data:
            fo.write("%s\n"%json.dumps(x))
        fo.close()
        logger.info("generated out file: %s"%outFile)   

    def saveBusinessCsv(self, outCsvFile, data):
        outFile = os.path.join(self.data_dir,outCsvFile)
        fo = open(outFile, 'w')
        fo.write('"business_id","name","review_count","stars","pos","wstarsx1k"\n')
        for pos,x in enumerate(data):
            fo.write('"%s","%s",%d,%2.2f,%d,%2.2f\n'%(x['business_id'].encode('utf-8'),x['name'].encode('utf-8'),x['review_count'],x['stars'],pos,x['wstarsx1k']))
        fo.close()
        logger.info("generated out file: %s"%outFile)         

    @staticmethod
    def dim(inList):
        return len(inList), len(inList[0].keys())

    @staticmethod
    def enrichBusiness(inList):
        sum_review_count = float(sum([ x['review_count'] for x in inList]))

        ret = list()
        for x in inList:
            y = dict(x)
            y['wstarsx1k']=1000.0 * float(x['stars'])*float(x['review_count'])/sum_review_count
            ret.append(y)
        return ret   

    def loadRevPerDay(self, in_review):
        rev_per_day = dict()
        for x in in_review:
            date = x['date']
            if date not in rev_per_day:
                rev_per_day[date] = 0
            rev_per_day[date] += 1

        return sorted(rev_per_day.items(), key=lambda x:x[0])
        

    def findSplitDay(self, in_rev_per_day, perc):

        logger.info("total num of days:%d"%len(in_rev_per_day))
        numrev = sum([x[1] for x in in_rev_per_day ]) 
        cut = float(perc)*numrev/100.0

        cutDay = in_rev_per_day[0][0]
        cum = 0
        for i,x in enumerate(in_rev_per_day):
            cum += x[1]
            if cum > cut:
                break

        cutDay = in_rev_per_day[i-1][0]
        logger.info("cut day:%s for perc %d%%"%(cutDay,perc))

        print "cutDay:",cutDay
        print "x:",x,i,cum,cut

        return cutDay

    def splitReviewByDay(self, in_review, cutDay):
        review_train = list()
        review_test = list()

        for x in in_review:
            if x['date'] <= cutDay:
                review_train.append(x)
            else:
                review_test.append(x)

        return review_train, review_test


    def findAndSaveUsersWithMoreThanXFriends(self, user, min_num_of_friends, outfile): 
        outFile = os.path.join(self.data_dir,outfile)
        fo = open(outFile, 'w')
        for x in user:
            if len(x['friends']) >= min_num_of_friends:
                #print "NUM FRIENDS:%d"%len(x['friends'])
                fo.write('%s\n'%(x['user_id']))
        fo.close()
        logger.info("generated out file: %s"%outFile)    

    
    def loadAllJson(self):   
        count = 0      
        # load business
        curr_file = 'yelp_academic_dataset_business.json'
        logger.info("Loading file:'%s'"%curr_file)
        start = time.time()
        inJsonFile = os.path.join(self.data_dir,curr_file)
        logger.info("Jsonfile:'%s'"%inJsonFile)
        businessTmp = self.loadJsonFilter(inJsonFile,BUSINESS_FILTER)
        businessTmp2 = self.filterLV_Nevada(businessTmp)
        business = self.enrichBusiness(businessTmp2)
        sorted_business = sorted(business, key = lambda x:(x['stars'],x['review_count']), reverse=True)
        end = time.time()
        logger.info("business: %d x %d"%(self.dim(business)))
        logger.info("[time:%s]"%(end-start))


        # review
        curr_file = 'yelp_academic_dataset_review.json'
        logger.info("Loading file:'%s'"%curr_file)
        start = time.time()
        inJsonFile = os.path.join(self.data_dir,curr_file)
        logger.info("Jsonfile:'%s'"%inJsonFile)
        review_filter = {'business_id':[x['business_id'] for x in business]}
        review = self.loadJsonFilterSet(inJsonFile,review_filter)

        rev_per_day = self.loadRevPerDay(review)
        perc = CUT_PERC_REVIEWS
        cutDay = self.findSplitDay(rev_per_day, perc)
        review_train, review_test = self.splitReviewByDay(review, cutDay)
        
        end = time.time()
        logger.info("review: %d x %d"%(self.dim(review)))
        logger.info("[time:%s]"%(end-start))

         # user
        curr_file = 'yelp_academic_dataset_user.json'
        logger.info("Loading file:'%s'"%curr_file)
        start = time.time()
        inJsonFile = os.path.join(self.data_dir,curr_file)
        logger.info("Jsonfile:'%s'"%inJsonFile)
        user_filter = {'user_id':[x['user_id'] for x in review]}
        user = self.loadJsonFilterSet(inJsonFile,user_filter)
        self.findAndSaveUsersWithMoreThanXFriends(user, MIN_NUM_OF_FRIENDS, OUTFILE_USR_WITH_MORE_X_FRIENDS%MIN_NUM_OF_FRIENDS)

        end = time.time()
        logger.info("user: %d x %d"%(self.dim(user)))
        logger.info("[time:%s]"%(end-start))

        # save business csv file
        self.saveBusinessCsv(OUTFILE_REST, sorted_business)
        logger.info("saved restaurants file %s"%(OUTFILE_REST))

        # save reviewers
        self.saveJson(OUTFILE_REV, review)
        logger.info("saved reviews file %s"%(OUTFILE_REV))

        self.saveObjJson(OUTFILE_OBJ_REV, review)
        logger.info("saved reviews obj file %s"%(OUTFILE_OBJ_REV))

        self.saveObjJson(OUTFILE_OBJ_REV_TRAIN, review_train)
        logger.info("saved reviews training obj file %s"%(OUTFILE_OBJ_REV_TRAIN))

        self.saveObjJson(OUTFILE_OBJ_REV_TEST, review_test)
        logger.info("saved reviews testing obj file %s"%(OUTFILE_OBJ_REV_TEST))

        # save users
        self.saveJson(OUTFILE_USR, user)
        logger.info("saved user file %s"%(OUTFILE_USR))

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
