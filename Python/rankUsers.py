# -*- coding: utf-8 -*-
import json
import sys,os
from datetime import datetime

import numpy as np
from sklearn import preprocessing
from pprint import pprint as pp
import time
from staticScoreUser import StaticScoreUser

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")


DATA_DIR='/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/'

INFILE_USR = 'usersLV.json'
OUTFILE_USR = 'usersLV_score.json'


class RankUsers:

    def __init__(self,data_dir=DATA_DIR, infile=INFILE_USR):
        self.data_dir = data_dir
        self.infile = infile
     
        self.data = dict()


    def loadJson(self):
        start = time.time()
        inFile = os.path.join(self.data_dir, self.infile)
        with open(inFile) as f:
            self.data = json.load(f)
        end = time.time()
        logger.info("loaded user file '%s' [time:%2.2f secs]"%(inFile,end-start))    

    def saveJson(self, outJsonFile, data):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        for x in data:
            fo.write("%s\n"%json.dumps(x))
        fo.close()
        logger.info("out file:%s"%outFile)
    
    def calcStaticRank(self):
        count = 0
        ret = list()
        for u in self.data:
            ssu = StaticScoreUser(u)
            score_dict = ssu.score()
            if not score_dict:
                print "ERRORE"
            else:
                u['score'] = score_dict['score']
                ret.append(u)

        return ret

    def score_normalizer(self,data_in):
        logger.info("starting normalization")
        scores = np.array([x['score'] for x in data_in])  

        # scaling (minus mean / sd())
        scores_scaled = preprocessing.scale(scores)

        # normalize min max, range 0,100
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
        scores_minmax = min_max_scaler.fit_transform(scores)

        # enrich final data
        ret = list()
        for i,x in enumerate(data_in):
            x['scores'] = dict()
            x['scores']['score'] = scores[i]
            x['scores']['scaled'] = scores_scaled[i]
            x['scores']['minmax'] = scores_minmax[i]
            ret.append(x)

        logger.info("end normalization")        
        return ret    
        
    def main(self):
        logger.info("Start")
        start = time.clock()
        # load json
        self.loadJson()

        # calc static rank per user
        data_out = self.calcStaticRank()

        # normalize: scale and minmax
        data_out_normalized = self.score_normalizer(data_out)

        # save json
        self.saveJson(OUTFILE_USR, data_out_normalized)
        
        elapsed = (time.clock() - start)
        
        print "done in %d secs"%int(elapsed)



if __name__ == "__main__":
    ru = RankUsers()
    ru.main()

