# -*- coding: utf-8 -*-
import json
import sys,os
from datetime import datetime

from pprint import pprint as pp
import time

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")


DATA_DIR='/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/'

INFILE_USR = 'usersLV.json'
OUTFILE_USR = 'TMPusersLV.json'
YEAR_DATASET = 2015


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

    def calcStaticRank(self):
        #for u in self.data[:3]:
        count = 0

        for u in self.data[:3:
            ssu = staticScoreUser(u)
            retScore = ssu.score()
        
        lenData = len(self.data)
        print "tot:%s, count:%d, perc:%2.2f"%(lenData, count, 100.0*float(count)/float(lenData))
    def main(self):
        logger.info("Start")
        start = time.clock()
        # load json
        self.loadJson()

        # calc static rank per user
        self.calcStaticRank()
        
        elapsed = (time.clock() - start)
        
        print "done in %d secs"%int(elapsed)

if __name__ == "__main__":
    ru = RankUsers()
    ru.main()
