# -*- coding: utf-8 -*-
import json
import sys,os
from datetime import datetime

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

    def saveJson(self, outJsonFile,data_out):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        json.dump(data_out,fo,indent=3)
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
        
    def main(self):
        logger.info("Start")
        start = time.clock()
        # load json
        self.loadJson()

        # calc static rank per user
        data_out = self.calcStaticRank()

        # save json
        self.saveJson(OUTFILE_USR, data_out)
        
        elapsed = (time.clock() - start)
        
        print "done in %d secs"%int(elapsed)

if __name__ == "__main__":
    ru = RankUsers()
    ru.main()
