# -*- coding: utf-8 -*-
import json
import sys,os
from datetime import datetime

import numpy as np
import networkx as nx


from sklearn import preprocessing
from pprint import pprint as pp
import time
from staticScoreUser import StaticScoreUser
import itertools

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")

PRINT_EVERY = 30000
MIN_WEIGHT = 0.001

DATA_DIR='/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/'
INFILE = 'pair_reviewsLV.json'
OUTFILE = 'rank_businessLV.json'

"""
It reads user_score, reviews and reviews_pred_naive_bayesLV.
for each user
    get list of user's reviews
    for each pair of review
        write to outfile user_id, user_score, rev1, rev1_stars,rev1_votes, rev1_pred_proba, rev1_pred_log_proba,rev2, rev2_stars,rev2_votes, rev2_pred_proba, rev2_pred_log_proba,
"""


class RankBusiness:

    def __init__(self,data_dir=DATA_DIR, infile=INFILE):
        self.data_dir = data_dir
        self.infile = infile
        self.G=nx.Graph()
        self.errors = 0

    """def loadDictJson(self, infile, k):
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
        return ret"""

    def saveFileJson(self, outJsonFile, data):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        json.dump(data,fo,indent=3)
        fo.close()

    def getVotes(self, rev):
        return rev['cool']+rev['funny']+rev['useful']    
   
    def algo(self,pair):
        rev1 = pair['rev1']
        rev2 = pair['rev2']

        diff_stars = rev1['y'] - rev2['y']
        if  diff_stars > 0:
            n1 = rev1['business_id']
            n2 = rev2['business_id']
            w = diff_stars
        elif diff_stars < 0:
            n1 = rev2['business_id']
            n2 = rev2['business_id']
            w = -diff_stars
            ret = (rev2['business_id'], rev1['business_id'], -diff_stars)
        elif diff_stars == 0:
            # use sentiment
            diff_sentiment = rev1['y_pred_proba'] - rev2['y_pred_proba']
            if  diff_sentiment > 0:
                n1 = rev1['business_id']
                n2 = rev2['business_id']
                w = 1
            elif diff_sentiment < 0:
                n1 = rev1['business_id']
                n2 = rev2['business_id']
                w = 1
            elif  diff_sentiment == 0:
                diff_votes = self.getVotes(rev1) - self.getVotes(rev1)
                if  diff_votes > 0:
                    n1 = rev1['business_id']
                    n2 = rev2['business_id']
                    w = 1
                elif  diff_votes < 0:
                    n2 = rev2['business_id']
                    n1 = rev2['business_id']
                    w = 1
                elif  diff_votes == 0:
                    n1 = rev1['business_id']
                    n2 = rev2['business_id']
                    w = MIN_WEIGHT
                    logger.info("MIN_WEIGHT: %s - %s"%(n1,n2))
                    self.errors += 1

        return (n1,n2,float(w))
          

    
    def main(self):
        logger.info("Start")
        start = time.clock()
        # create graph
        count = 0

        inFile = os.path.join(self.data_dir, self.infile)
        logger.info("creating graph from file '%s'"%(inFile))
        start_build_graph = time.clock()
        with open(inFile) as f:
            for i,line in enumerate(f):
                pair = json.loads(line.strip())
                #pp(pair)
                edge = self.algo(pair)
                self.G.add_weighted_edges_from([edge])
                count += 1

                if count % PRINT_EVERY == 0:
                    logger.info("Working on %d"%i)

        end_build_graph = time.clock()
        logger.info("graph built in %d secs"%int(end_build_graph-start_build_graph))

        logger.info("start pagerank")
        start_pagerank = time.clock()
        prank = nx.pagerank(self.G)
        sort_prank = sorted(prank.items(), key=lambda x:x[1], reverse=True)
        end_pagerank = time.clock()
        logger.info("end pagerank in %d secs"%(end_pagerank - start_pagerank))
        self.saveFileJson(OUTFILE,sort_prank)
        
        logger.info("count: %d"%(count))
        logger.info("errors: %d - ratio:%2.2f"%(self.errors, 100.0*float(self.errors)/float(count)))
 
        # see http://stackoverflow.com/questions/14563440/calculating-eigenvector-centrality-using-networkx
        
        elapsed = (time.clock() - start)
        logger.info("done - total time:%d secs"%int(elapsed))



if __name__ == "__main__":
    rb = RankBusiness()
    rb.main()

