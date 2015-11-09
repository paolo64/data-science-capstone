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
import csv

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")

PRINT_EVERY = 30000
MIN_WEIGHT = 0.001
STAR_DIFF_W = 10

DATA_DIR='/data/DataScience_JohnsHopkins/yelp_dataset_challenge_academic_dataset/'
INFILE = 'pair_reviewsLV.json'
INFILE_RESTAURANTS = 'restaurantsLV.csv'
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
        self.G=nx.DiGraph()
        self.info = dict()
        self.errors = 0

    def readCvs(self, infile):
        inFile = os.path.join(self.data_dir,infile)

        ret = dict()

        with open(inFile) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ret[row['business_id']] = row
        logger.info("loaded file:%s"%inFile)
        return ret   

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

    def saveFileJson(self, outJsonFile, data):
        outFile = os.path.join(self.data_dir,outJsonFile)
        fo = open(outFile, 'w')
        json.dump(data,fo,indent=3)
        fo.close()
        logger.info("saved outfile:%s"%outFile)

    def getVotes(self, rev):
        return rev['cool']+rev['funny']+rev['useful']

    def calcWeight(self, x):
        if x > 3:
            return (x**2 - x - 1)
        else:
            return x

    def algo(self,pair):
        rev1 = pair['rev1']
        rev2 = pair['rev2']

        #pp(pair)

        diff_stars = rev1['y'] - rev2['y']
        if  diff_stars > 0:
            n1 = rev2['business_id']
            n2 = rev1['business_id']
            w = diff_stars * STAR_DIFF_W
        elif diff_stars < 0:
            n1 = rev1['business_id']
            n2 = rev2['business_id']
            w = -diff_stars*STAR_DIFF_W
            ret = (rev2['business_id'], rev1['business_id'], -diff_stars)
        elif diff_stars == 0:
            # use sentiment
            diff_sentiment = rev1['y_pred_proba'] - rev2['y_pred_proba']
            if  diff_sentiment > 0:          
                n1 = rev2['business_id']
                n2 = rev1['business_id']
                w = 1
            elif diff_sentiment < 0:
                n1 = rev1['business_id']
                n2 = rev2['business_id']
                w = 1
            elif  diff_sentiment == 0:
                diff_votes = self.getVotes(rev1) - self.getVotes(rev1)
                if  diff_votes > 0:
                    n1 = rev2['business_id']
                    n2 = rev1['business_id']     
                    w = 1
                elif  diff_votes < 0:
                    n1 = rev1['business_id']
                    n2 = rev2['business_id']
                    w = 1
                elif  diff_votes == 0:
                    n1 = rev1['business_id']
                    n2 = rev2['business_id']
                    w = MIN_WEIGHT
                    logger.info("MIN_WEIGHT: %s - %s"%(n1,n2))
                    self.errors += 1

        # user score
        w = float(w) * pair['user_score_minmax']
        #return (n1,n2,float(w))
        return (n2,n1,float(w)) 

    def enrichResult(self,inList):
        ret = list()
        for x in inList:
            tmp = list(x)
            info = self.info[x[0]]
            tmp.append(info)
            ret.append(tmp)
        return ret

    
    def main(self):
        logger.info("Start")
        start = time.clock()
        # create graph
        count = 0

        self.info  = self.readCvs(INFILE_RESTAURANTS)

        inFile = os.path.join(self.data_dir, self.infile)
        logger.info("creating graph from file '%s'"%(inFile))
        start_build_graph = time.clock()
        with open(inFile) as f:
            for i,line in enumerate(f):
                pair = json.loads(line.strip())
                #pp(pair)
                edge = self.algo(pair)
                #self.G.add_weighted_edges_from([edge])
                self.G.add_edge(edge[0], edge[1], weight=edge[2])
                count += 1

                if count % PRINT_EVERY == 0:
                    logger.info("Working on %d"%i)

        end_build_graph = time.clock()
        logger.info("graph built in %d secs"%int(end_build_graph-start_build_graph))

        # pagerank
        logger.info("start pagerank")
        start_pagerank = time.clock()
        #prank = nx.pagerank(self.G, alpha=0.85, max_iter=100)
        prank = nx.pagerank(self.G)
        sort_prank = sorted(prank.items(), key=lambda x:x[1], reverse=True)
        sort_prank = self.enrichResult(sort_prank)
        end_pagerank = time.clock()
        logger.info("end pagerank in %d secs"%(end_pagerank - start_pagerank))

        # in_degree_centrality
        logger.info("start in_degree_centrality")
        start_in_degree_centrality = time.clock()
        inDegreeCentrality = nx.in_degree_centrality(self.G)
        sort_inDegreeCentrality = sorted(inDegreeCentrality.items(), key=lambda x:x[1], reverse=True)
        sort_inDegreeCentrality = self.enrichResult(sort_inDegreeCentrality)
        end_in_degree_centrality = time.clock()
        logger.info("end in_degree_centrality in %d secs"%(end_in_degree_centrality - start_in_degree_centrality))

        data = {'pagerank':sort_prank, 'in_degree_centrality':sort_inDegreeCentrality}
        self.saveFileJson(OUTFILE,data)
        
        logger.info("count: %d"%(count))
        logger.info("errors: %d - ratio:%2.2f"%(self.errors, 100.0*float(self.errors)/float(count)))
 
        # see http://stackoverflow.com/questions/14563440/calculating-eigenvector-centrality-using-networkx
        
        elapsed = (time.clock() - start)
        logger.info("done - total time:%d secs"%int(elapsed))



if __name__ == "__main__":
    rb = RankBusiness()
    rb.main()

