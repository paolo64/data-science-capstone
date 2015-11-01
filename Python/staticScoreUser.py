# -*- coding: utf-8 -*-
import json
import sys,os
from datetime import datetime


from pprint import pprint as pp
from pprint import pformat as pf
import time

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("capostone")


YEAR_DATASET = 2015
ELITE_FACTOR = 10.0

class StaticScoreUser:

    def __init__(self,user_data):
        self.user_data = user_data             

    def score(self):
        count = 0

        u = self.user_data
        if len(u['elite']) > 0:

            pp(u['yelping_since'])
            pp(u['elite'])
            count += 1
            date_since = datetime.strptime(u['yelping_since'], '%Y-%m').year
            date_for = YEAR_DATASET - date_since + 1
            len_elite = len(u['elite'])
            elite_years_perc = 1.0/ELITE_FACTOR
            if len_elite:
              elite_years_perc = float(len_elite)/float(date_for)

            review_count = u['review_count']
            review_count_per_year = float(review_count) / float(date_for)

            num_fans = u['fans']
            num_friends = len(u['friends'])
            fans_friends_ratio = float(num_fans) / float(num_friends)
            votes_useful = u['votes']['useful']
            votes_useful_ratio = float(votes_useful) / float(review_count)    

            # formula
            score = ELITE_FACTOR*elite_years_perc*(review_count_per_year + fans_friends_ratio + votes_useful_ratio)

            score_dict = {
              'score':score,
              'review_count_per_year':review_count_per_year,
              'fans_friends_ratio':fans_friends_ratio,
              'votes_useful_ratio':votes_useful_ratio,
              'review_count':review_count,
              'num_fans':num_fans,
              'num_friends':num_friends,
              'votes_useful':votes_useful,
              'date_for':date_for,
              'len_elite': len_elite,
              'elite_years_perc':elite_years_perc
              }

            msg = pf(score_dict).replace('\n', '')
            logger.info("[%s] score:%s - %s"%(u['user_id'], score_dict['score'], msg))
            return score_dict
  
###########################################################  
if __name__ == "__main__":
    userDict ={
      "yelping_since": "2004-10",
      "votes": {
         "funny": 6849,
         "useful": 12642,
         "cool": 9837
      },
      "user_id": "rpOyqD_893cqmDAtJLbdog",
      "name": "Jeremy",
      "elite": [
         2005,
         2006,
         2007,
         2008,
         2009,
         2010,
         2011,
         2012,
         2013,
         2014,
         2015
      ],
      "type": "user",
      "compliments": {
         "profile": 110,
         "cute": 209,
         "funny": 561,
         "plain": 921,
         "writer": 290,
         "list": 37,
         "note": 589,
         "photos": 287,
         "hot": 1032,
         "more": 129,
         "cool": 1521
      },
      "fans": 1012,
      "average_stars": 3.64,
      "review_count": 1233,
      "friends": ['']
    }
    ssu = StaticScoreUser(userDict)
    ret = ssu.score()
    pp(ret)
