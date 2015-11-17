python Python/filterJsonFilesTest.py
IN:


OUT:
users_with_mt10_friendsLV.txt
restaurants_trainLV.csv
restaurantsLV.csv
reviewsLV.json
reviews_objLV.json
reviews_train_objLV.json
reviews_test_objLV.json
usersLV.json


----
python rankUsers.py
IN:

OUT:
usersLV_score.json


-------------
python sentimentNaiveBayes.py

OUT:
reviews_train_sentiment_naive_bayesLV.json


--------------
python pairReviews.py

OUT:
pair_reviews_trainLV.json

---------
python rankBusiness.py

OUT:
out_rank_business_trainLV-inDegreeCentrality.json
out_rank_business_trainLV-PageRank.json
