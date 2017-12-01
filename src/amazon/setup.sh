mkdir data
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Amazon_Instant_Video.csv -P data/
mv data/ratings_Amazon_Instant_Video.csv data/amazon_reviews.csv

# python preprocess.py

# python preprocess2.py
