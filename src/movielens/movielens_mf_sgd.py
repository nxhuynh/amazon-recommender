import numpy as np
import pandas as pd
from ExplicitMF import ExplicitMF
import time

INPUT_PATH = 'data/ml-100k/u.data'

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

start_time = time.time()
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(INPUT_PATH, sep='\t', names=names)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print str(n_users) + ' users'
print str(n_items) + ' items'

ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
train, test = train_test_split(ratings)
print("Finished reading inputs in " + str(time.time() - start_time) + "s")

start_time = time.time()
MF_SGD = ExplicitMF(train, 40, learning='sgd', verbose=True)
iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
print("Finished SGD in " + str(time.time() - start_time) + "s")

# plot_learning_curve(iter_array, MF_SGD)

