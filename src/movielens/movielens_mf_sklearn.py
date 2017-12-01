import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import time

INPUT_PATH = 'data/ml-100k/u.data'
N_LATENT_FACTORS = 40

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


def get_mse(pred, actual):
    # Ignore nonzero terms
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

#ead inputs
start_time = time.time()
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(INPUT_PATH, sep='\t', names=names)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print str(n_users) + ' users'
print str(n_items) + ' items'
print("Finished reading inputs in " + str(time.time() - start_time) + "s")

# generate train & test sets
start_time = time.time()
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
train, test = train_test_split(ratings)
print("Finished creating train & test sets in " + str(time.time() - start_time) + "s")

# MF with sklearn NMF
start_time = time.time()
model = NMF(n_components=N_LATENT_FACTORS, init='random', random_state=0)
W_train = model.fit_transform(train)
print("Finished NMF in " + str(time.time() - start_time) + "s")
H = model.components_;
W_test = model.transform(test)
pred_train = np.dot(W_train, H)
pred_test = np.dot(W_test, H)
mse_train = get_mse(pred_train, train)
mse_test = get_mse(pred_test, test)
print("Training MSE: " + str(mse_train))
print("Testing MSE: " + str(mse_test))

# plot_learning_curve(iter_array, MF_SGD)

