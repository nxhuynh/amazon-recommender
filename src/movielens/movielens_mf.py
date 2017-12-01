# https://nipunbatra.github.io/blog/2017/nnmf-tensorflow.html

import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error

INPUT_PATH = 'data/users_products_matrix_trimmed.csv'

np.random.seed(0)

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# read users x products matrix
print("Reading input...")
start_time = time.time()
A_orig_df = pd.read_csv(INPUT_PATH, dtype=np.float16, na_values=0.0)
print("Finished reading input in " + str(time.time() - start_time) + "s")


# Mask 0 entries
start_time = time.time()
A_df_masked = A_orig_df.copy()
print(A_df_masked.loc[0])
np_mask = A_df_masked.notnull()
print("Finished masking data in " + str(time.time() - start_time) + "s")

# Basic Tensorflow setup
# Boolean mask for computing cost only on valid (not missing) entries
start_time = time.time()
tf_mask = tf.Variable(np_mask.values)

A = tf.constant(A_df_masked.values)
shape = A_df_masked.values.shape
print("Finished basic TF setup in " + str(time.time() - start_time) + "s")

#latent factors
rank = 3 

# Initializing random H and W
start_time = time.time()
temp_H = np.random.randn(rank, shape[1]).astype(np.float16)
temp_H = np.divide(temp_H, temp_H.max())

temp_W = np.random.randn(shape[0], rank).astype(np.float16)
temp_W = np.divide(temp_W, temp_W.max())

H =  tf.Variable(temp_H)
W = tf.Variable(temp_W)
WH = tf.matmul(W, H)
print("Finished init factor vectors in " + str(time.time() - start_time) + "s")

#cost of Frobenius norm
cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))

# Learning rate
lr = 0.001
# Number of steps
steps = 1000
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()

# Clipping operation. This ensure that W and H learnt are non-negative
clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
clip = tf.group(clip_W, clip_H)

start_time = time.time()
print("Starting TF main loop...")
steps = 1000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        sess.run(train_step)
        sess.run(clip)
        if i%100==0:
            print("\nCost: %f" % sess.run(cost))
            print("*"*40)
    learnt_W = sess.run(W)
    learnt_H = sess.run(H)

print("Finished TF main loop in " + str(time.time() - start_time) + "s")

# print factor vectors
print('H vector');
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(learnt_H)

print('W vector')
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(learnt_W)

pred = np.dot(learnt_W, learnt_H)
pred_df = pd.DataFrame(pred)
"""
print('Reconstructed matrix');
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(pred_df.round())

print('original matrix')
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(A_orig_df)
"""

# calculate MSE
start_time = time.time()
A_orig_df = pd.read_csv(INPUT_PATH, index_col=False)
mse = get_mse(pred, A_orig_df.values)
print("MSE = " + str(mse))
print("Finished calculating MSE in " + str(time.time() - start_time) + "s")

