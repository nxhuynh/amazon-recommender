# https://nipunbatra.github.io/blog/2017/nnmf-tensorflow.html

import numpy as np
import pandas as pd

INPUT_PATH = 'data/users_products_matrix.csv'

np.random.seed(0)

# read users x products matrix
print("Reading input...")
#A_orig_df = pd.read_csv(INPUT_PATH, dtype=np.float16, na_values=0.0).drop('user',1)
#A_orig_df = pd.read_csv(INPUT_PATH).drop('user',1)
A_orig_df = pd.read_csv(INPUT_PATH)
A_orig_df = A_orig_df.drop(A_orig_df.columns[0],axis=1)
A_orig_df.to_csv('data/users_products_matrix_trimmed.csv', index=False)
print("Finished reading input")

