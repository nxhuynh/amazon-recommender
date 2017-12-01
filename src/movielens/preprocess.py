"""
convert Amazon review data into users x product matrix
"""
import numpy as np
import pandas as pd
#from scipy.sparse import csc_matrix

INPUT_PATH = 'data/movielens_ratings.dat'
OUTPUT_PATH = 'data/users_products_matrix.csv'
N_PRODUCTS = 1000

# read input from csv file
fields = ['user','product','rating','time']
input_df = pd.read_csv(INPUT_PATH, header=None, names=fields, sep='::')

# get list of unique users and most common N_PRODUCTS products
users = input_df['user'].unique()
print('num users: ' + str(len(users)))
products = input_df['product'].unique()
print('num products: ' + str(len(products)))

# create user x common_product matrix
df = pd.DataFrame(0, index=users, columns=products)
#print(df.head)
for i, row in input_df.iterrows():
    user = row['user']
    product = row['product']
    if i % 10000 == 0:
        print(str(100 * i / input_df.shape[0]) + "% completed")
    df.loc[user,product] = row['rating']

df.to_csv(OUTPUT_PATH)
