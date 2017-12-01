import pandas as pd
from ExplicitMF import ExplicitMF
import time

INPUT_FILE_PATH = 'data/users_products_matrix_trimmed.csv'

start_time = time.time()
input_df = pd.read_csv(INPUT_FILE_PATH, index_col=False)
print("Finished reading inputs in " + str(time.time() - start_time) + "s")

start_time = time.time()
MF_SGD = ExplicitMF(input_df.values, 40, learning='sgd', verbose=True)
iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
MF_SGD.calculate_learning_curve(iter_array, input_df.values, learning_rate=0.001)
print("Finished SGD in " + str(time.time() - start_time) + "s")

# plot_learning_curve(iter_array, MF_SGD)

