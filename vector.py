import pandas as pd
import numpy as np
file_path = 'C:\\Users\\mones\\Downloads\\Lab Session1 Data.xlsx'
try:
    df = pd.read_excel(file_path, sheet_name="Purchase data")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
data_matrix = df[['Product 1', 'Product 2', 'Product 3']]
prices = df['Price']
dimensionality = data_matrix.shape[1]
num_vectors = data_matrix.shape[0]
if np.linalg.matrix_rank(data_matrix) == dimensionality:
    A_pseudo_inverse = np.linalg.pinv(data_matrix)
    cost_per_product = np.dot(A_pseudo_inverse, prices)
    print(f"Dimensionality of the vector space: {dimensionality}")
    print(f"Number of vectors in the vector space: {num_vectors}")
    print("Cost of each product using Pseudo-Inverse:")
    print(cost_per_product)
else:
    print("Warning: Matrix A is not full rank. Pseudo-inverse calculation might be unstable.")

