import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# A1. Matrix Operations
data_purchase = pd.read_excel(r"C:\\Users\\mones\\Downloads\\Lab Session1 Data.xlsx", sheet_name="Purchase data")
A = data_purchase.iloc[:, 1:4].to_numpy()
C = data_purchase.iloc[:, 4].to_numpy()

# A1. Activities
dimensionality = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A)
pseudo_inverse_A = np.linalg.pinv(A)
cost_vector = np.dot(pseudo_inverse_A, C)

# Output A1 Results
print(f"Dimensionality of the vector space: {dimensionality}")
print(f"How many vectors exist in this vector space: {num_vectors}")
print(f"Rank of Matrix A: {rank_A}")
print("\nCost of each product available for sale:")
print(cost_vector)

# Regularization term (adjust the value as needed)
reg_term = 1e-5

# Calculate pseudo-inverse with regularization
pseudo_inverse_A = np.linalg.pinv(A.T @ A + reg_term * np.eye(A.shape[1])) @ A.T @ C

# A2. Calculate Model Vector X
X = pseudo_inverse_A

# A3. Customer Classification
data_purchase['Customer_Class'] = np.where(data_purchase['Payment (Rs)'] > 200, 'RICH', 'POOR')

# A4. Stock Price Analysis
stock_data = pd.read_excel(r"C:\\Users\\mones\\Downloads\\Lab Session1 Data.xlsx", sheet_name="IRCTC Stock Price")

mean_price = stock_data['Price'].mean()
variance_price = stock_data['Price'].var()

wednesday_prices = stock_data[stock_data['Day'] == 'Tue']['Price']  # Change 'Tue' to your actual day
mean_wednesday_prices = wednesday_prices.mean()

april_prices = stock_data[stock_data['Month'] == 'Jun']['Price']  # Change 'Jun' to your actual month
mean_april_prices = april_prices.mean()

loss_probability = len(stock_data[lambda x: x['Chg%'] < 0]) / len(stock_data)
wednesday_profit_probability = len(wednesday_prices[lambda x: x > 0]) / len(wednesday_prices)
conditional_profit_probability = len(wednesday_prices[lambda x: x > 0]) / len(wednesday_prices)

print("Matrix A:")
print(A)
print("\nVector C:")
print(C)
print("\nPseudo-Inverse of A:")
print(pseudo_inverse_A)
print("\nModel Vector X:")
print(X)
print("\nCustomer Data with Classification:")
print(data_purchase[['Customer', 'Payment (Rs)', 'Customer_Class']])
print("\nStock Price Analysis:")
print(f"Mean Price: {mean_price}")
print(f"Variance of Price: {variance_price}")
print(f"Mean Price on Tuesday: {mean_wednesday_prices}")
print(f"Mean Price in June: {mean_april_prices}")
print(f"Probability of Making a Loss: {loss_probability}")
print(f"Probability of Making a Profit on Tuesday: {wednesday_profit_probability}")
print(f"Conditional Probability of Making Profit on Tuesday: {conditional_profit_probability}")

# Scatter plot
plt.scatter(stock_data['Day'], stock_data['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.show()
