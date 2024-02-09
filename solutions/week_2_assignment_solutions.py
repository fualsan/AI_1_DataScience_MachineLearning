import numpy as np

##### PART 1 #####

# ndarrays
# 1.
arr_1d = np.array([1, 2, 3, 4, 5])
print('1. ', arr_1d)

# 2.
arr_2d_zeros = np.zeros((3, 4))
print('2. ', arr_2d_zeros)

# 3.
arr_random = np.random.rand(7)
print('3. ', arr_random)

# Vectors
# 4.
vec_addition = np.array([1, 2, 3]) + np.array([4, 5, 6])
print('4. ', vec_addition)

# 5.
vec_dot_product = np.dot([2, 3, 1], [1, 2, 3])
print('5. ', vec_dot_product)

# Matrices
# 6.
identity_matrix = np.eye(3)
print('6. ', identity_matrix)

# 7.
matrix_A = np.array([[1, 2], [3, 4]])
matrix_B = np.array([[5, 6], [7, 8]])
matrix_multiplication = np.dot(matrix_A, matrix_B)
print('7. ', matrix_multiplication)

# 8.
matrix_C = np.array([[1, 2, 3], [4, 5, 6]])
matrix_transpose = matrix_C.T
print('8. ', matrix_transpose)

# Dot Product
# 9.
vec_dot_product = np.dot([3, 7], [7, 3])
print('9. ', vec_dot_product)

# 10.
vec_dot_product2 = np.dot([2, 4], [1, 3])
print('10.', vec_dot_product2)

# Elementwise Multiplication
# 11.
arr_elementwise_mult = np.array([2, 3, 4]) * 5
print('11.', arr_elementwise_mult)

# 12.
arr_elementwise_mult_vectors = np.array([1, 2, 3]) * np.array([4, 5, 6])
print('12.', arr_elementwise_mult_vectors)

# Matrix Multiplication
# 13.
matrix_D = np.array([[1, 2, 3], [4, 5, 6]])
matrix_E = np.array([[7, 8], [9, 10], [11, 12]])
matrix_mult_result = np.dot(matrix_D, matrix_E)
print('13.', matrix_mult_result)

# 14.
matrix_F = np.array([[2, 3], [4, 5]])
matrix_G = np.array([[1, 2], [3, 4]])
matrix_mult_result_2 = np.dot(matrix_F, matrix_G)
print('14.', matrix_mult_result_2)

# Generating ndarrays
# 15.
linear_spaced_array = np.linspace(1, 10, 5)
print('15.', linear_spaced_array)

# 16.
evenly_spaced_array = np.linspace(0, 1, 4)
print('16.', evenly_spaced_array)

# Slicing ndarrays
# 17.
second_element_H = np.array([10, 20, 30])[1]
print('17.', second_element_H)

# 18.
sliced_array_I = np.array([1, 2, 3, 4, 5])[2:5]
print('18.', sliced_array_I)

# Example Mathematical Functions
# 19.
mean_absolute_error = np.mean(np.abs(np.array([5, 7, 9]) - np.array([6, 8, 10])))
print('19.', mean_absolute_error)

# 20.
square_root_array = np.sqrt(np.array([4, 9, 16]))
print('20.', square_root_array)


##### PART 2 #####

import pandas as pd
import matplotlib.pyplot as plt

# Load the Black Friday Dataset
df = pd.read_csv('black_friday_dataset.csv')

# 1. Loading and Initial Exploration
# a.
print('First 5 rows of the dataset:')
print(df.head())

# b.
print('\nData types of each column:')
print(df.dtypes)

# c.
print('\nNumber of rows and columns in the dataset:')
print(df.shape)

# 2. Descriptive Statistics
# a.
print("\nSummary statistics for 'Purchase' column:")
print(df['Purchase'].describe())

# b.
print("\nUnique values in the 'Age' column:")
print(df['Age'].unique())

# c.
print('\nAverage purchase amount for each gender:')
print(df.groupby('Gender')['Purchase'].mean())

# 3. Handling Missing Data
# a.
print('\nNumber of missing values in each column:')
print(df.isnull().sum())

# b.
df['Product_Category_2'].fillna(df['Product_Category_2'].mean(), inplace=True)
df['Product_Category_3'].fillna(df['Product_Category_3'].median(), inplace=True)

# 4. Feature Scaling
# a.
df['Occupation'] = (df['Occupation'] - df['Occupation'].min()) / (df['Occupation'].max() - df['Occupation'].min())

# b.
df['Purchase'] = (df['Purchase'] - df['Purchase'].mean()) / df['Purchase'].std()

# 5. Plotting Data
# a.
plt.hist(df['Age'])
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# b.
plt.boxplot(df['Purchase'])
plt.title('Box Plot of Purchase Amount')
plt.ylabel('Purchase Amount')
plt.show()

# c.
avg_purchase_by_age = df.groupby('Age')['Purchase'].mean()
avg_purchase_by_age.plot(kind='bar', color='skyblue')
plt.title('Average Purchase Amount by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Purchase Amount')
plt.show()

# d.
plt.scatter(df['Occupation'], df['Purchase'], alpha=0.1)
plt.title('Scatter Plot of Occupation vs. Purchase')
plt.xlabel('Occupation')
plt.ylabel('Purchase Amount')
plt.show()

