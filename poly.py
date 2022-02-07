import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Depickle
pickle_in = open('data/train.pkl', 'rb')
result = pickle.load(pickle_in)
result[:1]

# Shuffle the dataset
# np.random.shuffle(result)

# Separate the dataset
num_datasets = 16
train_sets = np.split(result, num_datasets)

# Setup
polynom = PolynomialFeatures(degree=2)

# Separate the features and labels
instances = 100
x = train_sets[0][:instances, 0]
x = np.sort(x)
y = train_sets[0][:instances, 1]
print(x)

# Convert X from a 1d array to a 2d array
x = x[:, np.newaxis]
y = y[:, np.newaxis]
print(x)
print(x.shape)
print(y.shape)

polynomial_features = PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

# Polynomial regression results
plt.scatter(x, y, s=10)
plt.plot(x, y_poly_pred, color='b')
plt.show()