import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = np.array(list(range(1, 50)))
y = np.array([
   -1.07312323, -1.12360264, -1.16848888, -1.21237286, -1.24931163,
   -1.24563078, -1.25029589, -1.25804974, -1.26992981, -1.2759396 ,
   -1.31707672, -1.28845207, -1.2553561 , -1.21670196, -1.17405228,
   -1.13823657, -1.10201293, -1.0652651 , -1.01830663, -0.95872599,
   -0.86864519, -0.77287454, -0.67380868, -0.56936508, -0.47234488,
   -0.38025164, -0.28073984, -0.17953134, -0.08026437,  0.01376177,
    0.09177617,  0.15270399,  0.2005737 ,  0.23841612,  0.2860362 ,
    0.34606907,  0.39385415,  0.44154466,  0.49050035,  0.5338063 ,
    0.58003198,  0.61416929,  0.59416923,  0.56887929,  0.53366038,
    0.4907952 ,  0.45338928,  0.40975728,  0.35098762
])

# transforming the data to include another axis
print(x)
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

plt.scatter(x, y, s=10)
plt.plot(x, y_poly_pred, color='m')
plt.show()