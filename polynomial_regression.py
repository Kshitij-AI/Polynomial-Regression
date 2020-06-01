# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
#upper bound is excluded and we need x to be in matrix always
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''
# Data is very small

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
linreg1 = LinearRegression()
linreg1.fit(x, y)

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
linreg2 = LinearRegression()
linreg2.fit(x_poly, y)

# Visualising the linear Regression results
plt.scatter(x, y, color = "red")
plt.plot(x, linreg1.predict(x), color = "blue")
plt.title("Linear Regression")
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = "red")
plt.plot(x, linreg2.predict(poly_reg.fit_transform(x)), color = "blue")
plt.title("Polynomial Regression")
plt.show()

# Predicting a new result with Linear Regression
linreg1.predict(np.array([[6.5]]))

# Predicting a new result with Linear Regression
linreg2.predict(poly_reg.fit_transform([[6.3]]))
