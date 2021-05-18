
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/uatozhandson/2_7/salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), )

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# predict
regressor.predict([[6.5]])
# answer is the the same at 6.  closest point

X_plot = np.arange(min(X), max(X), 0.1)
X_plot = X_plot.reshape(len(X_plot), 1)
plt.scatter(X, y, color='red')
plt.plot(X_plot, regressor.predict(X_plot), color='blue')
plt.show()