
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/uatozhandson/2_7/salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), )


from sklearn.ensemble import RandomForestClassifier
regressor =RandomForestClassifier(n_estimators=10, random_state=0)
regressor.fit(X, y)
regressor.predict([[6.5]])


X_plot = np.arange(min(X), max(X), 0.1)
X_plot = X_plot.reshape(len(X_plot), 1)
plt.scatter(X, y, color='red')
plt.plot(X_plot, regressor.predict(X_plot), color='blue')
plt.show()

y_pred = regressor.predict(X)
from sklearn.metrics import r2_score
# but doesn't make sense to do this.
r2_score(y, y_pred)