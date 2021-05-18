# why use? with the kernal, it handles non linear relationship

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/uatozhandson/2_7/salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), )

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X, y)

# no try to predict level 6.5
x_interest = sc_X.transform([[6.5]])
y_interest = svr.predict(x_interest)
# now inverse y
final_y = sc_y.inverse_transform(y_interest)



# scatter the real.
# plot the pred
X_org = dataset.iloc[:, 1:-1].values
plt.scatter(X_org, sc_y.inverse_transform(y), color='red')
plt.plot(X_org, sc_y.inverse_transform(svr.predict(X)), color='blue')
plt.show()


## same thing. but smoth lines
X_plot = np.arange(min(X_org), max(X_org), 0.1)
X_plot = X_plot.reshape(len(X_plot),1)
plt.scatter(X_org, sc_y.inverse_transform(y), color='red')
plt.plot(X_plot, sc_y.inverse_transform(svr.predict(sc_X.transform(X_plot))), color='blue')
plt.show()