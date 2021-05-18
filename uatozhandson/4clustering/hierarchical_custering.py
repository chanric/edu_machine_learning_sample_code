import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_file= '/Users/rchann/Documents/poc/edu_machine_learning_sample_code/uatozhandson/4clustering/Mall_Customers.csv'

dataset = pd.read_csv(data_file)

# really should be 4 but use column 3,4
X = dataset.iloc[:, -3:].values



import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('customers - observation point')
plt.xlabel('eulidean distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 1], X[y_hc == 0 , 2], s=100, c='red', label='cluster 1')
plt.scatter(X[y_hc == 1, 1], X[y_hc == 1, 2], s=100, c='blue', label='cluster 2')
plt.scatter(X[y_hc == 2, 1], X[y_hc == 2, 2], s=100, c='green', label='cluster 3')
plt.scatter(X[y_hc == 3, 1], X[y_hc == 3, 2], s=100, c='blue', label='cluster 4')
plt.scatter(X[y_hc == 4, 1], X[y_hc == 4, 2], s=100, c='cyan', label='cluster 5')
#plt.scatter(y_hc.cluster_centers_[:,1], y_hc.cluster_centers_[:,2], s=300, c='pink', label='centriods')
plt.show()