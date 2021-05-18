import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_file= '/Users/rchann/Documents/poc/edu_machine_learning_sample_code/uatozhandson/4clustering/Mall_Customers.csv'

dataset = pd.read_csv(data_file)

# really should be 4 but use column 3,4
X = dataset.iloc[:, -3:].values

# alternative
# X = dataset.iloc[:, [3,4]].values

# for row in X:
#     print(row)
#     if row[1] == 'Male':
#         row[1] = 1
#     else:
#         row[1] = 0



from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i, init='k-means++')
    k_means.fit(X)
    # inertia is wcss
    wcss.append(k_means.inertia_)



plt.plot(range(1,11), wcss)
plt.xlabel("num cluster")
plt.ylabel("wcss")
plt.show()


k_means = KMeans(n_clusters=5, init='k-means++')
y_kmeans = k_means.fit_predict(X)


plt.scatter(X[y_kmeans == 0, 1], X[y_kmeans == 0 , 2], s=100, c='red', label='cluster 1')
plt.scatter(X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s=100, c='blue', label='cluster 2')
plt.scatter(X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s=100, c='green', label='cluster 3')
plt.scatter(X[y_kmeans == 3, 1], X[y_kmeans == 3, 2], s=100, c='blue', label='cluster 4')
plt.scatter(X[y_kmeans == 4, 1], X[y_kmeans == 4, 2], s=100, c='cyan', label='cluster 5')
plt.scatter(k_means.cluster_centers_[:,1], k_means.cluster_centers_[:,2], s=300, c='pink', label='centriods')
plt.show()