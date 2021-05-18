import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_file= '/Users/rchann/Documents/poc/edu_machine_learning_sample_code/uatozhandson/3_14logistic_regression/sna.csv'

dataset = pd.read_csv(data_file)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
classifier.fit(X_train, y_train)

# predict customer age 30, salary 870000
classifier.predict(sc_X.transform([[30, 87000]]))

y_pred= classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
