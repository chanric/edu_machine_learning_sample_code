import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_file= '/Users/rchann/Documents/poc/edu_machine_learning_sample_code/uatozhandson/5association_rule_learning/Market_Basket_Optimisation.csv'
dataset = pd.read_csv(data_file, header=None)
transaction = []

for i in range(7501):
    transaction.append([str(dataset.values[i, j]) for j in range(20)])


from apyori import apriori
# try min confidence 0.8, then 0.4
# lifts should be above 3 otherwise it isn't a big deal
rules = apriori(transactions=transaction,
                min_support=3*7/7501,
                min_confidence=0.2,
                min_lift=3,
                min_length=2,
                max_length=2,
                )

results = list(rules)
print(results)


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

results_in_df = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


## Displaying the results non sorted
print(results_in_df)

## Displaying the results sorted by descending lifts
print(results_in_df.nlargest(n=10, columns='Lift'))
