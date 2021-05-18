import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_file = '/Users/rchann/Documents/poc/edu_machine_learning_sample_code/uatozhandson/6reenforcement_learning/Ads_CTR_Optimisation.csv'

dataset = pd.read_csv(data_file)

# simulate picking i on iteration n.  Return reward
def select_result(i, n):
    return dataset.values[n, i]

import math
N = 10000
# num add
D = 10
ads_selected = []
number_selections = [0 for i in range(D)]
sum_rewards = [0 for i in range(D)]
total_rewards = 0
for n in range(N):
    ad_idx = 0
    best_upper_bound = 0
    for i in range(D):
        if number_selections[i] > 0:
            average_reward = sum_rewards[i] / number_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = float('inf')
        if upper_bound > best_upper_bound:
            ad_idx = i
            best_upper_bound = upper_bound
    ads_selected.append(ad_idx)
    number_selections[ad_idx] += 1
    reward = select_result(ad_idx,n)
    total_rewards += reward
    sum_rewards[ad_idx] += reward



# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()