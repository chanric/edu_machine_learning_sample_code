import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

data_file = '/Users/rchann/Documents/poc/edu_machine_learning_sample_code/uatozhandson/6reenforcement_learning/Ads_CTR_Optimisation.csv'

dataset = pd.read_csv(data_file)

# simulate picking i on iteration n.  Return reward
def select_result(i, n):
    return dataset.values[n, i]

import math
N = 1000
# num ads
D = 10
ads_selected = []
number_of_rewards_1 = [0 for i in range(D)]
number_of_rewards_0 = [0 for i in range(D)]
total_rewards = 0
for n in range(N):
    ad = 0
    #math random is thea
    max_random = -1
    for i in range(D):
        alpha = number_of_rewards_1[i] + 1
        beta = number_of_rewards_0[i] + 1
        random_beta = random.betavariate(alpha, beta)
        #random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            ad = i
            max_random = random_beta
    reward = select_result(ad, n)
    total_rewards += reward
    if reward == 0:
        number_of_rewards_0[ad] += 1
    else:
        number_of_rewards_1[ad] += 1
    ads_selected.append(ad)








# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
