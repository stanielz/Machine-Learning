import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

#  import data and name columns
data = pd.read_csv("iris.data", sep=',')
data.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm',
                'class']
data_top = data.head()

#  create list for classification
result_list = []

#  sorts results into positive and negative into result list
for i in data.iloc[:, 4]:
    if i == 'Iris-setosa':
        result_list.append(1)
    elif i == 'Iris-versicolor':
        result_list.append(-1)
    else:
        result_list.append('None')

#  appends result list to data framework
data['Result'] = result_list

#  creates array for Perceptron calculation
sepal = np.array(data['sepal width in cm'])
petal = np.array(data['petal width in cm'])
features = list(zip(sepal, petal))[0:99]
w = np.array([0, 0])
x = np.dot(features, w)
results = []
count = 1

#  tests and updates weight(w) until we have more than 50 consecutive predictions on learning data
#  s variable guarantees even index
while count < 50:
    x = np.dot(features, w)
    s = 2 * random.randint(0, 48)
    if (int(x[s]) >= 0 and int(result_list[s]) < 0) or (int(x[s]) < 0 and int(result_list[s]) >= 0):
        w = w + result_list[s] * np.transpose(features[s])
        count = 0
    else:
        count += 1

#  validates weight (w) calculated above on the odd numbered indexes (validation data).
x_validate = []
x2_validate = []
for i in range(0, 49):
    s = (2 * random.randint(0, 48)) + 1
    if np.dot(features[s], w) >= 0:
        x_validate.append(1.1)
    else:
        x_validate.append(-1.1)
    x2_validate.append(result_list[s])

#  plots predicted data vs actual results
plt.plot(x2_validate, 'o', color='red', label='actual results')
plt.plot(x_validate, 'o', color='blue', label='predicted results')
plt.legend(loc='center')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('results1.png', dpi=100)
