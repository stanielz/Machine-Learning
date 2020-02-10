import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


def col_vector(array1, array2):
    col_vector_list = []
    for i in range(0, len(array1)):
        col_vector_list.append((array1[i], array2[i]))

    return col_vector_list


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
features = list(zip(sepal, petal))[0:100]
w = np.array([0, 0])
x = np.dot(features, w)
results = []
count = 1

#  tests and updates weight(w) until we have more than 50 consecutive predictions on learning data
#  s variable guarantees even index for learning
while count <= 49:
    x = np.dot(features, w)
    s = 2 * random.randint(0, 49)
    for s in range(0, 49):
        if (int(x[2 * s]) >= 0 and int(result_list[2 * s]) < 0) or (int(x[2 * s]) < 0 and int(result_list[2 * s]) >= 0):
            w = w + result_list[2 * s] * np.transpose(features[2 * s])
            count = 0
        else:
            count += 1

#  validates weight (w) calculated above on the odd numbered indexes (validation data).
#  s variable now guarantees odd index for validation
x_validate = []
x2_validate = []
for i in range(0, 49):
    if np.dot(features[2 * i + 1], w) >= 0:
        x_validate.append(1.1)
    else:
        x_validate.append(-1.1)
    x2_validate.append(result_list[2 * i + 1])

#  plots predicted data vs actual results
plt.plot(x2_validate, 'o', color='red', label='actual results')
plt.plot(x_validate, 'o', color='blue', label='predicted results')
plt.legend(loc='center')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('results1.png', dpi=100)
