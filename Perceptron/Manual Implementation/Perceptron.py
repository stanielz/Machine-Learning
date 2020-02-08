import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

data = pd.read_csv("iris.data", sep=',')
data.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm',
                'class']
data_top = data.head()

result_list = []

for i in data.iloc[:, 4]:
    if i == 'Iris-setosa':
        result_list.append(1)
    elif i == 'Iris-versicolor':
        result_list.append(-1)
    else :
        result_list.append('None')

data['Result'] = result_list

sepal = np.array(data['sepal width in cm'])
petal = np.array(data['petal width in cm'])
features = list(zip(sepal, petal))[0:99]
w = np.array([0, 0])
x = np.dot(features, w)
results = []
count = 1
while count < 50:
    x = np.dot(features, w)
    s = 2 * random.randint(0, 48)
    if (int(x[s]) >= 0 and int(result_list[s]) < 0) or (int(x[s]) < 0 and int(result_list[s]) >= 0):
            w = w + result_list[s] * np.transpose(features[s])
            count = 0
    else:
            count += 1

x_validate = np.dot(features, w)
for i in range(0, 50):
    s = (2 * random.randint(0, 48)) + 1

plt.plot(x_validate, 'o', color='red')
plt.plot(result_list[0: 99], 'o', color='blue')

print(w)
plt.show()



