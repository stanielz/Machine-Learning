import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#plt.plot(x, y, 'r')
data = pd.read_csv("iris.data", sep=',')
data.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm',
                'class']
data_top = data.head()
plt.plot(data.iloc[:99, 2], data.iloc[:99, 3],  'o', color='green')
#plt.plot(data.iloc[:, 3], 'o', color='black')
#plt.plot(data.iloc[:, 4], 'o', color='red')

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
petal_t = np.transpose(petal)
y = []
y.append(3)

for i in range(1,80):
    y.append(sepal[i] * petal[i])

x = np.linspace(0, 5, 100)
y = -x + 3
w1 = 1
w2 = 1



print(y[80])


#print(petal_transpose)
#plt.xlabel('Length')
#plt.show()
