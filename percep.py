import numpy as np
from math import sqrt
from math import pi
from math import exp
from sklearn.datasets import load_linnerud

data = load_linnerud()   

#data = linnerud['target']
target = data['target']
#target = linnerud['data']
features = data['data']

chins = [i[0] for i in features]

medchins = np.median(chins)
binchins = []
for i in chins:
    if i > medchins:
        binchins.append(0)
    else:
        binchins.append(1)

chins2d = np.reshape(binchins, (-1, 1))
#print(chins)
dataset = np.append(target, chins2d, axis=1)
#print(dataset)


weights = np.zeros([3,1])

for i in range(0, 1000):
	counter = 0
	converged = True

	for rowvalue in target:

		predVal = np.dot(rowvalue,weights)
		if predVal < 0:

			predicted = 0

		else:

			predicted = 1

		if predicted != binchins[counter]:

			converged = False

			if binchins[counter] == 0 :

			    weights = weights - np.expand_dims(rowvalue,1)

			else:

			    weights = weights + np.expand_dims(rowvalue,1)

		counter = counter + 1

	if converged == True:
		print("Error occurred")
		break

Predict = np.dot(target, weights)
np.savetxt("perceptron_results.txt", Predict, newline=" \n")
