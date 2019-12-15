# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Sun Oct 20 04:25:30 2019

@author: Bill Morrisson Talla Jr, Kadidia Sow Diallo

"""

#Had wrong probabilities because it's Naive Bayes and not Gaussian NB

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



def sep_classes(dataset):
	separate = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separate):
			separate[class_value] = list()
		separate[class_value].append(vector)
	return separate

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 

def brief_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 

def sumrybyclass(dataset):
	separate = sep_classes(dataset)
	summaries = dict()
	for class_value, rows in separate.items():
		summaries[class_value] = brief_dataset(rows)
	return summaries

def gpdf(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def prob_pred(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= gpdf(row[i], mean, stdev)
	return probabilities



statistics = sumrybyclass(dataset)

for i in range(20):
    probab = prob_pred(statistics, dataset[i])
#print(probab)


np.savetxt("gnb_results.txt", probab, newline=" \n")

