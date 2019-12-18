# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 04:25:30 2019

@author: Bill Morrisson Talla Jr, Kadidia Sow Diallo
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt

#initializing array with random number between 0's and 1's
n_elements = 10
#question 1A
vector_array=np.random.rand(1000, n_elements)

msub_vector_array = vector_array - np.tile(np.expand_dims(np.mean(vector_array,axis=1),axis=1),[1,n_elements])

corrmat = np.zeros([1000,1000])

for i in np.arange(0,1000):
    tilevec_i = np.tile(np.expand_dims(msub_vector_array[i,:],axis=0),[1000,1])
    corrmat[i,:] = np.sum(tilevec_i * msub_vector_array, axis = 1) \
        / (np.sqrt(np.sum(tilevec_i * tilevec_i , axis = 1)) * np.sqrt(np.sum(msub_vector_array * msub_vector_array, axis = 1)))

triangle = np.tril(corrmat)
vals = triangle[np.where(np.logical_and(triangle != 0, triangle < 0.999))]

hist = np.histogram(vals, 100)

plt.plot(hist[1][1:], hist[0])
