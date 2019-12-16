"""
Created on Sun Oct 20 04:25:30 2019

@author: Bill Morrisson Talla Jr, Kadidia Sow Diallo
"""
from sklearn.datasets import load_linnerud
import numpy as np
import matplotlib.pyplot as plt


data = load_linnerud()
#print(data)

#print(data['data'])
weight = [i[0] for i in data['target']]
waist = [i[1] for i in data['target']]
pulse = [i[2] for i in data ['target']]

chins = [i[0] for i in data['data']]
situps = [i[1] for i in data['data']]
jumps = [i[2] for i in data['data']]



def best_fit(X, Y):
    
    S = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(S, Y, rcond=None)[0]
    
    #print("y ={:.2f}x + {:.2f}".format(m, c))
    return m,c



fig, axs = plt.subplots(3, 3, figsize=(9,9))

m,c = best_fit(weight, chins)
#print("intercept of weight and chins: ", c)
#print("slope of weight and chins: ", m)
axs[0,0].scatter(weight, chins, color='gray')
fit = [c + m * x for x in weight]
axs[0, 0].plot(weight, fit)
axs[0,0].set(xlabel='weight')
axs[0,0].set(ylabel='chins')
axs[0,0].set_title("y =-0.08x + 24.35")
axs[0,0].grid()
plt.tight_layout()


m,c = best_fit(weight, situps)
#print("intercept of weight and situps: ", c)
#print("slope of weight and situps: ", m)
axs[0,1].scatter(weight, situps, color='gray')
fit = [c + m * x for x in weight]
axs[0,1].plot(weight, fit)
axs[0,1].set(xlabel='weight')
axs[0,1].set(ylabel='situps')
axs[0,1].set_title("y =-1.25x + 368.71")
axs[0,1].grid()
plt.tight_layout()


m,c = best_fit(weight, jumps)
#print("intercept of weight and jumps: ", c)
#print("slope of weight and jumps: ", m)
axs[0,2].scatter(weight, jumps, color='gray')
fit = [c + m * x for x in weight]
axs[0,2].plot(weight, fit)
axs[0,2].set(xlabel='weight')
axs[0,2].set(ylabel='jumps')
axs[0,2].set_title("y =-0.47x + 154.24")
axs[0,2].grid()
plt.tight_layout()


m,c = best_fit(waist, chins)
#print("intercept of waist and chins: ", c)
#print("slope of waist and chins: ", m)
axs[1,0].scatter(waist, chins, color='gray')
fit = [c + m * x for x in waist]
axs[1,0].plot(waist, fit)
axs[1,0].set(xlabel='waist')
axs[1,0].set(ylabel='chins')
axs[1,0].set_title("y =-0.91x + 41.72")
axs[1,0].grid()
plt.tight_layout()


m,c = best_fit(waist, situps)
#print("intercept of waist and situps: ", c)
#print("slope of waist and situps: ", m)
axs[1,1].scatter(waist, situps, color='gray')
fit = [c + m * x for x in waist]
axs[1,1].plot(waist, fit)
axs[1,1].set(xlabel='waist')
axs[1,1].set(ylabel='situps')
axs[1,1].set_title("y =-12.61x + 592.12")
axs[1,1].grid()
plt.tight_layout()


m,c = best_fit(waist, jumps)
#print("intercept of waist and jumps: ", c)
#print("slope of waist and jumps: ", m)
axs[1,2].scatter(waist, jumps, color='gray')
fit = [c + m * x for x in waist]
axs[1,2].plot(waist, fit)
axs[1,2].set(xlabel='waist')
axs[1,2].set(ylabel='jumps')
axs[1,2].set_title("y =-3.07x + 178.86")
axs[1,2].grid()
plt.tight_layout()


m,c = best_fit(pulse, chins)
#print("intercept of pulse and chins: ", c)
#print("slope of pulse and chins: ", m)
axs[2,0].scatter(pulse, chins, color='gray')
fit = [c + m * x for x in pulse]
#plt.subplot(3, 3, 7)
axs[2,0].plot(pulse, fit)
axs[2,0].set(xlabel='pulse')
axs[2,0].set(ylabel='chins')
axs[2,0].set_title("y =0.11x + 3.25")
axs[2,0].grid()
plt.tight_layout()


m,c = best_fit(pulse, situps)
#print("intercept of pulse and situps: ", c)
#print("slope of pulse and situps: ", m)
axs[2,1].scatter(pulse, situps, color='gray')
fit = [c + m * x for x in pulse]
axs[2,1].plot(pulse, fit)
axs[2,1].set(xlabel='pulse')
axs[2,1].set(ylabel='situps')
axs[2,1].set_title("y =1.95x + 36.00")
axs[2,1].grid()
plt.tight_layout()


m,c = best_fit(pulse, jumps)
#print("intercept of pulse and jumps: ", c)
#print("slope of pulse and jumps: ", m)
axs[2,2].scatter(pulse, jumps, color='gray')
fit = [c + m * x for x in pulse]
axs[2,2].plot(pulse, fit)
axs[2,2].set(xlabel='pulse')
axs[2,2].set(ylabel='jumps')
axs[2,2].set_title("y =0.25x + 56.36")
axs[2,2].grid()
plt.tight_layout()


