# -*- coding: utf-8 -*-
"""
β-γ拡張
20210623-toyama

Created on Wed Jun 23 17:32:40 2021

@author: user01
"""


import numpy as np
import matplotlib.pyplot as plt


#--------------------
#import data
inputs = np.loadtxt("inputs_data.csv", delimiter=",")

plt.plot(inputs)
plt.show()




#-----------------------
#parameters

r = int(len(inputs)/10)

if r >= len(inputs):
    print("Error")

#1=sf,2=GF
filter_switch = 1

#filter_size
#filter_len = len(inputs)#/5
filter_len = 41





#-----------------------
#filter

sf_y = filter_len
sf_x = (int)((sf_y-1)/2 + 1)
ramuda = (len(inputs)-1)/5
sf = np.zeros(sf_y)
omomi = 0

#SF
for i in range(sf_x):
    
    if i == (sf_x-1):
        sf[i] = np.pi / ramuda * np.sin((np.sqrt(2) * np.pi * i) / ramuda + np.pi / 4) * np.exp((-np.sqrt(2) * np.pi * i) / ramuda) / 2
    else:
        sf[i] = np.pi / ramuda * np.sin((np.sqrt(2) * np.pi * i) / ramuda + np.pi / 4) * np.exp((-np.sqrt(2) * np.pi * i) / ramuda)
    
    if i != 0:
        sf[sf_y-i] = sf[i]
        omomi += sf[i]*2
    else:
        omomi += sf[0]
    
sf /= omomi
print("SF総和:",np.sum(sf))


for i in range(sf_x-1):
    if sf[1+i] != sf[sf_y-i-1]:
        print("False")      

#sf = np.fft.fftshift(sf)

plt.plot(sf)
plt.show()



#-----------------------
#b-r
#20210623 曲げなし

a = 1/(16*((np.sin(np.pi*5/len(inputs)))**4))
print("μ/Δx**3 :", a)

b_up = 0
b_down = 0
n = len(inputs)
r = r+1

#分子
for i in range(r):
    b1 = sf[i]
    b2 = -inputs[i]
    for j in range(i+1, r):
        b1 += 2*sf[j]
    for j in range(1, r+i):
        b2 += sf[j-i] * inputs[j]
    for j in range(1, r-i):
        b2 -= sf[j+i] * inputs[j]
    
    #T1計算
    
    b_up += b1 * b2
b_up *= (-1)

#分母
for i in range(r):
    b3 = sf[i]
    for j in range(i+1, r):
        b3 += 2*sf[j]
    
    #T2計算
    
    b_down += b3**2

b = b_up / b_down





r_up = 0
r_down = 0

#分子
for i in range(r):
    r1 = sf[i]
    r2 = -inputs[n-1-i]
    for j in range(i+1, r):
        r1 += 2*sf[j]
    for j in range(1, r+i):
        r2 += sf[j-i] * inputs[n-1-j]
    for j in range(1, r-i):
        r2 -= sf[j+i] * inputs[n-1-j]
    
    #T1計算
    
    r_up += r1 * r2
r_up *= (-1)

#分母
for i in range(r):
    r3 = sf[i]
    for j in range(i+1, r):
        r3 += 2*sf[j]
    
    #T2計算
    
    r_down += r3**2

r = r_up / r_down
print("β : ", b)
print("γ : ", r)
print("GF br_bunbo=3.981512, β=3.284931, γ=0.000000")
print("SF br_bunbo=14.882633, β=2.699508, γ=0.540349")