# -*- coding: utf-8 -*-
"""
β-γ拡張
20210623-toyama
Created on Wed Jun 23 17:32:40 2021
@author: user01



"""


import numpy as np
import matplotlib.pyplot as plt


#m →　拡張に使うフィルタサイズ
#k →　片側データの拡張サイズ
def b_r(inputs, k=0, b=0, rr=0, m=5, switch=1):
    
    print("β-γ start-------------")
    
    
    if b==0 and rr==0:
        
        #-----------------------
        #parameters
        
        r = int(len(inputs)/m)
        filter_len = r*2+1
        
        print("half_filter_size : L/", m, "→",r)
        if filter_len >= len(inputs):
            print("Error")
        
        if k == 0:
            k = len(inputs) / 2
        
        k = int(k)
        print("k:", k)
            
        
        
        
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
        print("SF総和 :",np.sum(sf))
        
        
        for i in range(sf_x-1):
            if sf[1+i] != sf[sf_y-i-1]:
                print("False")      
        
        
        
        #-----------------------
        #b-r
        #20210627 曲げあり
        
        a = 1/(16*((np.sin(np.pi*5/len(inputs)))**4))
        print("μ/Δx**3 :", a)
        
        n = len(inputs) - 1
        r += 1
        
        
        #-----------------
        #β
        b_up = 0
        b_down = 0
        
        #分子
        for i in range(r):
            b1 = sf[i]
            b2 = -inputs[i]
            t1 = 0
            for j in range(i+1, r):
                b1 += 2*sf[j]
            for j in range(1, r+i):
                b2 += sf[j-i] * inputs[j]
            for j in range(1, r-i):
                b2 -= sf[j+i] * inputs[j]
            
            #T1計算
            t1 = (-1) * inputs[0] * (sf[i-1] - sf[i+1])
            for j in range(2, r+i):
                t1 += sf[j-i] * (inputs[j-1] - 2*inputs[j] + inputs[j+1])
            for j in range(2, r-i):
                t1 -= sf[j+i] * (inputs[j-1] - 2*inputs[j] + inputs[j+1])
            t1 *= (sf[i-1] - sf[i+1])
            
            b_up += b1 * b2 + a*t1
        b_up *= (-1)
        
        #分母
        for i in range(r):
            b3 = sf[i]
            for j in range(i+1, r):
                b3 += 2*sf[j]
            
            #T2計算
            t2 = (sf[i+1] - sf[i-1])**2
            
            b_down += b3**2 + a*t2
        
        b = b_up / b_down
        
        
        #-------------------
        #γ
        
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
            t1 = (-1) * inputs[n] * (sf[i-1] - sf[i+1])
            for j in range(2, r+i):
                t1 += sf[j-i] * (inputs[n-j] - 2*inputs[n-j-1] + inputs[n-j-2])
            for j in range(2, r-i):
                t1 -= sf[j+i] * (inputs[n-j] - 2*inputs[n-j-1] + inputs[n-j-2])
            t1 *= (sf[i-1] - sf[i+1])
            
            r_up += r1 * r2 + a*t1
        r_up *= (-1)
        
        #分母
        for i in range(r):
            r3 = sf[i]
            for j in range(i+1, r):
                r3 += 2*sf[j]
            
            #T2計算
            t2 = (sf[i+1] - sf[i-1])**2
            
            r_down += r3**2 + a*t2
        
        rr = r_up / r_down
        print("β :", b)
        print("γ :", rr)
        
    else:
        print("β,γ is already determined")
        if k == 0:
            k = len(inputs) / 2
        
        k = int(k)
        print("k:", k)
        
        
        
    #--------------------------------------
    #拡張
    
    inputs[0] = b
    inputs[-1] = rr
    
    
    new_in = np.zeros(len(inputs)+k*2)
    b_data = inputs[1:k+1]
    b_data = b_data[::-1]
    r_data = inputs[-k-1:-1]
    r_data = r_data[::-1]
    
    new_in[:k] = 2*b - b_data
    new_in[k+len(inputs):] = 2*rr - r_data
    new_in[k:k+len(inputs)] = inputs
    
    print("end-------------------\n")
    
    
    return(new_in, k)
    
