# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:32:25 2021

@author: user01
"""

#実空間FMSF


import numpy as np
import matplotlib.pyplot as plt
from b_r import b_r




def fmsf(inputs_, b=0, rr=0):
    
    
    #--------------------------------------------
    #パラメーター
    
    m = 21
    m3 = m + (m-1)*2
    z = 50
    z_ = z + m3
    filter_size = 5 #これだと5λc
    
    
    
    
    
    #---------------------------------------------
    #inputsdata
    
    # nyuryoku = np.loadtxt("inputs/step.csv", delimiter=",")
    # inputs_ = nyuryoku[:,0]
    # fmsf = nyuryoku[:,1]
    
    
    len_z = z_
    
    
    
    
    
    
    #--------------------------------------------
    #2次B-spline
    
    b_spline1 = np.zeros(z_)
    b_spline1[:m//2+1] = 1/m
    b_spline1[z_-m//2:] = 1/m
    
    ft_b_spline1 = np.fft.fft(b_spline1)
    ft_b_spline2 = ft_b_spline1 **2
    ft_b_spline3 = ft_b_spline1 * ft_b_spline2
    
    bs1 = np.fft.ifft(ft_b_spline3)
    bs = np.zeros(m3)
    bs[:m3//2+1] = bs1[:m3//2+1]
    bs[m3//2+1:] = bs1[z_-m3//2:]
    bs = np.fft.fftshift(bs)
    
    # plt.plot(bs, label="s_spline")
    # plt.legend()
    # plt.show()
    
    
    
    
    #--------------------------------------------
    #b-r
    
    inputs, kk = b_r(inputs_, k=len(inputs_)/5*filter_size/2, b=b, rr=rr)
    len_x = len(inputs)
    
    plt.plot(inputs, label="β-γ")
    plt.legend()
    plt.show()
    
    
    
    
    
    #--------------------------------------------
    #SF
    
    sf_y = int(len(inputs_) / 5 * filter_size) + 1 #5λc
    sf_x = (int)((sf_y-1)/2 + 1)
    ramuda = (sf_y-1)/5
    sf = np.zeros(sf_y)
    omomi = 0
    
    #フーリエ用SF
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
           
    
    sf = np.fft.fftshift(sf)
    # plt.plot(sf, label="SF")
    # plt.legend()
    # plt.show()
    
    
    
    
    
    
    #--------------------------------------------
    #離散化
    
    z_data = np.zeros((z_,len(inputs)))
    max_data = max(inputs)
    min_data = min(inputs)
    step = (max_data - min_data) / z
    inputs_after = inputs - min_data + step/2
    
    step_num = inputs_after / step
    step_floor = np.floor(step_num)
    step_middle = step_floor + 0.5
    dif = step_num - step_middle
    
    for i in range(len(inputs)):
        # height = (int)(z - step_floor[i] + m3//2)
        height = (int)(step_floor[i] + m3//2)
        if dif[i] > 0:
            z_data[height ,i] = step_floor[i] + 1.5 - step_num[i]
            z_data[height+1, i] = 1 - z_data[height ,i]
        else:
            z_data[height, i] = step_num[i] - step_floor[i] + 0.5
            z_data[height-1, i] = 1 - z_data[height ,i]
    
    
    
    
    
    
    
    
    #------------------------------------------------
    #SF,B-Spline 畳み込み
    
    x_, y_ = z_data.shape
    fmsf_sf = np.zeros((x_, y_-len(sf)+1))
    for k in range(len(fmsf_sf[:,0])):
        for i in range(len(fmsf_sf[0,:])):
            for j in range(len(sf)):
                fmsf_sf[k,i] += z_data[k,i+j] * sf[j] 
            
            
    x_, y_ = fmsf_sf.shape
    fmsf_bs = np.zeros((x_-len(bs), y_))
    for i in range(len(fmsf_bs[0,:])):
        for k in range(len(fmsf_bs[:,0])):
            for j in range(len(bs)):
                fmsf_bs[k,i] += fmsf_sf[k+j,i] * bs[j]
    
    
    fmsf_risan = fmsf_bs
    
    
    
    
    
    #-----------------------------------------------
    #Fast M estimation Method
    
    max_num = np.zeros(len(inputs_), dtype=np.int)
    fmsf_p = np.zeros(len(inputs_), dtype=np.float64)
    
    for i in range(len(fmsf_risan[0,:])):
        max_num[i] = int(fmsf_risan[:,i].argmax())
        
        #最高点計算
        fmsf_p[i] = ((fmsf_risan[max_num[i]-1,i] - fmsf_risan[max_num[i],i])
             /(fmsf_risan[max_num[i]-1,i] - 2*fmsf_risan[max_num[i],i] + fmsf_risan[max_num[i]+1,i]))# - 0.5)# * step
    
    zz = ((max_num) + fmsf_p)*step + min_data - step/2
    
    
    
    
    
    #----------------------------------------------
    #β-γの拡張文を元に戻す
    
    inputs = inputs[kk:kk+len(inputs_)]
    
    
    
    # plt.plot(inputs)
    # plt.plot(zz, label="real")
    # #plt.plot(fmsf, label="fmsf")
    # plt.legend()
    # plt.show()
    
    
    # gosa = (zz - fmsf)# / (max_data - min_data)
    
    # plt.plot(gosa[30:170], label="pv")
    # plt.legend()
    # plt.show()
    
    
    return zz
    
