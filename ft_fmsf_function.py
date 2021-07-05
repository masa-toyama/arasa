# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:32:39 2021

@author: user01
"""



import numpy as np
import matplotlib.pyplot as plt
from b_r import b_r



def ft_fmsf(inputs_, b=0, rr=0):
    
    
    
    #--------------------------------------------
    #パラメーター
    
    m =21
    m3 = m + (m-1)*2
    z = 100
    z_ = z + m3
    
    
    
    
    
    #---------------------------------------------
    #inputsdata
    
    # nyuryoku = np.loadtxt("inputs/step.csv", delimiter=",")
    # inputs_ = nyuryoku[:,0]
    # fmsf = nyuryoku[:,1]
    
    
    len_z = z_
    
    judge = False
    if len(inputs_) % 2 == 0:
        judge = True
    
    
    
    
    
    #--------------------------------------------
    #2次B-spline
    
    b_spline1 = np.zeros(z_)
    b_spline1[:m//2+1] = 1/m
    b_spline1[z_-m//2:] = 1/m
    
    ft_b_spline1 = np.fft.fft(b_spline1)
    ft_b_spline2 = ft_b_spline1 **2
    ft_b_spline3 = ft_b_spline1 * ft_b_spline2
    
    
    
    
    
    #--------------------------------------------
    #b-r
    
    inputs, k = b_r(inputs_, k=0, b=b, rr=rr)
    len_x = len(inputs)
    
    
    
    
    
    #--------------------------------------------
    #SF
    
    # sf_y = len(inputs)
    # if judge:
    #     sf_y += 1
    # sf_x = (int)((sf_y-1)/2 + 1)
    # ramuda = (sf_y-1)/5
    # sf = np.zeros(sf_y)
    # omomi = 0
    
    # #フーリエ用SF
    # for i in range(sf_x):
        
    #     if i == (sf_x-1) and judge:
    #         sf[i] = np.pi / ramuda * np.sin((np.sqrt(2) * np.pi * i) / ramuda + np.pi / 4) * np.exp((-np.sqrt(2) * np.pi * i) / ramuda) / 2
    #     else:
    #         sf[i] = np.pi / ramuda * np.sin((np.sqrt(2) * np.pi * i) / ramuda + np.pi / 4) * np.exp((-np.sqrt(2) * np.pi * i) / ramuda)
        
    #     if i != 0:
    #         sf[sf_y-i] = sf[i]
    #         omomi += sf[i]*2
    #     else:
    #         omomi += sf[0]
        
    # sf /= omomi
    # print("SF総和:",np.sum(sf))
    
    
    # for i in range(sf_x-1):
    #     if sf[1+i] != sf[sf_y-i-1]:
    #         print("False")
    # if judge:        
    #     sf = np.delete(sf, sf_x)       
    
    # ft_sf = np.fft.fft(sf)
    
    # # plt.plot(sf)
    # # plt.show()
    
    #3次スプライン
    l = len(inputs)
    ft_sf = np.zeros(l)
    for i in range(l):
        ft_sf[i] = (1 + (np.sin(np.pi*i/l) / np.sin(np.pi*5/l))**(2*2))**(-1)
    
    plt.plot(ft_sf, label="ft_sf")
    plt.legend()
    plt.show()
    
    
    
    
    #--------------------------------------------
    #離散化→DFT
    
    z_data = np.zeros((z_,len(inputs)))
    max_data = max(inputs)
    min_data = min(inputs)
    step = (max_data - min_data) / z
    inputs_after = inputs - min_data #+ step/2
    
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
    
    ft_z_data = np.fft.ifftshift(z_data)
    ft_z_data = np.fft.fft2(ft_z_data)
    
    
    
    
    
    
    #------------------------------------------------
    #DFT-FMSF計算→IDFT
    
    ft_fmsf = np.zeros(z_data.shape, dtype=np.complex)
    
    for i in range(len_z):
        ft_fmsf[i,:] = ft_z_data[i,:] * ft_sf
    
    for i in range(len_x):
        ft_fmsf[:,i] = ft_fmsf[:,i] * ft_b_spline3
    
    
    fmsf_risan1 = np.fft.ifft2(ft_fmsf)
    fmsf_risan = np.zeros(len_x, dtype=np.float64)
    fmsf_risan = np.fft.fftshift(fmsf_risan1)
    
    
    
    
    
    
    #-----------------------------------------------
    #Fast M estimation Method
    
    max_num = np.zeros(len_x, dtype=np.int)
    fmsf_p = np.zeros(len_x, dtype=np.float64)
    
    for i in range(len_x):
        max_num[i] = int(fmsf_risan[:,i].argmax())
        
        #最高点計算
        fmsf_p[i] = ((fmsf_risan[max_num[i]-1,i] - fmsf_risan[max_num[i],i])
             /(fmsf_risan[max_num[i]-1,i] - 2*fmsf_risan[max_num[i],i] + fmsf_risan[max_num[i]+1,i]))# - 0.5)# * step
    
    zz = ((max_num - m3//2) + fmsf_p)*step + min_data #- step/2
    
    
    
    
    
    #----------------------------------------------
    #β-γの拡張文を元に戻す
    
    
    zzz = zz[k:k+len(inputs_)]
    inputs = inputs[k:k+len(inputs_)]
    
    
    
    
    # plt.plot(inputs)
    # plt.plot(zzz, label="ft")
    # #plt.plot(fmsf, label="fmsf")
    # plt.legend()
    # plt.show()
    
    
    # gosa = (zzz - fmsf) / (max_data - min_data)
    
    # plt.plot(gosa[30:170], label="pv")
    # plt.legend()
    # plt.show()
    
    
    
    return zzz
    
    
