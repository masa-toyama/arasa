# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:32:40 2021

@author: user01
"""


import numpy as np
import matplotlib.pyplot as plt
from fmsf_function import fmsf
from ft_fmsf_function import ft_fmsf


nyuryoku = np.loadtxt("inputs/takahashi.csv", delimiter=",")
inputs = nyuryoku[:,0]
sf = nyuryoku[:,1]




# fmsf = fmsf(inputs)
# ft_fmsf = ft_fmsf(inputs)


#β-γの値を設定したい場合
beta = sf[0]
ganma = sf[-1]
fmsf = fmsf(inputs, b=beta, rr=ganma)
ft_fmsf = ft_fmsf(inputs,b=beta, rr=ganma)

plt.plot(inputs)
plt.plot(fmsf, label="real")
plt.plot(ft_fmsf, label="ft")
plt.plot(sf, label="SF")
plt.legend()
plt.show()


max_data = max(inputs)
min_data = min(inputs)



gosa_real = (sf - fmsf) / (max_data - min_data)
gosa_imag = (sf - ft_fmsf) / (max_data - min_data)
gosa = (fmsf - ft_fmsf)/(max_data - min_data)

plt.plot(gosa_real[50:150], label="sf-fmsf")
plt.plot(gosa_imag[50:150], label="sf-ft_fmsf")
plt.plot(gosa[50:150], label="fmsf-ft_fmsf")
plt.legend()
plt.show()

plt.plot(gosa_real, label="sf-fmsf")
plt.plot(gosa_imag, label="sf-ft_fmsf")
plt.plot(gosa, label="fmsf-ft_fmsf")
plt.legend()
plt.show()

syutu = np.zeros((200,7))
syutu[:,0] = inputs
syutu[:,1] = sf
syutu[:,2] = fmsf
syutu[:,3] = ft_fmsf
syutu[:,4] = gosa_real
syutu[:,5] = gosa_imag
syutu[:,6] = gosa
np.savetxt("seimitukougakkai.csv", syutu, delimiter=",")