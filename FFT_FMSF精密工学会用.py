# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:03:03 2021

@author: user01
"""




import numpy as np
import matplotlib.pyplot as plt
import cmath




#------------------------------------------
#フーリエ変換

def dft(f):
    n = len(f)
    A = np.arange(n)
    M = cmath.e**(-1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * cmath.pi / n)
    return np.sum(f * M, axis=1)




#--------------------------------------------
#逆フーリエ変換

def idft(f):
    n = len(f)
    A = np.arange(n)
    M = cmath.e**(1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * cmath.pi / n)
    return np.sum(f * M, axis=1) / n





#--------------------------------------------
#パラメーター

m = 21
m3 = m + (m-1)*2
z_ = 200
z = z_ - m3
data_size = 201
noise_data = np.loadtxt("inputs_data.csv")
inputs = noise_data[:data_size]

#データを奇数個にするためにとりあえず
a = np.arange(201)
inputs = 60*np.sin(2*np.pi/201*a) #+ np.random.rand(len(inputs))*14 -7

len_x = len(inputs)
len_z = z_




#--------------------------------------------
#2次B-spline

b_spline1 = np.zeros(z_)
b_spline1[:m//2+1] = 1/m
b_spline1[z_-m//2:] = 1/m

ft_b_spline1 = dft(b_spline1)
ft_b_spline2 = ft_b_spline1 **2
ft_b_spline3 = ft_b_spline1 * ft_b_spline2






#--------------------------------------------
#SF

sf_y = len(inputs)
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

ft_sf = dft(sf)




#--------------------------------------------
#離散化→DFT

z_data = np.zeros((z_,len(inputs)))
max_data = max(inputs)
min_data = min(inputs)

if min_data < 0:
    inputs_after = inputs - min_data
    plt.plot(inputs_after)
    plt.show()
else:
    inputs_after = inputs

if max_data > 0:
    if min_data > 0:
        min_data = 0
elif max_data < 0:
    max_data = -min_data
    min_data = 0
else:
    print("error----")
    
    
    

step = (max_data - min_data) / z
step_num = inputs_after / step
step_floor = np.floor(step_num)
step_middle = step_floor + 0.5
dif = step_middle - step_num


for i in range(len(inputs)):
    # height = (int)(z - step_floor[i] + m3//2)
    height = (int)(step_floor[i] + m3//2)
    if dif[i] > 0:
        z_data[height ,i] = step_num[i] - step_floor[i]
        z_data[height+1, i] = 1 - z_data[height ,i]
    else:
        z_data[height, i] = step_num[i] - step_floor[i]
        z_data[height-1, i] = 1 - z_data[height ,i]

#np.savetxt("risanka.csv", z_data, delimiter=",")
ft_z_data = np.fft.ifftshift(z_data)
#np.savetxt("risankaaaaaaaaaaa.csv", ft_z_data, delimiter=",")
ft_z_data = np.fft.fft2(ft_z_data)

# a = np.zeros(z_data.shape, dtype=np.complex)
# b = np.zeros(z_data.shape, dtype=np.complex)
# for i in range(len_x):
#     a[:,i] = dft(ft_z_data[:,i])

# for i in range(len_z):
#     b[i,:] = dft(a[i,:])
# ft_z_data  = b








#------------------------------------------------
#DFT-FMSF計算→IDFT

ft_fmsf = np.zeros(z_data.shape, dtype=np.complex)

for i in range(len_z):
    ft_fmsf[i,:] = ft_z_data[i,:] * ft_sf

for i in range(len_x):
    ft_fmsf[:,i] = ft_fmsf[:,i] * ft_b_spline3

fmsf_risan = np.fft.ifft2(ft_fmsf)

# a = np.zeros(z_data.shape, dtype=np.complex)
# b = np.zeros(z_data.shape, dtype=np.complex)

# for i in range(len_x):
#     a[:,i] = idft(ft_fmsf[:,i])
    
# for i in range(len_z):
#     b[i,:] = idft(a[i,:])

# fmsf_risan = b
fmsf_risan = np.fft.fftshift(fmsf_risan)    
    
np.savetxt("riririririri.csv", fmsf_risan, delimiter=",")







#-----------------------------------------------
#Fast M estimation Method

max_num = np.zeros(len_x)

for i in range(len_x):
    max_num[i] = fmsf_risan[:,i].argmax()

zz = (max_num - m3//2)*step + min_data

plt.plot(inputs)
plt.plot(zz)
plt.show()
