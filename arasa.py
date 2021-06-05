"""
1次元CNN

"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv1D, MaxPooling1D, Average, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model
from keras.constraints import non_neg
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K


#inputs
def sin(n, line):
    return np.sin(line * n / 1200)

def cos(n, line):
    return np.cos(line * n / 1200)

data = np.arange(8000)
normal_data = 2*sin(1, data) + 0.7*sin(3, data) + 0.3*cos(2, data) + cos(7, data)
noise = np.random.rand(len(normal_data)) * 0.5 - 0.25
noise_data = normal_data + noise

# plt.plot(range(len(normal_data)), noise_data, label="noise")
# plt.legend()
# plt.show()


#testデータ
#noise_data_test = np.loadtxt("C:/Users/user01/Desktop/AI/cnn/noise_data.csv")
# plt.plot(range(len(noise_data)), noise_data, label="noise")
# plt.legend()
# plt.show()


data_len = int(len(noise_data) / 100)
train = np.zeros((data_len,400))
true = np.zeros((data_len,320))
j = 0

for i in range(data_len-3):
    train[i,:] = noise_data[j:j+400]
    true[i,:] = normal_data[j+40:j+360]
    j += 30

train = train.reshape([80,400,1])
true = true.reshape([80,320,1])


#conv
def build_generator():
    inputs = Input(shape=(400,1))
    conv1 = Conv1D(1, 81)(inputs)
    return Model(inputs, conv1)


#discriminater
def build_discriminator():
    inputs = Input(shape=(320,1))
    dis1 = Conv1D(4, 15)(inputs)
    dis2 = Conv1D(8, 15)(dis1)
    dis3 = Conv1D(4, 15)(dis2)
    dis4 = Conv1D(1, 15)(dis3)
    flat = Flatten()(dis4)
    dense = Dense(240)(flat)
    dense2 = Dense(60)(dense)
    dense3 = Dense(1, activation="sigmoid")(dense2)
    return Model(inputs, dense3)



def masa(y_true, y_pred):
    squared_difference = K.abs(y_true - y_pred)
    
    return K.sum(squared_difference, axis=-1)


optimizer = Adam(0.0001, 0.5)
optimizer1 = Adam(0.0005)


#create discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])
discriminator.summary()


#create generator
generator = build_generator()
generator.summary()
generator.compile(optimizer = optimizer1, loss = "mae")


#create combinemodel
z = Input(shape=(400,1))
fake = generator(z)
discriminator.trainable = False
valid = discriminator(fake)

combine = Model(z,valid)
combine.summary()
combine.compile(loss='binary_crossentropy', optimizer=optimizer)




epochs = 3000
real = np.ones((80,1))
fake = np.zeros((80,1))
i = 0

for epoch in range(epochs):
    
    
    
    g_loss = generator.train_on_batch(train, true)
        
    fake_img = generator.predict(train)
    
    d_loss_fake = discriminator.train_on_batch(fake_img, fake)
    d_loss_real = discriminator.train_on_batch(true, real)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    c_loss = combine.train_on_batch(train, real)
            
    
    
    print("%d, gan_loss:%.2f" % (epoch, 100 * d_loss[1]))
    print("D_loss:: real:%f, fake:%f" %(d_loss_real[0], d_loss_fake[0]))
    print("G_loss:: mae :%f, gan :%f" %(g_loss, c_loss))
    # print("%d [D loss: %f, acc.: %.2f%%] [C loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], c_loss))
    # print("[C loss: %f]"%g_loss)
    
    if (epoch%10) == 0:
        
        fig = plt.figure()
        ax_1 = fig.add_subplot(221)
        ax_2 = fig.add_subplot(222)
        #ax_3 = fig.add_subplot(223)
        
        fake_img = generator.predict(train)
        fake_plot = fake_img[2,:]
        true_plot = true[2,:]
        ax_1.plot(range(len(fake_plot)), fake_plot, label="fake")
        ax_1.plot(range(len(true_plot)), true_plot, label="true")
        plt.legend()
        
        
        weights1 = generator.layers[1].get_weights()[0]
        wei1 = weights1.reshape([81])
        # weights2 = discriminator.layers[1].get_weights()[0]
        # wei2 = weights2.reshape([81])
        ax_2.plot(range(len(wei1)), wei1, label="generator")
        # ax_3.plot(range(len(wei2)), wei2, label="discriminator")
        plt.legend()
        
        plt.savefig("C:/Users/user01/Desktop/AI/cnn/0604/img_{}.png".format(i))
        i+=1
        #plt.show()

#モデルの保存
generator.save('C:/Users/user01/Desktop/AI/cnn/my_model.h5')