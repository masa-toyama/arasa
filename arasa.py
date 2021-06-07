"""
表面粗さ用
CNNとGANを用いたハイブリット学習

"""


import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv1D, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam





#===============================================================
#create dataset
dataset = np.zeros((900,400))
noise_data = np.zeros((900,400))
step = 0

#slop
for i in range(300):
    
    #スロープの開始位置
    #rand1 = np.random.randint(100,250)
    rand1 = 125
    #スロープの傾き
    rand2 = np.random.randint(-4,4) + np.random.rand()
    #データの高さ調整
    rand3 = np.random.randint(-100,100)
    #noise
    rand4 = np.random.randn(400)/10
    
    dataset[step,:rand1] = 0
    num = np.arange(150)
    dataset[step,rand1:rand1+150] = rand2 * num
    dataset[step,rand1+150:] = dataset[step,rand1+149]
    
    ave = dataset[step,399] / 2
    dataset[step,:] += -ave + rand3
    noise_data[step,:] = dataset[step,:] + rand4
    
    step += 1


#step
for i in range(300):
    
    #stepの開始位置
    #rand1 = np.random.randint(100,250)
    rand1 = 125
    #stepの高さ
    rand2 = np.random.randint(0,50) + np.random.rand()
    #データの高さ調整
    rand3 = np.random.randint(-25,25)
    #noise
    rand4 = np.random.randn(400)/10
    
    dataset[step,:rand1] = 0
    num = np.ones(150)
    dataset[step,rand1:rand1+150] = rand2 * num
    
    #stepの高さ
    rand5 = np.random.randint(-50,0) + np.random.rand()
    num = np.ones(125)
    dataset[step,rand1+150:] = (rand2+rand5) * num
    noise_data[step,:] = dataset[step,:] + rand4
    
    step += 1
    


#spike
for i in range(300):
    
    #stepの開始位置
    #rand1 = np.random.randint(100,250)
    rand1 = 125
    #データの高さ
    rand2 = np.random.randint(-30,30)
    #spikeの高さ調整
    rand3 = np.random.randint(-100,100,3)
    #noise
    rand4 = np.random.randn(400)/10
    
    if np.random.rand() > 0.5:
        noise_data[step,100] = rand3[0]
        noise_data[step,200] = rand3[1]
        noise_data[step,300] = rand3[2]
    else:
        noise_data[step,125] = rand3[0]
        noise_data[step,275] = rand3[1]
    
    if np.random.rand() > 0.3:
        dataset[step,:] += rand2
        noise_data[step,:] += rand2
    
    noise_data[step,:] += rand4
    
    step += 1
    





#========================================================
#crate model

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
    dense3 = Dense(20, activation="sigmoid")(dense2)
    return Model(inputs, dense3)


optimizer = Adam(0.0001, 0.5)
optimizer1 = Adam(0.0005)


#create discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])
#discriminator.summary()


#create generator
generator = build_generator()
#generator.summary()
generator.compile(optimizer = optimizer1, loss = "mae")


#create combinemodel
z = Input(shape=(400,1))
fake = generator(z)
discriminator.trainable = False
valid = discriminator(fake)

combine = Model(z,valid)
#combine.summary()
combine.compile(loss='binary_crossentropy', optimizer=optimizer)







#===========================================================
#train

epochs = 100
true = np.ones((1,20))
fake = np.zeros((1,20))
k = 0

for epoch in range(epochs):
    
    if epoch == 0:
        print("train start!")
    
    generator_loss = 0
    discriminator_loss = 0
    for i in range(900):
        
        rand_num = np.arange(900)
        np.random.shuffle(rand_num)
        m = rand_num[i]
        
        train = noise_data[m,:]#.reashape(1,400,1)
        target = dataset[m,40:360]#.reshape(1,320,1)
        train = train.reshape(1,400,1)
        target = target.reshape(1,320,1)
        
        fake_img = generator.predict(train)
        d_loss_fake = discriminator.train_on_batch(fake_img, fake)
        d_loss_real = discriminator.train_on_batch(target, true)
        g_loss = generator.train_on_batch(train, target)
        c_loss = combine.train_on_batch(train, true)
        
        generator_loss += 0.5 * (g_loss + c_loss)
        discriminator_loss += 0.5 * (d_loss_real[0] + d_loss_fake[0])
        
    generator_loss /= 900
    discriminator_loss /= 900
    
    if(epoch % 10 == 0):
        print("epoch:",epoch+1)
        print("g_loss:", generator_loss)
        print("d_loss:", discriminator_loss)
    
    
    if (epoch%1) == 0:
        
        fig = plt.figure()
        ax_1 = fig.add_subplot(221)
        ax_2 = fig.add_subplot(222)
        
        rand = np.random.randint(0,900)
        train = noise_data[rand,:]#.reashape(1,400,1)
        train = train.reshape(1,400,1)
        fake_img = generator.predict(train)
        fake_plot = fake_img.reshape(320)
        true_plot = dataset[rand,40:360]
        ax_1.plot(range(len(fake_plot)), fake_plot, label="fake")
        ax_1.plot(range(len(true_plot)), true_plot, label="true")
        plt.legend()
        
        
        weights1 = generator.layers[1].get_weights()[0]
        wei1 = weights1.reshape([81])
        ax_2.plot(range(len(wei1)), wei1, label="generator")
        plt.legend()
        
        plt.savefig("0520\img_{}.png".format(k))
        k+=1
        #plt.show()

#モデルの保存
generator.save('C:/Users/user01/Desktop/AI/cnn/my_model.h5')
