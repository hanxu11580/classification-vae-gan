import pandas as pd
import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

# 这里使用的数据还是tcp连接数据[, 9]

tcp_connect_lists = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent']

data = pd.read_csv('./tcp_connect.csv', engine='python')



train_data = data[tcp_connect_lists]
train_data['duration'] = train_data['duration'] / 58329.
train_data['protocol_type'] = train_data['protocol_type'] / 3.
train_data['service'] =  train_data['service'] / 70.
train_data['flag'] =  train_data['flag'] / 11.
train_data['src_bytes'] =  train_data['src_bytes'] / 1379963888.
train_data['dst_bytes'] =  train_data['dst_bytes'] / 1309937401.
train_data['wrong_fragment'] = train_data['wrong_fragment'] / 3.
train_data['urgent'] = train_data['urgent'] / 14.
pd.set_option('display.max_columns', None)
train_data = np.array(train_data)


train_x = train_data
test_x = train_x[:10]

batch_size = 1000
start_dim = 9
latent_dim = 18
epochs = 50

# 因为每条数据格式为[, 9],无需再变为1维数据了

x = Input(shape=(9, ))

encode1 = Dense(128, activation='relu')(x)
encode2 = Dense(64, activation='relu')(encode1)
mean = Dense(latent_dim)(encode2) # 均值
log_var = Dense(latent_dim)(encode2) # log方差

# 然后我们需要通过均值和方差从正态分布中采样数据 z
# 因为直接从N(u,方差)采样是无法实现回传优化参数的，因为采样操作是不可导的
# lambda参数为一个执行张量操作的函数，这里先定义一个函数


def samling(args):
    mean, long_var = args
    epsilon = K.random_normal(shape=K.shape(mean)) # 从标准正态分布抽样数据
    return mean + K.exp(log_var/2)* epsilon

z = Lambda(samling, output_shape=(latent_dim,))([mean, log_var])

# z便是我们从原数据x的专属p(z|x)正态分布中采样的数据，用于decode解码

decode_1 = Dense(64, activation='relu')
decode_2 = Dense(128, activation='relu')
decode_mean = Dense(start_dim, activation='sigmoid')
decode1 = decode_1(z)
decode2 = decode_2(decode1)
result = decode_mean(decode2)

# 定义vae的损失函数，损失函数有2:
    # 1、用于重构误差的也就是 result 和 真实 z的
    # 2、由于我们的专属正态分布(也就是后验分布)本质上就是得到的encode出来的数据加上高斯噪音(也就是正态分布噪音数据)
    #    在1、阶段通过不断与真实数据重构误差会将 方差噪音优化掉(方差=0) 这时变成了最为普通的自编码了，所以我们有2种方案
    # ···1、通过计算均值loss和方差loss:就是mean/log_var这俩的绝对值的平方，最终目标这俩还变为0，也就是均值为0，log(方差)=0，方差为1
    #        这样也就使得最好情况优化为正态分布了N(0, 1),但是问题在于无法知道这2个loss比例权重为多少，所以采用第二种
    # ···2、通过计算N(0,1)和N(mean, 方差)这2个的分布相似度就是KL散度，从而使得mean和方差构成的正态分布趋近于N(0, 1)公式在本子上

# 这次采用第二种方法


mean_loss = K.sum(K.binary_crossentropy(x, result), axis=-1)
kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
vae_loss = K.mean(mean_loss + kl_loss)


vae = Model(x, result)

vae.add_loss(vae_loss)
adam = Adam(0.0001)
vae.compile(loss=None,optimizer=adam)

vae.fit(train_x,
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size)

pred = vae.predict(test_x)

# print(pred[0])
# print(test_x[0])

true_data = test_x[0].flatten()
fake_data = pred[0].flatten()

true_data[0] = true_data[0] * 58329.
true_data[1] = true_data[1] * 3.
true_data[2] = true_data[2] * 70.
true_data[3] = true_data[3] * 11.
true_data[4] = true_data[4] * 1379963888.
true_data[5] = true_data[5] * 1309937401.
true_data[7] = true_data[7] * 3.
true_data[8] = true_data[8] * 14.

fake_data[0] = fake_data[0] * 58329.
fake_data[1] = fake_data[1] * 3.
fake_data[2] = fake_data[2] * 70.
fake_data[3] = fake_data[3] * 11.
fake_data[4] = fake_data[4] * 1379963888.
fake_data[5] = fake_data[5] * 1309937401.
fake_data[7] = fake_data[7] * 3.
fake_data[8] = fake_data[8] * 14.

print(true_data)
print(fake_data)











