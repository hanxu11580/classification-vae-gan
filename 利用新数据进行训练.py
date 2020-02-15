import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

# 这里使用keras
tcp_connect_lists = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent']

data = pd.read_csv('./tcp_connect.csv', engine='python')

train_x = np.array(data[tcp_connect_lists])
test = train_x[:10000] # 取一万个

minmax = preprocessing.MinMaxScaler()

train_x = minmax.fit_transform(train_x)
test_x = minmax.fit_transform(test)


# print(test_x.shape)
# print(test_x)

encoding_dim = 3 # 压缩特征值为3个

input_ = Input(shape=(9,)) # 输入为9个特征值

# 定义encoder层
encoded1 = Dense(6, activation='tanh')(input_)
encoder_output = Dense(encoding_dim)(encoded1)


# 定义decoder层

decoded1 = Dense(6, activation='tanh')(encoder_output)
decoded2 = Dense(9, activation='tanh')(decoded1)

autoencoder = Model(
    input=input_,
    output=decoded2
)

# 定义一个encoder模型，用于查看encode的结果
encoder = Model(
    input=input_,
    output=encoder_output
)

adam = Adam(lr=0.001)

autoencoder.compile(
    optimizer=adam,
    loss='mse',
    metrics=['accuracy']
)

autoencoder.fit(train_x,
                train_x,
                epochs=10,
                batch_size=1000,
                shuffle=True,
                )


predict = autoencoder.predict(test_x, batch_size=1000)
loss, accuracy = autoencoder.evaluate(test_x, test_x, batch_size=1000)
print(loss)
print(accuracy)
print(test_x[0])
print(predict[0])











