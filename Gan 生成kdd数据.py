import pandas as pd
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam


# 这里使用的数据还是tcp连接数据[, 9]

tcp_connect_lists = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent']

data = pd.read_csv('./tcp_connect.csv', engine='python')

train_data = data[tcp_connect_lists].astype(np.float32)

# --- 这里先归一化一下
# 这里是利用物理意义来归一(因为取的是10%数据)， sklearn中的minmax归一化化是利用整体数据
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




class Gan(object):
    def __init__(self):
        # 这里构建训练生成器的模型
        self.start_dim = 100 #拿[1,100]高斯噪音生成数据
        self.g_data_dim = 9 #生成数据size

        optimizer = Adam(0.0001)
        # optimizer = RMSprop(0.0001)

        self.discriminator = self.build_d()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.generator = self.build_g()


        x = Input(shape=(self.start_dim, ))
        sim_data = self.generator(x)

        # 这里关闭判别器的权重更新(这里是为了训练生成器，如果开启，将会使判别器误认为生成的数据为真)
        self.discriminator.trainable = False

        score = self.discriminator(sim_data)
        # 这里生成器对应的是 noise->判断器出来的结果(是真实值的概率)
        # 这里会根据loss = -1*log(true_score)来更新生成器f(x)的权重和偏置
        self.combind = Model(x, score)
        self.combind.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
        )



    def build_g(self):
        # 返回生成器
        noise_input = Input(shape=(self.start_dim,))

        g_1 = Dense(150, activation='relu')(noise_input)
        g_2 = Dense(200, activation='relu')(g_1)
        g_3 = Dense(300, activation='relu')(g_2)
        g_data = Dense(self.g_data_dim)(g_3)

        return Model(noise_input, g_data)


    def build_d(self):
        # 返回判别器
        data_input = Input(shape=(self.g_data_dim, ))

        d_1 = Dense(150, activation='relu')(data_input)
        d_2 = Dense(100, activation='relu')(d_1)
        true_score = Dense(1, activation='sigmoid')(d_2)

        return Model(data_input, true_score)


    def train(self, epochs, batch_size):


        true_data = train_data #真实数据

        valid = np.ones((batch_size, 1)) # 真实数据的label
        fake = np.zeros((batch_size, 1)) # 生成数据的label

        for epoch in range(epochs):

            # 每轮随机生成高斯噪音，随机抽取真实数据
            noise = np.random.normal(0, 1, (batch_size, self.start_dim))
            index = np.random.randint(0, true_data.shape[0], batch_size)
            batch_true_data = true_data[index]

            fake_data = self.generator.predict(noise)

            # 在这个过程会训练判别器
            d_loss_real = self.discriminator.train_on_batch(batch_true_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 这里我们生成的数据的label为1
            # 正常情况下回传loss时，会优化判别器，让判别器认为虚假数据为真实数据
            # 但我们将判别器关闭了，这就导致回传时并不会优化判别器的参数，从而导致优化生成器的参数，让生成器的生成的数据更加接近真实数据
            g_loss = self.combind.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))


if __name__ == "__main__":
    gan = Gan()
    gan.train(epochs=500, batch_size=1000)

    data = gan.generator.predict(np.random.normal(0, 1, (1, 100)))
    print(data)
    g_data = data.flatten()

    g_data[0] = g_data[0] * 58329.
    g_data[1] = g_data[1] * 3.
    g_data[2] = g_data[2] * 70.
    g_data[3] = g_data[3] * 11.
    g_data[4] = g_data[4] * 1379963888.
    g_data[5] = g_data[5] * 1309937401.
    g_data[7] = g_data[7] * 3.
    g_data[8] = g_data[8] * 14.

    print(g_data)
















