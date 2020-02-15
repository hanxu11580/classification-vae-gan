import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
import keras as K
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras.losses as K_loss

'''
    分别使用了标准的交叉熵和防止过拟合化的均匀分布的交叉熵和svm_loss
'''

data = pd.read_csv('./kddcup99_10%.csv', engine='python')
columns_list = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent',
                'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_hot_login',
                'is_guest_login',
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                'label']
x_lists = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent',
                'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_hot_login',
                'is_guest_login',
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',]
data_x, data_y = data[x_lists], data['label']


# print(train_x.shape) # (444618, 41)
# print(test_x.shape) # (49403, 41)


minmax = MinMaxScaler()

train_x = minmax.fit_transform(data_x)

one_hot_train_y = np_utils.to_categorical(data_y, num_classes=23)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> softmax-categorical_crossentropy<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# 经过均匀分布的交叉熵
# 这玩意具体意思就是说90%利用正确数据标签[1,0·····0]的交叉熵，10%用[1/23,1/23···1/23]共23个均匀分布的交叉熵
# def my_categorical_crossentropy(y_true, y_pred, e=0.1):
#     loss1 = K_loss.categorical_crossentropy(y_true, y_pred)
#     loss2 = K_loss.categorical_crossentropy(K.backend.ones_like(y_pred)/23, y_pred)
#     return (1-e) * loss1 + e * loss2


def svm_loss(y_true, y_pred, delta=1.):
    yi = K.backend.max(y_true * y_pred, axis=-1)
    yi = K.backend.reshape(yi, (-1, 1))
    one_ = y_pred - yi + delta
    losses = K.backend.sum(K.backend.maximum(one_ - y_true, 0.), axis=-1)
    return K.backend.mean(losses)

def linear_regression_equality(y_true, y_pred):
    diff = K.backend.abs(y_true-y_pred)
    return K.backend.mean(K.backend.cast(diff < 0.001, tf.float32))


model = Sequential()
# 41->64->128->64->32->23
model.add(Dense(64, input_dim=41, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(23, activation='softmax')) # 多分类问题用概率来

adam = Adam(0.001)

# 普通交叉熵， 过份自信在训练样本会出现非常高的准确率，导致在测试期间会出现过拟合的现象
# model.compile(
#     optimizer=adam,
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# 均匀分布过的交叉熵：

# model.compile(
#     optimizer=adam,
#     loss=my_categorical_crossentropy,
#     metrics=["accuracy"]
# )


# svm_loss
model.compile(
    optimizer=adam,
    loss=svm_loss,
    metrics=["accuracy"]
)

# model.fit(train_x, one_hot_train_y, epochs=10, batch_size=1000, shuffle=True)
# 对于均匀交叉熵，这里10轮和20轮会出现不同的准确率，

model.fit(train_x, one_hot_train_y, epochs=10, batch_size=10000, shuffle=True)
pred = model.predict(train_x)
print(pred[0])
print(data_y[0])

# predict = model.predict(test_x)
# print(test_x.values[:1, :])
# print(predict[0])

# loss = model.evaluate(test_x, one_hot_test_y)
# print(loss)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>softmax-categorical_crossentropy<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 定义多分类svm_loss
# def svm_loss(y_true, y_pred, margin=1, train_batch=1000):
    # loss_list = []
    # for i in y_pred:
    #     one_loss = 0
    #     for j in i:
    #         if j != y_true:
    #             one_loss += max(i[j] - i[y_true])
    #         loss_list.append(one_loss)
    # return sum(loss_list)/train_batch









