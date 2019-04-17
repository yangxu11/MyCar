import os
import numpy as np
import matplotlib.image as mpimg
import random

from skimage.io import imread
from skimage.transform import resize

import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import load_model, Model, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD


# 全局变量
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

class MY_Generator(Sequence):  # generator 继承自 Sequence

    def __init__(self, image_filenames, labels, batch_size):
        # image_filenames - 图片路径
        # labels - 图片对应的类别标签
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        # 计算 generator要生成的 batches 数，
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        # idx - 给定的 batch 数，以构建 batch 数据 [images_batch, GT]
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (120, 160))
            for file_name in batch_x]), np.array(batch_y)



# 如果有多个 CPU 核，可以设置 use_multiprocessing=True,即可在 CPU 上并行运行.
# 设置 workers=CPU 核数，用于 batch 数据生成.


def process_img(img_path, key):
    print(img_path, key)

    # Use PIL to convert image file to numpy array
    # image = Image.open(img_path)
    # image_array = np.array(image)
    # image_array = np.expand_dims(image_array,axis = 0)

    # Use matplotlib to convert image file to numpy array
    # image_array = mpimg.imread(img_path)
    # image_array = np.expand_dims(image_array, axis=0)
    # print(image_array.shape)

    if key == 2:
        label_array = [0., 0., 1., 0., 0.]
    elif key == 3:
        label_array = [0., 0., 0., 1., 0.]
    elif key == 0:
        label_array = [1., 0., 0., 0., 0.]
    elif key == 1:
        label_array = [0., 1., 0., 0., 0.]
    elif key == 4:
        label_array = [0., 0., 0., 0., 1.]

    return label_array




# step2 建立模型
def build_model(keep_prob):
    print("开始编译模型")
    model = Sequential()
    model.add(Lambda(lambda x: (x / 102.83 - 1), input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(keep_prob))  # Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接
    model.add(Flatten())
    # model.add(Dense(500, activation='elu'))
    model.add(Dense(250, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    model.add(Dense(5))
    model.summary()

    return model


# step3 训练模型
def train_model(model, learning_rate, nb_epoch, samples_per_epoch,
                batch_size, training_filenames, GT_training):
    # 值保存最好的模型存下来
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    # EarlyStopping patience：当earlystop被激活（如发现loss相比上一个epoch训练没有下降），
    # 则经过patience个epoch后停止训练。
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
    early_stop = EarlyStopping(monitor='loss', min_delta=.0005, patience=10,
                               verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=20, write_graph=True, write_grads=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    # 编译神经网络模型，loss损失函数，optimizer优化器， metrics列表，包含评估模型在训练和测试时网络性能的指标
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    # 训练神经网络模型，batch_size梯度下降时每个batch包含的样本数，epochs训练多少轮结束，
    # verbose是否显示日志信息，validation_data用来验证的数据集
    # generator 的使用
    my_training_batch_generator = My_Generator(training_filenames,
                                               GT_training,
                                               batch_size)
    # my_validation_batch_generator = My_Generator(validation_filenames,
    #                                              GT_validation,
    #                                              batch_size)

    model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch=(samples_per_epoch // batch_size),
                        epochs=nb_epoch,
                        verbose=2,
                        callbacks=[tensorboard, checkpoint, early_stop],
                        max_queue_size=1)
                        # validation_data=my_validation_batch_generator,
                        # validation_steps=(num_validation_samples // batch_size),
                        # use_multiprocessing=True,
                        # workers=16,
                        # )

def main():
    # 打印出超参数

    print('-'*30)
    print('parameters')
    print('-'*30)

    path = "training_data"
    files = os.listdir(path)

    random.shuffle(files)

    keep_prob = 0.5
    learning_rate = 0.0001
    nb_epoch = 100
    samples_per_epoch = len(files)
    batch_size = 30

    print('keep_prob = ', keep_prob)
    print('learning_rate = ', learning_rate)
    print('nb_epoch = ', nb_epoch)
    print('samples_per_epoch = ', samples_per_epoch)
    print('batch_size = ', batch_size)
    print('-' * 30)



    train_labels = np.zeros((1, 5), 'float')
    print("生成标签矩阵。。。")
    for file in files:
        if not os.path.isdir(file) and file[len(file) - 3:len(file)] == 'jpg':
            try:
                key = int(file[0])
                label_array = process_img(path + "/" + file, key)
                train_labels = np.vstack((train_labels, label_array))
            except:
                print('prcess error')

    # 编译模型
    model = build_model(keep_prob)
    # 在数据集上训练模型，保存成model.h5
    train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size,files,train_labels)
    print("模型训练完毕")