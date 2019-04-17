import os
import numpy as np
import matplotlib.image as mpimg
import random
import cv2

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
PATH = 'training_data'

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
                batch_size, train_image, train_label):
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
    # my_validation_batch_generator = My_Generator(validation_filenames,
    #                                              GT_validation,
    #                                              batch_size)

    model.fit_generator(generator=get_train_batch(train_image,train_label,160,120),
                        steps_per_epoch=(samples_per_epoch // batch_size),
                        epochs=nb_epoch,
                        verbose=2,
                        callbacks=[tensorboard, checkpoint, early_stop],
                        max_queue_size=1,
                        shuffle=True)
                        # validation_data=my_validation_batch_generator,
                        # validation_steps=(num_validation_samples // batch_size),
                        # use_multiprocessing=True,
                        # workers=16,
                        # )


# 读取图片函数

def get_im_cv2(paths,img_cols, img_rows):
    '''
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行  height高  120
        img_cols:图片列  width 宽  160
        color_type:图片颜色通道
    返回:
        imgs: 图片数组
    '''
    # Load as grayscale
    # imgs = []
    train_imgs = np.zeros([1, 120, 160, 3])
    for path in paths:
        image_array = mpimg.imread(PATH+"/"+path)
        image_array = np.expand_dims(image_array, axis=0)
        train_imgs = np.vstack((train_imgs, image_array))
        # img = cv2.imread(path)
        #     # Reduce size
        # resized = cv2.resize(img, (img_cols, img_rows))#前面为宽，后面为高
        # imgs.append(resized)

    train_imgs = train_imgs[1:, :]
    return train_imgs
    # return np.array(imgs).reshape(len(paths), img_rows, img_cols)


def get_train_batch(X_train, y_train, batch_size, img_w, img_h):
   '''
   参数：
       X_train：所有图片路径列表
       y_train: 所有图片对应的标签列表
       batch_size:批次
       img_w:图片宽
       img_h:图片高
       color_type:图片类型
       is_argumentation:是否需要数据增强
   返回:
       一个generator，

    x: 获取的批次图片

    y: 获取的图片对应的标签
   '''
   while 1:

    for i in range(0, len(X_train), batch_size):
           x = get_im_cv2(X_train[i:i+batch_size], img_w, img_h)
           y = y_train[i:i+batch_size]

# 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
           yield({'input': x}, {'output': y})
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

    train_labels = train_labels[1:, :]

    # 编译模型
    model = build_model(keep_prob)
    # 在数据集上训练模型，保存成model.h5
    train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size,files,train_labels)
    print("模型训练完毕")