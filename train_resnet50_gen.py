# -*- coding: utf-8 -*-
import os
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random


PATH = 'training_data'

class PowerTransferMode:

    # ResNet模型
    def ResNet50_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=5, img_rows=120, img_cols=160, RGB=True,
                       is_plot_model=False):
        color = 3 if RGB else 1
        base_model = ResNet50(weights='imagenet', include_top=False, pooling=None,
                              input_shape=(img_rows, img_cols, color),
                              classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = Flatten()(x)
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘制模型
        if is_plot_model:
            plot_model(model, to_file='resnet50_model.png', show_shapes=True)

        return model


    # 训练模型
    def train_model(self, model, epochs, train_generator, steps_per_epoch, validation_generator, validation_steps,
                    model_url, is_load_model=False):
        # 载入模型
        if is_load_model and os.path.exists(model_url):
            model = load_model(model_url)

        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)
        # 模型保存
        model.save(model_url, overwrite=True)
        return history_ft

    # 画图
    def plot_training(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'b-')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')
        plt.figure()
        plt.plot(epochs, loss, 'b-')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.show()

def get_im_cv2(paths):
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


def get_batch(X_train, y_train, batch_size):
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
           x = get_im_cv2(X_train[i:i+batch_size])
           y = y_train[i:i+batch_size]
# 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
           yield(x, y)

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

if __name__ == '__main__':
    image_size = 224
    batch_size = 32
    path = "training_data"
    files = os.listdir(path)

    random.shuffle(files)

    keep_prob = 0.5
    nb_epoch = 100
    samples_per_epoch = len(files)

    print('keep_prob = ', keep_prob)
    print('nb_epoch = ', nb_epoch)
    print('samples_per_epoch = ', samples_per_epoch)
    print('batch_size = ', batch_size)
    print('-' * 30)

    labels = np.zeros((1, 5), 'float')
    print("生成标签矩阵。。。")
    for file in files:
        if not os.path.isdir(file) and file[len(file) - 3:len(file)] == 'jpg':
            try:
                key = int(file[0])
                label_array = process_img(path + "/" + file, key)
                labels = np.vstack((labels, label_array))
            except:
                print('prcess error')

    train_num = round(samples_per_epoch * 0.8)
    valid_num = samples_per_epoch - train_num
    train_files = files[0: train_num]
    valid_files = files[train_num:, :]
    train_labels = labels[0:, train_num]
    valid_labels = labels[train_num:, :]

    train_generator = get_batch(train_files,train_labels,batch_size)
    validation_generator = get_batch(valid_files,valid_labels,batch_size)

    transfer = PowerTransferMode()


      # ResNet50
    model = transfer.ResNet50_model(nb_classes=5, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    history_ft = transfer.train_model(model, nb_epoch, train_generator, train_num // batch_size, validation_generator, valid_num//batch_size,
                                      'resnet50_model_weights.h5', is_load_model=False)

    # 训练的acc_loss图
    transfer.plot_training(history_ft)