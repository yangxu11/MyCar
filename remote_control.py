import io
import car_control1
import os
os.environ['SDL_VIDEODRIVE'] = 'x11'
import pygame     # 检测模块
from time import ctime,sleep,time
import threading
import numpy as np

global train_labels, train_img, is_capture_running, key


def my_car_control():
    global is_capture_running, key
    key = 4
    pygame.init()
    pygame.display.set_mode((1, 1))  # 窗口
    car_control1.carStop()
    sleep(0.1)
    print("Start control!")

    while is_capture_running:
        # get input from human driver
        #
        for event in pygame.event.get():
            # 判断事件是不是按键按下的事件
            if event.type == pygame.KEYDOWN:
                key_input = pygame.key.get_pressed()  # 可以同时检测多个按键
                print(key_input[pygame.K_w], key_input[pygame.K_a], key_input[pygame.K_d])
                # 按下前进，保存图片以2开头
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    car_control1.carMoveForward()
                # 按下左键，保存图片以0开头
                elif key_input[pygame.K_a]:
                    print("Left")
                    car_control1.carForwardLeft()
                    sleep(0.1)
                    key = 0
                # 按下d右键，保存图片以1开头
                elif key_input[pygame.K_d]:
                    print("Right")
                    car_control1.carForwardRight()
                    sleep(0.1)
                    key = 1
                # 按下s后退键，保存图片为3开头
                elif key_input[pygame.K_s]:
                    print("Backward")
                    car_control1.carMoveBack()
                    key = 3
                # 按下k停止键，停止
                elif key_input[pygame.K_k]:
                    car_control1.carStop()
                    # 按下d右键，保存图片以1开头

                elif key_input[pygame.K_o]:
                    print("Out")
                    is_capture_running = False
                    break
            # 检测按键是不是抬起
            elif event.type == pygame.KEYUP:
                key_input = pygame.key.get_pressed()
                # w键抬起，轮子回正
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    car_control1.carMoveForward()
                # s键抬起
                elif key_input[pygame.K_s] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Backward")
                    key = 3
                    car_control1.carMoveBack()
                else:
                    print("Stop")
                    car_control1.carStop()
    car_control1.clean()


if __name__ == '__main__':
     global train_labels, train_img, key,is_capture_running

    print("capture thread")
    print('-' * 50)
    # capture_thread = threading.Thread(target=pi_capture, args=())  # 开启线程
    # capture_thread.setDaemon(True)
    # capture_thread.start()

    is_capture_running = True

    my_car_control()

    while is_capture_running:
        pass#占位   空语句

    print("Done!")
    car_control1.carStop()
    car_control1.clean()
