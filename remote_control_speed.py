import io
import car_controll_speed as car
import os
os.environ['SDL_VIDEODRIVE'] = 'x11'
import pygame     # 检测模块
from time import ctime,sleep,time
import threading
import numpy as np

global train_labels, train_img, is_capture_running, key,speedRight,speedLeft


def my_car_control():
    global is_capture_running, key,speedRight,speedLeft
    key = 4
    pygame.init()
    pygame.display.set_mode((1, 1))  # 窗口
    car.carStop()
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
                    car.carSpeedUp()
                    sleep(0.1)
                    speedRight = car.getSpeedRight()
                    speedLeft = car.getSpeedLeft()
                # 按下左键，保存图片以0开头
                elif key_input[pygame.K_a]:
                    print("Left")
                    car.carLeft()
                    sleep(0.1)
                    speedRight = car.getSpeedRight()
                    speedLeft = car.getSpeedLeft()
                    key = 0
                # 按下d右键，保存图片以1开头
                elif key_input[pygame.K_d]:
                    print("Right")
                    car.carRight()
                    sleep(0.1)
                    speedRight = car.getSpeedRight()
                    speedLeft = car.getSpeedLeft()
                    key = 1
                # 按下s后退键，保存图片为3开头
                elif key_input[pygame.K_s]:
                    print("Backward")
                    car.carSpeedDown()
                    sleep(0.1)
                    speedRight = car.getSpeedRight()
                    speedLeft = car.getSpeedLeft()
                    key = 3
                # 按下k停止键，停止
                elif key_input[pygame.K_k]:
                    car.carStop()
                    # 按下d右键，保存图片以1开头

                elif key_input[pygame.K_q]:
                    car.carInit()

                elif key_input[pygame.K_o]:
                    print("Out")
                    is_capture_running = False
                    break
            # 检测按键是不是抬起
            # elif event.type == pygame.KEYUP:
            #     key_input = pygame.key.get_pressed()
            #     # w键抬起，轮子回正
            #     if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
            #         print("Forward")
            #         speedRight = car.getSpeedRight()
            #         speedLeft = car.getSpeedLeft()
            #         key = 2
            #         car.carSpeedUp()
            #     # s键抬起
            #     elif key_input[pygame.K_s] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
            #         print("Backward")
            #         speedRight = car.getSpeedRight()
            #         speedLeft = car.getSpeedLeft()
            #         key = 3
            #         car.carMoveBack()
            #     else:
            #         print("Stop")
            #         car.carStop()
    car.clean()


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
    car.carStop()
    car.clean()