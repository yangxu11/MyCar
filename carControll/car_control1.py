import RPi.GPIO as GPIO
import time

motor1_1 = 7
motor1_2 = 11#前右轮

motor2_1 = 13
motor2_2 = 15#前左轮

en1 = 12
en2 = 16

speed0 = 10#低速
speed1 = 15#中速
speed2 = 25#快速

global speedLeft,speedRight

GPIO.setmode(GPIO.BOARD)  # 物理引脚board编码格式

# 设置管脚为输出
GPIO.setup(motor1_1, GPIO.OUT)
GPIO.setup(motor1_2, GPIO.OUT)
GPIO.setup(motor2_1, GPIO.OUT)
GPIO.setup(motor2_2, GPIO.OUT)


GPIO.setup(en1, GPIO.OUT)  # 电机1使能
GPIO.setup(en2, GPIO.OUT)  # 电机2使能



motor1Pwm = GPIO.PWM(en1,80)# p = GPIO.PWM(channel, frequency)
motor2Pwm = GPIO.PWM(en2,80)
motor1Pwm.start(30)  # p.start(dc)   # dc 代表占空比（范围：0.0 <= dc <= 100.0）
motor2Pwm.start(30)

#给电机初始短暂的高占空比，使其能动起来
def carInit():
    motor1Pwm.start(30)  # p.start(dc)   # dc 代表占空比（范围：0.0 <= dc <= 100.0）
    motor2Pwm.start(30)
    time.sleep(0.1)

def carStop():#电机1，2使能关闭
    GPIO.output(en1, GPIO.LOW)
    GPIO.output(en2, GPIO.LOW)

#轮子正转
def wheelForward():
    GPIO.output(motor1_1, GPIO.HIGH)
    GPIO.output(motor1_2, GPIO.LOW)  # 电机1正转
    GPIO.output(motor2_1, GPIO.HIGH)
    GPIO.output(motor2_2, GPIO.LOW)  # 电机2正转

#轮子反转
def wheelBack():
    GPIO.output(motor1_1, GPIO.LOW)
    GPIO.output(motor1_2, GPIO.HIGH)  # 电机1反转
    GPIO.output(motor2_1, GPIO.LOW)
    GPIO.output(motor2_2, GPIO.HIGH)  # 电机2反转

#切换轮子速度
def changeSpeed(right,left):
    motor1Pwm.ChangeDutyCycle(right) #右轮
    motor2Pwm.ChangeDutyCycle(left)  #左轮
    speedRight = right
    speedLeft = left



# def carStraight():#启动准备
#     GPIO.output(en1, GPIO.HIGH)
#     GPIO.output(en1, GPIO.HIGH)
#     motor1Pwm.ChangeDutyCycle(speed1)
#     motor2Pwm.ChangeDutyCycle(speed1)
#     GPIO.output(motor1_1, GPIO.HIGH)
#     GPIO.output(motor1_2, GPIO.HIGH)  #电机1停
#     GPIO.output(motor2_1, GPIO.HIGH)
#     GPIO.output(motor2_2, GPIO.HIGH)  #电机2停


def carMoveForward():
    wheelForward()
    carInit()
    changeSpeed(speed0,speed0) #低速

def carMoveBack():
    wheelBack()
    carInit()
    changeSpeed(speed0,speed0) #低速

def carForwardRight():
    wheelForward()
    carInit()
    changeSpeed(speed0,speed1)#右轮 低速  左轮 中速

def carForwardLeft():
    wheelForward()
    carInit()
    changeSpeed(speed1,speed0)#右轮 中速 左轮 低速

# def carBackRight():
#     motor1Pwm.ChangeDutyCycle(speed0)#右轮 低速
#     motor2Pwm.ChangeDutyCycle(speed1) #左轮 中速
#     GPIO.output(motor1_1, GPIO.LOW)
#     GPIO.output(motor1_2, GPIO.HIGH)  #电机1反转
#     GPIO.output(motor2_1, GPIO.LOW)
#     GPIO.output(motor2_2, GPIO.HIGH)  #电机2反转
#
#
# def carBackLeft():
#     motor1Pwm.ChangeDutyCycle(speed0)#右轮 中速
#     motor2Pwm.ChangeDutyCycle(speed1) #左轮 低速
#     GPIO.output(motor1_1, GPIO.LOW)
#     GPIO.output(motor1_2, GPIO.HIGH)  #电机1反转
#     GPIO.output(motor2_1, GPIO.LOW)
#     GPIO.output(motor2_2, GPIO.HIGH)  #电机2反转


def clean():
    GPIO.cleanup()
    motor1Pwm.stop()
    motor2Pwm.stop()


if __name__ == '__main__':
    carMoveForward()
    time.sleep(2)

    carStop()
    time.sleep(0.5)

    carMoveBack()
    time.sleep(2)

    carStop()
    time.sleep(0.5)

    carMoveForward()
    time.sleep(1)

    carForwardLeft()
    time.sleep(1)

    carForwardRight()
    time.sleep(1)

    carStop()
    clean()