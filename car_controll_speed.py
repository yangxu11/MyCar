import RPi.GPIO as GPIO
import time

motor1_1 = 7
motor1_2 = 11#前右轮

motor2_1 = 13
motor2_2 = 15#前左轮

en1 = 12
en2 = 16

speed0 = 10#转弯速
speed1 = 15#低速
speed2 = 25#中速
speed3 = 30#快速

special_speed = 28 #右转时快轮的速度  特殊情况

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
    GPIO.output(motor1_1, GPIO.LOW)
    GPIO.output(motor1_2, GPIO.LOW)  # 电机1停转
    GPIO.output(motor2_1, GPIO.LOW)
    GPIO.output(motor2_2, GPIO.LOW)  # 电机2停转

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

def carMoveForward(speed):
    wheelForward()
    carInit()
    changeSpeed(speed,speed) #低速

def carMoveBack(speed):
    wheelBack()
    carInit()
    changeSpeed(speed,speed) #低速

def carForwardRight(speedR,speedL):
    wheelForward()
    carInit()
    changeSpeed(speedR,speedL)#右轮 转弯速  左轮 低速

def carForwardLeft():
    wheelForward()
    carInit()
    changeSpeed(speed2,speed0)#右轮 低速 左轮 转弯速

def carBackRight():
    wheelBack()
    carInit()
    changeSpeed(speed0,speed2)#右轮 转弯速  左轮 低速

def carBackLeft():
    wheelBack()
    carInit()
    changeSpeed(speed2, speed0)#右轮 低速 左轮 转弯速


def clean():
    GPIO.cleanup()
    motor1Pwm.stop()
    motor2Pwm.stop()

def speedUp():
    changeSpeed(speed2,speed2)

def speedDown():
    changeSpeed(speed1,speed1)

if __name__ == '__main__':
    pass