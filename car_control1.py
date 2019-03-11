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
speed2 = 30#快速

GPIO.setmode(GPIO.BOARD)  # 物理引脚board编码格式

# 设置管脚为输出
GPIO.setup(motor1_1, GPIO.OUT)
GPIO.setup(motor1_2, GPIO.OUT)
GPIO.setup(motor2_1, GPIO.OUT)
GPIO.setup(motor2_2, GPIO.OUT)


GPIO.setup(en1, GPIO.OUT)  # 电机1使能
GPIO.setup(en2, GPIO.OUT)  # 电机2使能



motor1Pwm = GPIO.PWM(en1, 100)  # p = GPIO.PWM(channel, frequency)
motor2Pwm = GPIO.PWM(en2, 100)
motor1Pwm.start(0)  # p.start(dc)   # dc 代表占空比（范围：0.0 <= dc <= 100.0）
motor2Pwm.start(0)

def carStop():#电机1，2使能关闭
    GPIO.output(en1, GPIO.LOW)
    GPIO.output(en2, GPIO.LOW)

def carStraight():#启动准备
    GPIO.output(en1, GPIO.HIGH)
    GPIO.output(en1, GPIO.HIGH)
    motor1Pwm.ChangeDutyCycle(speed1)
    motor2Pwm.ChangeDutyCycle(speed1)
    GPIO.output(motor1_1, GPIO.HIGH)
    GPIO.output(motor1_2, GPIO.HIGH)  #电机1停
    GPIO.output(motor2_1, GPIO.HIGH)
    GPIO.output(motor2_2, GPIO.HIGH)  #电机2停


def carMoveForward():
    motor1Pwm.ChangeDutyCycle(speed2)
    motor2Pwm.ChangeDutyCycle(speed2) #快速
    GPIO.output(motor1_1, GPIO.LOW)
    GPIO.output(motor1_2, GPIO.HIGH)  #电机1正转
    GPIO.output(motor2_1, GPIO.LOW)
    GPIO.output(motor2_2, GPIO.HIGH)  #电机2正转

def carMoveBack():
    motor1Pwm.ChangeDutyCycle(speed1)
    motor2Pwm.ChangeDutyCycle(speed1)  # 中速
    GPIO.output(motor1_1, GPIO.HIGH)
    GPIO.output(motor1_2, GPIO.LOW)  # 电机1反转
    GPIO.output(motor2_1, GPIO.HIGH)
    GPIO.output(motor2_2, GPIO.LOW)  # 电机2反转

def carForwardRight():
    motor1Pwm.ChangeDutyCycle(speed0)#右轮 低速
    motor2Pwm.ChangeDutyCycle(speed1) #左轮 中速
    GPIO.output(motor1_1, GPIO.LOW)
    GPIO.output(motor1_2, GPIO.HIGH)  #电机1正转
    GPIO.output(motor2_1, GPIO.LOW)
    GPIO.output(motor2_2, GPIO.HIGH)  #电机2正转

def carBackRight():
    motor1Pwm.ChangeDutyCycle(speed0)#右轮 低速
    motor2Pwm.ChangeDutyCycle(speed1) #左轮 中速
    GPIO.output(motor1_1, GPIO.HIGH)
    GPIO.output(motor1_2, GPIO.LOW)  # 电机1反转
    GPIO.output(motor2_1, GPIO.HIGH)
    GPIO.output(motor2_2, GPIO.LOW)  # 电机2反转


def carForwardLeft():
    motor1Pwm.ChangeDutyCycle(speed1)#右轮 中速
    motor2Pwm.ChangeDutyCycle(speed0) #左轮 低速
    GPIO.output(motor1_1, GPIO.LOW)
    GPIO.output(motor1_2, GPIO.HIGH)  #电机1正转
    GPIO.output(motor2_1, GPIO.LOW)
    GPIO.output(motor2_2, GPIO.HIGH)  #电机2正转

def carBackLeft():
    motor1Pwm.ChangeDutyCycle(speed0)#右轮 中速
    motor2Pwm.ChangeDutyCycle(speed1) #左轮 低速
    GPIO.output(motor1_1, GPIO.HIGH)
    GPIO.output(motor1_2, GPIO.LOW)  # 电机1反转
    GPIO.output(motor2_1, GPIO.HIGH)
    GPIO.output(motor2_2, GPIO.LOW)  # 电机2反转


def clean():
    GPIO.cleanup()
    motor1Pwm.stop()
    motor2Pwm.stop()


if __name__ == '__main__':
    carStraight()
    carMoveForward()
    time.sleep(2)

    carStop()
    time.sleep(2)

    carStraight()
    carForwardLeft()
    time.sleep(2)
    carForwardRight()
    time.sleep(2)
    carBackRight()
    time.sleep(2)
    carBackLeft()
    time.sleep(2)

    carMoveBack()
    time.sleep(2)
    carStraight()
    carStop()

    clean()