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
speed_step = 1

special_speed = 28 #右转时快轮的速度  特殊情况

global speedLeft,speedRight,speedStraight




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


#切换轮子速度
def changeSpeed(right,left):
    global speedLeft, speedRight, speedStraight
    motor1Pwm.ChangeDutyCycle(right) #右轮
    motor2Pwm.ChangeDutyCycle(left)  #左轮
    speedRight = right
    speedLeft = left

def carStraight():
    carInit()
    changeSpeed(speedStraight,speedStraight)

def carSpeedUp():
    global speedLeft, speedRight, speedStraight
    # if speedLeft != speedRight : return
    if speedRight + speed_step < 25:
        speedRight = speedRight + speed_step
    else:
        speedRight = 25

    if speedRight + speed_step < 25:
        speedLeft = speedLeft + speed_step
    else:
        speedLeft = 25
    changeSpeed(speedRight, speedLeft)

def carSpeedDown():
    global speedLeft, speedRight, speedStraight
    #if speedLeft != speedRight : return
    if speedRight-speed_step > 10 :
        speedRight = speedRight - speed_step
    else :
        speedRight = 10

    if speedLeft-speed_step > 10:
        speedLeft = speedLeft - speed_step
    else :
        speedLeft = 10
    changeSpeed(speedRight,speedLeft)

def carLeft():
    global speedLeft,speedRight,speedStraight
    # left减小 或者 right增大
    if speedRight >= speedLeft:
        if speedRight < 25:
            speedRight = speedRight + speed_step
        else :
            speedRight = 25
    else:
        if speedLeft > 10:
            speedLeft = speedLeft - speed_step
        else :
            speedLeft = 10
    changeSpeed(speedRight,speedLeft)


def carRight():
    global speedLeft, speedRight, speedStraight
    #left增大 或者 right减小
    if speedRight <= speedLeft:
        if speedLeft < 25:
            speedLeft = speedLeft + speed_step
        else:
            speedRight = 25
    else:
        if speedRight > 10:
            speedRight = speedRight - speed_step
        else:
            speedRight = 10
    changeSpeed(speedRight, speedLeft)


#给电机初始短暂的高占空比，使其能动起来
def carInit():
    wheelForward()
    motor1Pwm.start(30)  # p.start(dc)   # dc 代表占空比（范围：0.0 <= dc <= 100.0）
    motor2Pwm.start(30)
    time.sleep(0.1)
    changeSpeed(speed1,speed1)

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

def getSpeedRight():
    global speedRight
    return speedRight

def getSpeedLeft():
    global speedLeft
    return speedLeft

def clean():
    GPIO.cleanup()
    motor1Pwm.stop()
    motor2Pwm.stop()

if __name__ == '__main__':
    pass