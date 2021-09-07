#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
* @par Copyright (C): 2010-2020, hunan CLB Tech
* @file         Basic_movement
* @version      V2.0
* @details
* @par History

@author: zhulin
"""
import RPi.GPIO as GPIO
import time
import os
import sys
import socket

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24


# newList = []
def t_up(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, False)  # AIN2  左后
    GPIO.output(AIN1, True)  # AIN1  左前

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, False)  # BIN2 右后
    GPIO.output(BIN1, True)  # BIN1 右前


def t_stop():
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2, False)  # AIN2
    GPIO.output(AIN1, False)  # AIN1

    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2, False)  # BIN2
    GPIO.output(BIN1, False)  # BIN1


def t_down(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, True)  # AIN2
    GPIO.output(AIN1, False)  # AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, True)  # BIN2
    GPIO.output(BIN1, False)  # BIN1


def t_left(speed):
    L_Motor.ChangeDutyCycle(speed - 30)
    GPIO.output(AIN2, False)  # AIN2  左后
    GPIO.output(AIN1, True)  # AIN1  左前

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, False)  # BIN2 右后
    GPIO.output(BIN1, True)  # BIN1 右前


def t_right(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, False)  # AIN2  左后
    GPIO.output(AIN1, True)  # AIN1  左前

    R_Motor.ChangeDutyCycle(speed - 30)
    GPIO.output(BIN2, False)  # BIN2 右后
    GPIO.output(BIN1, True)  # BIN1 右前


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # GPIO为通用输入输出端口
GPIO.setup(AIN2, GPIO.OUT)  # 设置输出端口
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)

GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)

# PWM可以简单理解为，通过快速的高低电平的切换，达到控制电机的有效电压的效果，进而控制小车的速度
L_Motor = GPIO.PWM(PWMA, 100)  # 进入通用输入输出端口的脉冲调制模式，设置（输出：之前定义）端口PWMA的频率（高低电平）为100Hz
L_Motor.start(0)  # 启用

R_Motor = GPIO.PWM(PWMB, 100)
R_Motor.start(0)


# try:  # 主函数

def start():
    t_stop()
    server.listen(5)  # 监听
    print("我要开始接收数据了")
    while True:
        conn, addr = server.accept()  # 等数据进来
        # conn就是客户端连过来而在服务器端为其生成的一个连接实例
        print("收到来自{}请求".format(addr))
        while True:
            data = conn.recv(1024)  # 接收数据，获取图像名称的命令，指定需要传输的图片
            if not data:
                break
            print("recv:", data)
            if data.decode('utf-8') == '0':
                t_up(50)
            elif data.decode('utf-8') == '4':
                t_stop()
            elif data.decode('utf-8') == '5':
                t_stop()
            elif data.decode('utf-8') == '2':
                t_down(50)
            elif data.decode('utf-8') == '3':
                t_left(50)
            elif data.decode('utf-8') == '1':
                t_right(50)
        break


# while True:
server = socket.socket()  # 1.声明协议类型，同时生成socket链接对象  server is 250(pi)
server.bind(('192.168.43.250', 6666))  # 绑定要监听端口=(服务器的ip地址+任意一个端口)

while True:
    start()
