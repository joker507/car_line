# -*- coding: utf-8 -*-
# @Time : 2020/8/17 15:46
# @Author : XXX
# @Site : 
# @File : test.py
# @Software: PyCharm 
import Demo
import cv2
line = Demo.Car_lines()
img = cv2.imread('lines/alway.jpg')
result, data = line.search(img,colors=[[[0, 110, 110],[100, 255, 255]]])
print(data)
Demo.Show(result)