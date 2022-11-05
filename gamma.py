import cv2
import numpy as np
#分道计算每个通道的直方图
img0 = cv2.imread('0001.bmp')
hist_b = cv2.calcHist([img0],[0],None,[256],[0,256])
hist_g = cv2.calcHist([img0],[1],None,[256],[0,256])
hist_r = cv2.calcHist([img0],[2],None,[256],[0,256])
def gamma_trans(img,gamma):
    #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    #实现映射用的是Opencv的查表函数
    return cv2.LUT(img0,gamma_table)
img0_corrted = gamma_trans(img0, 0.6)
cv2.imshow('img0',img0)
cv2.imshow('gamma_image',img0_corrted)
cv2.imwrite('gamma_image.png',img0_corrted)
cv2.waitKey()
