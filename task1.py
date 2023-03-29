import cv2 as cv
import numpy as np

def distence(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

img_path = "test.jpg"

#读入图像
img = cv.imread(img_path,0)
height,width = img.shape
img_color = cv.imread(img_path,1)
cv.imshow('input_img',img_color)

#Canny边缘检测算法得到边缘图像
edges = cv.Canny(img,100,200)
#cv.imshow('Canny',edges)

#根据二值图找出封闭轮廓
_,edges = cv.threshold(edges,127,255,cv.THRESH_BINARY)
contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # 根据二值图找轮廓
contour = np.zeros(img.shape)
index,max_p = 0,0
#找出封闭轮廓，认为点最多的点集为文档的轮廓
for i in range(0,len(contours)):
    count = len(contours[i])
    if count > max_p:
        index,max_p = i,count
cv.drawContours(contour,contours,index,255,1)
contour = contour.astype(np.uint8)
#cv.imshow('contour',contour)

#对轮廓图进行Harris角点检测
mat = np.zeros(img.shape)
mat = cv.cornerHarris(contour,blockSize=2,ksize=3,k=0.04)
mat = cv.normalize(mat,0,255,cv.NORM_MINMAX)
R = mat.max() * 0.1  #将阈值设为最大值的0.1 大于这个值的数认为是数角点
point = np.zeros(img.shape)
point[mat>R] = 255
#cv.imshow('Harris',point)

#寻找文档的四个角点
max_d = distence((0,0),img.shape)
left_top=[0,0,max_d]
left_bottom=[height,0,max_d]
right_top=[0,width,max_d]
right_bottom=[height,width,max_d]

#这里认为距离图片四个角距离最近的点为四个角点
for x,y in np.argwhere(point == 255):
    d1 = distence((0,0),(x,y))
    d2 = distence((height,0),(x,y))
    d3 = distence((0,width),(x,y))
    d4 = distence((height,width),(x,y))
    if d1 < left_top[2]:
        left_top = [y,x,d1]
    if d2 < left_bottom[2]:
        left_bottom = [y,x,d2]
    if d3 < right_top[2]:
        right_top = [y,x,d3]
    if d4 < right_bottom[2]:
        right_bottom = [y,x,d4]
        
#对图像做投影变换
input_pts = np.float32([left_top[0:2],left_bottom[0:2],right_bottom[0:2],right_top[0:2]])
output_pts = np.float32([[0,0], [0,height-1], [width-1,height-1], [width-1,0]])
M = cv.getPerspectiveTransform(input_pts,output_pts)
final = cv.warpPerspective(img_color,M,(width,height),flags=cv.INTER_LINEAR)

#绘制角点和轮廓并显示和输出
cv.drawContours(img_color,contours,index,(0,0,255),2)
for p in [left_top,left_bottom,right_top,right_bottom]:
    cv.circle(img_color, p[0:2], 4, (255,0,0), 10);
cv.imshow("contour_and_point", img_color)
cv.imshow("final", final)
cv.imwrite("ouput_mark.jpg",img_color)
cv.imwrite("ouput_final.jpg",final)
cv.waitKey(0)