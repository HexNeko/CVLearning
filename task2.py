import cv2
import numpy as np

#对比图一图二是不是一个人
img1 = cv2.imread('face_img/test1.jpg')
img2 = cv2.imread('face_img/test2.jpg')

new_shape = (300, 300)   # 统一缩放为 300*300
cos_thresh = 0.363       # cos阈值，距离越大越接近
L2_thresh = 1.128        # L2阈值，距离越小越接近
img1 = cv2.resize(img1, new_shape)
img2 = cv2.resize(img2, new_shape)

# 初始化模型：
faceDetector = cv2.FaceDetectorYN_create(model='model/yunet.onnx', config='', input_size=new_shape)
faceRecognizer = cv2.FaceRecognizerSF_create(model='model/face_recognizer_fast.onnx', config='')

# 检测、对齐、提取特征：
# detect输出的是一个二维元祖，其中第二维是一个二维数组: n*15,n为人脸数，
# 15为人脸的xywh和5个关键点（右眼瞳孔、左眼、鼻尖、右嘴角、左嘴角）的xy坐标及置信度
faces1 = faceDetector.detect(img1)  
aligned_face1 = faceRecognizer.alignCrop(img1, faces1[1][0])    # 对齐后的图片
feature1 = faceRecognizer.feature(aligned_face1)                # 128维特征

faces2 = faceDetector.detect(img2)
aligned_face2 = faceRecognizer.alignCrop(img2, faces2[1][0])
feature2 = faceRecognizer.feature(aligned_face2)

# 框出人脸并画出人脸的关键点
faces_detect = (faces1[1], faces2[1])
images = (img1, img2)
for i in range(len(images)):
    for face_index, face_coords in enumerate(faces_detect[i]):
        thickness = 2
        # 坐标转成int类型
        coords = face_coords[:-1].astype(np.int32)
        # 框出人脸
        cv2.rectangle(images[i], (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 1)
        # 画出左右瞳孔、左右嘴角、鼻尖的位置
        cv2.circle(images[i], (coords[4], coords[5]), 1, (255, 0, 0), thickness)
        cv2.circle(images[i], (coords[6], coords[7]), 1, (0, 0, 255), thickness)
        cv2.circle(images[i], (coords[8], coords[9]), 1, (0, 255, 0), thickness)
        cv2.circle(images[i], (coords[10], coords[11]), 1, (255, 0, 255), thickness)
        cv2.circle(images[i], (coords[12], coords[13]), 1, (0, 255, 255), thickness)
        # 显示图片
        cv2.putText(images[i], 'face%d'%(i), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
# 人脸匹配值打分：
cos_score = faceRecognizer.match(feature1, feature2, 0)
L2_score = faceRecognizer.match(feature1, feature2, 1)
output = "cos_score:%.6f"%(cos_score) + "  L2_score:%.6f"%(L2_score)
# 输出结果：
print('cos_score: ', cos_score)
print('L2_score: ', L2_score)

if cos_score > cos_thresh:
    print('same face')
    output += "  same"
else:
    print('diffrent face')
    output += "  diffrent"

if L2_score < L2_thresh:
    print('same face')
    output += "/same"
else:
    print('diffrent face')
    output += "/diffrent"
    
res = np.concatenate((img1, img2), axis=1)
cv2.putText(res, output, (1, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imshow("res",res)

cv2.waitKey(0)