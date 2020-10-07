import cv2 as cv
import numpy as np
import pytesseract as test  # OCR 识别
from PIL import Image

############################
# if __name__ == '__main__':
#     MyPonits = []
MyPoints = []

def BasicProcess(image):
    '''
    :param image: Original image
    :function: Median Filtering and gray and morphologyEx
    :return: list
    '''
    BasicList = []
    # 1. Median Filtering(中值滤波处理)
    BlurImage = cv.medianBlur(image, 5)  # 卷积核为5*5
    BasicList.append(BlurImage)

    # 2. Gray chance
    GrayImage = cv.cvtColor(BlurImage, cv.COLOR_BGR2GRAY)
    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 创建模板
    GrayImage = cv.dilate(GrayImage, kernel = dilate_kernel)  # 调用膨胀的API
    BasicList.append(GrayImage)

    # 3. morphologyEx-->blackHat(得到前景ID) and dilate
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 得到结构元素
    blackHat = cv.morphologyEx(GrayImage, cv.MORPH_BLACKHAT, kernel=kernel)  # 黑帽
    blackHat = cv.add(blackHat, 150)
    BasicList.append(blackHat)

    # 4. Binarization(二值化处理)
    ret, Binary_image = cv.threshold(blackHat, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # if ret is False: break
    BasicList.append(Binary_image)

    # 5. morphologyEx-->MORPH_CLOSE
    Close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 创建模板
    # dilate_binary = cv.dilate(Binary_image,kernel=dilate_kernel, iterations = 2)  #调用膨胀的API,执行2次
    dilate_binary = cv.morphologyEx(Binary_image, cv.MORPH_CLOSE, kernel = Close_kernel, iterations = 7)
    BasicList.append(dilate_binary)

    # 6. return basic list
    return BasicList

def GetContourPoints(background, blankCopy):
    backgroundCopy = background.copy()
    contourList = []
    # 1. find all contours
    contours, hireachy = cv.findContours(background, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 2. ergodic(遍历) all contours
    for i, contour in enumerate(contours):
        peri = cv.arcLength(contour, True)  # 计算每一个轮廓的周长
        approxCurve = cv.approxPolyDP(contours[i], 0.02 * peri, True)  # 得到拟变形轮廓
        area = cv.contourArea(contour)
        # 3. area 刷选
        if (area >= 1500 and area <=2000) or (area >= 4000):
            if len(approxCurve) >= 4 or len(approxCurve) <= 5:
                contourList.append(contours[i])  # 将当前轮廓信息保存
                # x, y, w, h = cv.boundingRect(contours[i])  # 得到轮廓（白色区域为图像）外接矩形的坐标
                # cv.rectangle(blankCopy, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 绘制外接矩形
                # cv.imshow("rectangle", blankCopy)

    # 4. 将满足区域的轮廓按照面积大小进行降序排列
    contourList = sorted(contourList, key = cv.contourArea, reverse=True)
    return contourList

def ProcessContours(background, blankCopy):
    '''
    :param background:前景图像
    :param blankCopy:原图像的副本
    :return: 返回的两个图像-->数字区域的图像
    '''
    contourList = GetContourPoints(background, blankCopy)
    ROI = []
    # 1. 处理大的区域输出图像
    if len(contourList) == 2:
        for i in range(len(contourList)):
            # peri = cv.arcLength(contourList[i], True)  # 计算每一个轮廓的周长
            # approx = cv.approxPolyDP(contourList[i], 0.02 * peri, True)  # 得到拟变形轮廓
            x, y, w, h = cv.boundingRect(contourList[i])  # 得到轮廓（白色区域为图像）外接矩形的坐标
            ROI.append(blankCopy[y:y+h, x:x+w])
            point = [x, y, w, h]
            MyPoints.append(point)  # 储存坐标信息
    return ROI

def contourFill(img):
    '''
    :Function: 使用填充函数的原因是OCR检测最好是扩大边缘
    :param img:
    :return:
    '''
    top_size, bottom_size, left_size, right_size = (100,100,50,50)#指定上下左右填充的元素个数
    imgNew = cv.copyMakeBorder(img,top_size, bottom_size, left_size, right_size,
           borderType=cv.BORDER_CONSTANT, value = 255)  # 调用边界填充API
    return imgNew

def NumberOutput(ROI):
    textlist = []
    for i in range(0,len(ROI)):
        # 1. gray
        GrayImage = cv.cvtColor(ROI[i], cv.COLOR_BGR2GRAY)
        # 2. Binary
        ret, Binary_image = cv.threshold(GrayImage, 10, 255, cv.THRESH_BINARY )

        cv.imshow("ROI[{}]".format(i), Binary_image)
        # 3. 调用边界填充函数
        Binary_image = contourFill(Binary_image)
        cv.imshow("ROI[{}]".format(i), Binary_image)

        # 3. OCR识别+
        textImage = Image.fromarray(Binary_image)  # 变为text图像
        text = test.image_to_string(textImage)  # OCR识别
        textlist.append(text)

    print('你的银行卡卡号为:{}'.format(textlist[1]+textlist[0]))




