import cv2
import numpy as np
import pytesseract as test  # OCR 识别
from PIL import Image
import StackImage as SI
import ordinaryBlankCard as BasicOperation

#################################
# 1.read blank ID card image
blank_card = cv2.imread('G:/ProgramSpace/opencv/Image/ZjPh.jpg')
blackImage = np.zeros_like(blank_card) # Create a black image of the same size
blankCopy = blank_card.copy()  # create a copy image
# 2.Basic Operation
BasicList = BasicOperation.BasicProcess(blankCopy)
if len(BasicList) == 5:
    images = [blank_card, BasicList[0], BasicList[1],
              BasicList[2],BasicList[3]] # creat Stack Image list
else:
    images = [blank_card, blackImage]

# BasicList[4]得到的就是前景照片
# cv2.imshow("dilate_binary", BasicList[4])  # show stack images

# Process contour，ROI列表接受的是数字区域的图像
ROI = BasicOperation.ProcessContours(BasicList[4],blankCopy)
# cv2.imshow("ROI[0]", ROI[0])
# cv2.imshow('ROI[1]', ROI[1])
BasicOperation.NumberOutput(ROI)

# 5. show blank_card
stackImages = SI.stackImage(images,0.5)  # call stack function
cv2.imshow("stackImages", stackImages)  # show stack images


# 3. last control
cv2.waitKey(0)  # wait user input
cv2.destroyAllWindows()  # free memory


#################################
