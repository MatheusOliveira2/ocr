import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image as im

#import image
<<<<<<< HEAD
filename = '/Users/CromAI/Documents/ocr/ocr/exemplos/exemplo10.jpeg'
image = cv2.imread(filename)

#image_display = cv2.resize(image,(1024,1048))
#cv2.imshow('orig',image_display)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24,24))
b, g, r = cv2.split(image)
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
image = cv2.merge((b,g,r))

#image_display = cv2.resize(image,(1024,1048))
#cv2.imshow('clahe',image_display)
#cv2.imshow('orig',image)
=======
image = cv2.imread('exemplos/exemplo9.jpeg')
image = cv2.resize(image,(1024,1048))
cv2.imshow('orig',image)
>>>>>>> upstream/main
# cv2.waitKey(0)

dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 3, 15) 
# Plotting of source and destination image 

#imagedst_display = cv2.resize(dst,(1024,1048))
#cv2.imshow('dst',imagedst_display)

grayDST = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray)
# cv2.waitKey(0)
#threshold

#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6) #imgf contains Binary image
#cv2.imshow('thresh',thresh)

threshDST = cv2.adaptiveThreshold(grayDST,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,6) #imgf contains Binary image



<<<<<<< HEAD
cv2.imwrite(filename.split('.')[0]+'_ocr.jpg',threshDST)
#threshDST_display = cv2.resize(threshDST,(1024,1048))
#cv2.imshow('threshDST',threshDST_display)
#cv2.imshow('threshDST',threshDST)

=======
threshDST = cv2.adaptiveThreshold(grayDST,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6) #imgf contains Binary image
cv2.imshow('threshDST',threshDST)
cv2.imwrite('exemplo9.jpg',threshDST)
>>>>>>> upstream/main
# img = cv2.imread('j.png',0)
# kernel = np.ones((1,1),np.uint8)
# erosion = cv2.erode(threshDST,kernel,iterations = 1)
# cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()